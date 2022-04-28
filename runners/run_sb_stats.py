import pathlib
import os
import subprocess
import random
import contextlib
import csv
import json
import enum
import collections
import time
import xml.etree.ElementTree as ET
from typing import Optional

import numpy

from settings import EXPERIMENT_NAME
from worlds.science_birds_interface.demo.naive_agent_groundtruth import ClientNaiveAgent

import worlds.science_birds as sb
from agent.sb_hydra_agent import SBHydraAgent, RepairingSBHydraAgent
import settings
import logging
from agent.planning.nyx.syntax import constants

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_sb_states")

SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
ANU_LEVELS_PATH = SB_DATA_PATH / 'ANU_Levels.tar.gz'
STATS_BASE_PATH = pathlib.Path(__file__).parent.absolute()


class AgentType(enum.Enum):
    RepairingHydra = 0
    Hydra = 1
    Baseline = 2
    Datalab = 3
    Eaglewings = 4


NOVELTY = 0
TYPE = [246] # [226, 243, 244, 252] #[222, 225, 236, 245, 246, 253, 254, 257, 555]  #245 555, 222, 225, 226, 236, 243, 252,  , 257
NOVELTY_SET = {0: [226, 227]}  # 11:[50, 130], 12:[30, 110]  13: [20, 90], 14:[1]
SAMPLES = 3

AGENT = AgentType.RepairingHydra


def extract_levels(source, destination=None):
    """ Extract ANU levels. """

    if destination is None:
        destination = source.parent / os.path.splitext(source.stem)[0]

    if not destination.exists():
        os.mkdir(destination)
        cmd = "tar xzf {} -C {}".format(source, destination)
        subprocess.run(cmd, shell=True)

    return destination


def prepare_config(config_template, config_path, levels, notify_novelty):
    """ Prepare a configuration file from a config template and a set of level paths. """

    tree = ET.parse(config_template)

    if notify_novelty is not None:
        xpath = './trials/trial'
        trial = tree.getroot().find(xpath)
        trial.set('notify_novelty', str(notify_novelty))

    xpath = './trials/trial/game_level_set'
    level_set = tree.getroot().find(xpath)
    level_set.set('time_limit', '500000')
    level_set.set('total_interaction_limit', '1000000')

    for child in list(level_set):
        level_set.remove(child)

    for level in levels:
        relpath = os.path.relpath(level, SB_BIN_PATH)
        ET.SubElement(level_set, 'game_levels', level_path=relpath)

    tree.write(config_path)


def glob_directories(base_path, pattern='*'):
    """ List directories from base path which conform to pattern. """
    return list(p for p in base_path.glob(pattern) if p.is_dir())


def diff_directories(a, b):
    """ Return the difference between two lists of directories. """
    if b is None:
        return None

    difference = set(b) - set(a)
    difference = [d for d in difference if any(d.iterdir())]
    if len(difference) == 1:
        return difference.pop()
    elif len(difference) > 1:
        difference.sort(key=lambda d: time.ctime(os.path.getctime(d)), reverse=True)
        return difference[0]
    else:
        return None


@contextlib.contextmanager
def run_agent(config, agent, agent_stats=None):
    """ Run science birds and the hydra agent. """
    if agent_stats is None:
        agent_stats = list()
    env = sb.ScienceBirds(None, launch=True, config=config)

    if agent == AgentType.Hydra:
        hydra = SBHydraAgent(env, agent_stats)
        hydra.main_loop(max_actions=10000)
    elif agent == AgentType.RepairingHydra:
        hydra = RepairingSBHydraAgent(env, agent_stats)
        hydra.main_loop(max_actions=10000)
    elif agent == AgentType.Baseline:
        naive_config = collections.namedtuple(
            'NaiveConfig', ['save_logs', 'agent_host', 'agent_port', 'observer_host', 'observer_port'])
        naive_config.agent_host = env.sb_client.server_host
        naive_config.agent_port = env.sb_client.server_port
        naive_config.observer_host = None
        naive_config.observer_port = None
        ground_truth = ClientNaiveAgent(str(env.id), naive_config)
        ground_truth.run()
    elif agent == AgentType.Datalab:
        datalab = sb.DatalabAgent()
        datalab.run()
    elif agent == AgentType.Eaglewings:
        eaglewings = sb.EaglewingsAgent()
        eaglewings.run()
    # finally:
    env.kill()


def get_object_count(level_path="{}/Levels/novelty_level_0/type2/Levels/00501_0_0_4_0.xml".format(SB_BIN_PATH)):
    with open(level_path, 'rb') as fp:
        content = fp.read()
        try:
            content = content.decode('ascii')
        except UnicodeError:
            content = content.decode('utf-16-le')

        lines = content.splitlines()
        n = 0
        for line in lines:
            # print(line)
            if '<GameObjects>' in line:
                start = n
            if '</GameObjects>' in line:
                stop = n
            n = n + 1
        num_game_objects = stop - start
        return num_game_objects


def get_bird_count(level_path):
    """ Given the path to a level XML, return a dictionary with bird count per type of bird. """
    birds = collections.defaultdict(int)
    birds_section = ''
    with open(level_path, 'rb') as f:
        content = f.read()
        try:
            content = content.decode('ascii')
        except UnicodeError:
            content = content.decode('utf-16-le')
        start = content.find('<Birds>')
        end = content.find('</Birds>')
        birds_section = content[start:end + 8]

    try:
        tree = ET.fromstring(birds_section)
        for elem in tree:
            if elem.tag == 'Bird':
                bird_type = elem.attrib.get('type', None)
                if bird_type is not None:
                    birds[bird_type] += 1
    except Exception:
        pass
    return birds


def compute_stats(results_path, agent, agent_stats=list()):
    """ Inspect evaluation directory from science birds and generate a stats dict. """
    stats = {'levels': [], 'overall': None}

    passed = 0
    failed = 0
    scores = []
    bird_scores = collections.defaultdict(lambda: {"passed": 0, "failed": 0})

    evaluation_data = list(results_path.glob('*_EvaluationData.csv'))
    if len(evaluation_data) == 1:
        evaluation_data = evaluation_data[0]
        with open(evaluation_data) as f:
            data = csv.DictReader(f)
            for i, row in enumerate(data):
                level_path = row['levelName']
                birds = get_bird_count(os.path.join(SB_BIN_PATH, level_path))
                status = row['LevelStatus']
                if 'Pass' in status:
                    passed += 1
                    for bird in birds.keys():
                        bird_scores[bird]['passed'] += 1
                else:
                    failed += 1
                    status = 'Fail'
                    for bird in birds.keys():
                        bird_scores[bird]['failed'] += 1

                score = float(row['Score'])
                scores.append(score)

                num_objects = get_object_count(os.path.join(SB_BIN_PATH, level_path))

                level_stats = {'level': level_path,
                               'score': score,
                               'status': status,
                               'birds_remaining': int(row['birdsRemaining']),
                               'birds_start': int(row['birdsAtStart']),
                               'pigs_remaining': int(row['pigsRemaining']),
                               'pigs_start': int(row['pigsAtStart']),
                               'objects': num_objects,
                               'birds': birds}

                # Get agent stats
                if i < len(agent_stats):
                    agent_stats_for_level = agent_stats[i]
                    for key in agent_stats_for_level:
                        level_stats[key] = agent_stats_for_level[key]

                stats['levels'].append(level_stats)

    stats['overall'] = {'passed': passed, 'failed': failed,
                        'avg_score': 0 if len(scores) == 0 else numpy.average(scores),
                        'agent': agent.name,
                        'birds': bird_scores}

    return stats


def compute_eval_stats(results_path, agent, agent_stats=list()):
    """ Inspect evaluation directory from science birds and generate a stats dict. (2021 Eval) """
    stats = {'levels': [], 'overall': None}

    passed = 0
    failed = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    true_positives = 0
    scores = []
    bird_scores = collections.defaultdict(lambda: {"passed": 0, "failed": 0})

    evaluation_data = list(results_path.glob('*_EvaluationData.csv'))
    if len(evaluation_data) == 1:
        evaluation_data = evaluation_data[0]
        with open(evaluation_data) as f:
            data = csv.DictReader(f)
            for i, row in enumerate(data):
                level_path = row['levelName']
                birds = get_bird_count(os.path.join(SB_BIN_PATH, level_path))
                status = row['LevelStatus']
                if 'Pass' in status:
                    passed += 1
                    for bird in birds.keys():
                        bird_scores[bird]['passed'] += 1
                else:
                    failed += 1
                    status = 'Fail'
                    for bird in birds.keys():
                        bird_scores[bird]['failed'] += 1

                score = float(row['Score'])
                scores.append(score)

                # output if this level actually contained novelty TODO

                level_stats = {'level': level_path,
                               'score': score,
                               'status': status,
                               'birds_remaining': int(row['birdsRemaining']),
                               'birds_start': int(row['birdsAtStart']),
                               'pigs_remaining': int(row['pigsRemaining']),
                               'pigs_start': int(row['pigsAtStart']),
                               'birds': birds}

                # print("STATS: agent stats {}".format(agent_stats))

                # Get agent stats
                if i < len(agent_stats):
                    agent_stats_for_level = agent_stats[i]
                    for key in agent_stats_for_level:
                        level_stats[key] = agent_stats_for_level[
                            key]  # Novelty probability should be passed through here

                # Categorize novelty detection result
                if 'novelty_level_0' in level_path:  # Is not novel - TODO: find a better way to determine non novel levels
                    if 'novelty_likelihood' not in level_stats:
                        true_negatives += 1
                    else:
                        if level_stats[
                            'novelty_likelihood'] == 1:  # Detected novelty when there is none - false_positive
                            false_positives += 1
                        else:
                            true_negatives += 1
                else:  # This is a novel level
                    if 'novelty_likelihood' not in level_stats:
                        false_negatives += 1
                    else:
                        if level_stats['novelty_likelihood'] == 1:
                            true_positives += 1
                        else:
                            false_negatives += 1

                # Add to levels
                stats['levels'].append(level_stats)

    stats['overall'] = {
        'passed': passed,
        'failed': failed,
        'avg_score': 0 if len(scores) == 0 else numpy.average(scores),
        'agent': agent.name,
        'birds': bird_scores,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'true_positives': true_positives
    }

    print("STATS: stats for {} are {}".format(results_path, stats['overall']))

    return stats


def run_sb_stats(extract=False, seed=None, record_novelty_stats=False):
    """ Run science birds agent stats. """
    novelties = NOVELTY_SET
    samples = SAMPLES
    run_performance_stats(novelties, agent_type=AGENT, seed=seed, samples=samples,
                          record_novelty_stats=record_novelty_stats)


def run_performance_stats(novelties: dict,
                          agent_type: AgentType,
                          seed: Optional[int] = None,
                          samples: Optional[int] = SAMPLES,
                          notify_novelty: Optional[bool] = None,
                          suffix: Optional[str] = None,
                          bin_path: pathlib.Path = SB_BIN_PATH,
                          levels_path: pathlib.Path = SB_BIN_PATH,
                          stats_base_path: pathlib.Path = STATS_BASE_PATH,
                          template: pathlib.Path = SB_CONFIG_PATH / 'test_config.xml',
                          config: pathlib.Path = SB_CONFIG_PATH / 'stats_config.xml',
                          level_lookup: Optional[dict] = None,
                          record_novelty_stats=False):
    """ Run science birds agent stats. """
    if seed is not None:
        random.seed(seed)

    for novelty, types in novelties.items():
        for novelty_type in types:
            settings.NOVELTY_TYPE = novelty_type
            pattern = 'Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
            levels = list(levels_path.glob(pattern))

            number_samples = len(levels)
            if samples is not None:
                number_samples = min(number_samples, samples)
            levels = levels[20:20 + number_samples]
            # levels = random.sample(levels, number_samples)

            if level_lookup:
                levels = [levels_path / l for l in level_lookup[str(novelty)][str(novelty_type)]]

            prepare_config(template, config, levels, notify_novelty)
            pre_directories = glob_directories(bin_path, 'Agent*')
            post_directories = None

            agent_stats = list()
            logger.info('running agent!')
            run_agent(config.name, agent_type, agent_stats)  # TODO: Typo?
            post_directories = glob_directories(SB_BIN_PATH, 'Agent*')

            results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is None:
                post_directories = glob_directories(SB_BIN_PATH, 'Agent*')
                results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is not None:
                stats = compute_stats(results_directory, agent_type, agent_stats)
                filename = "stats_{}_novelty{}_type{}_agent{}".format(settings.EXPERIMENT_NAME, novelty, novelty_type,
                                                                      agent_type.name)
                if suffix is None or len(suffix) == 0:
                    current_suffix = ''
                else:
                    current_suffix = '_' + suffix
                filename = "{}{}.json".format(filename, current_suffix)
                with open(stats_base_path / filename, 'w') as f:
                    json.dump(stats, f, sort_keys=True, indent=4)


# def do_record_novelty_stats(novelty, novelty_type, config, agent_stats):
#
#     file = open("{}/novelty_detection/ensemble_learning.csv".format(SB_DATA_PATH), "a")
#     print(agent_stats)
#     line = "{},{},{},{},{},{}\n".format(novelty, novelty_type, agent_stats[0]['unknown_object'],
#                              agent_stats[0]['reward_estimator_likelihood'],
#                              agent_stats[0]['pddl_novelty_likelihood'],
#                              agent_stats[0]['novelty_detection'])
#     file.write(line)
#     file.close()


def run_eval_stats(novelties: dict,
                   agent_type: AgentType,
                   seed: Optional[int] = None,
                   samples: Optional[int] = None,
                   notify_novelty: Optional[bool] = None,
                   suffix: Optional[str] = None,
                   bin_path: pathlib.Path = SB_BIN_PATH,
                   levels_path: pathlib.Path = SB_BIN_PATH,
                   stats_base_path: pathlib.Path = STATS_BASE_PATH,
                   template: pathlib.Path = SB_CONFIG_PATH / 'test_config.xml',
                   config: pathlib.Path = SB_CONFIG_PATH / 'stats_config.xml',
                   level_lookup: Optional[dict] = None):
    """ Run science birds agent stats. """
    if seed is not None:
        random.seed(seed)

    results = []

    for novelty, types in novelties.items():
        for novelty_type in types:
            pattern = 'Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)

            if level_lookup:
                levels = [levels_path / l for l in level_lookup[str(novelty)][str(novelty_type)]]
            else:
                levels = list(levels_path.glob(pattern))

            number_samples = len(levels)
            if samples is not None:
                number_samples = min(number_samples, samples)
            sampled_levels = levels[:number_samples]
            # sampled_levels = random.sample(levels,number_samples)  TODO: Discuss design: this sampling kills the order of the levels, causing the non-novel levels to appear after the novel ones

            prepare_config(template, config, sampled_levels, notify_novelty)
            pre_directories = glob_directories(bin_path, 'Agent*')
            post_directories = None

            agent_stats = list()
            with run_agent(config.name, agent_type, agent_stats=agent_stats) as env:  # TODO: Typo?
                post_directories = glob_directories(SB_BIN_PATH, 'Agent*')

            results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is None:
                post_directories = glob_directories(SB_BIN_PATH, 'Agent*')
                results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is not None:
                results_for_novelty_type = compute_eval_stats(results_directory, agent_type, agent_stats=agent_stats)

                # Write intermediate results to file
                results_filename = "eval_novelty{}_type{}_agent{}".format(novelty, novelty_type, agent_type.name)
                with open(results_filename, "w+") as f:
                    logger.info(
                        "Writing results for novelty {} type {} to {}".format(novelty, novelty_type, results_filename))
                    json.dump(results_for_novelty_type, f, indent=3)

                results.append(results_for_novelty_type)

    # Output results per level
    results_filename = os.path.join(STATS_BASE_PATH, "eval_trial{}_results.json".format(suffix))
    print("Print evaluation results to file {}".format(results_filename))
    with open(results_filename, "w+") as f:
        logger.info("STATS: writing to {}".format(results_filename))
        json.dump(results, f, indent=3)

    # Output evaluation metrics for the entire run
    stat_results = _compute_stats(results, suffix)
    stats_filename = os.path.join(STATS_BASE_PATH, "eval_trial{}_stats.json".format(suffix))
    logger.info("Print evaluation metrics to file {}".format(stats_filename))
    with open(stats_filename, "w+") as f:
        logger.info("STATS: writing to {}".format(stats_filename))
        json.dump(stat_results, f, indent=4)

    return results


def _compute_stats(results, file_suffix):
    """ Compute the evaluation metrics for the given results. """

    stat_results = {}
    # M1: avg number of False Negatives among CDTs
    # M2: % of CDTs across all trials
    # Collect CDTs
    CDTs = []
    for result in results:
        print("STATS: Processing result: {}".format(result))
        if result['overall']['true_positives'] > 0 and result['overall']['false_positives'] == 0:
            CDTs.append(result)
        print("------------------------------------------------------------------------------------")
    print("STATS: CDTs are: {}".format(CDTs))
    # For every CDT, count false negatives and average
    sum_false_neg = sum([cdt['overall']['false_negatives'] for cdt in CDTs])
    if len(CDTs) > 0:
        stat_results['m1'] = sum_false_neg / len(CDTs)
    else:
        stat_results['m1'] = 0
    # Determine % of CDTs
    if len(results) > 0:
        stat_results['m2'] = len(CDTs) / len(results)
    else:
        stat_results['m2'] = 0
    # M2.1: % of Trials with at least 1 False Positive
    # Do 1 - % of CDTs
    trial_w_fp = 0
    for result in results:
        if result['overall']['false_positives'] > 0:
            trial_w_fp += 1
    if len(results) > 0:
        stat_results['m2.1'] = trial_w_fp / len(results)
    else:
        stat_results['m2.1'] = 0
    # M3 + M4: Ratio of agent post-novelty performance vs baseline agent pre-novelty performance (TODO: find pre performance records)
    # M5: Post novelty performance overall vs baseline agent
    # M6: Asymptotic performance vs baseline agent
    # M7: False positive rate and True positive rate

    # TODO: Implement these metrics

    return stat_results


if __name__ == '__main__':
    # run_sb_stats(seed=0, record_novelty_stats=True)

    # baseline_agents = [AgentType.Baseline, AgentType.Datalab, AgentType.Eaglewings]  #[] # ,
    # for agent in baseline_agents:
    #     AGENT = agent
    #     settings.EXPERIMENT_NAME = agent.name
    #     run_sb_stats(record_novelty_stats=True)

    ICAPS_benchmarks = [
        # ('bfs', '0'),
        # ('gbfs', '5'),
        # ('gbfs', '11'),
        # ('dfs', '0'),
        # ('gbfs', '2')
        #
    ]

    AGENT = AgentType.RepairingHydra
    constants.SB_W_HELPFUL_ACTIONS = False
    for alg, heuristic in ICAPS_benchmarks:
        settings.SB_ALGO_STRING = alg
        settings.SB_HEURISTIC_STRING = heuristic
        settings.EXPERIMENT_NAME = alg + heuristic
        run_sb_stats(record_novelty_stats=True)

    constants.SB_W_HELPFUL_ACTIONS = True
    settings.SB_ALGO_STRING = 'gbfs'
    settings.EXPERIMENT_NAME = 'helpful_actions'
    run_sb_stats(record_novelty_stats=True)

