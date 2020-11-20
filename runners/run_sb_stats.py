import pathlib
import os
import subprocess
import random
import contextlib
import csv
import json
import enum
import time
import collections
import xml.etree.ElementTree as ET
from typing import Optional

import numpy
from worlds.science_birds_interface.demo.naive_agent_groundtruth import ClientNaiveAgent

import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
from agent.repairing_hydra_agent import RepairingHydraSBAgent
import settings

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
TYPE = 2
SAMPLES = 1
AGENT = AgentType.Baseline

def extract_levels(source, destination=None):
    ''' Extract ANU levels. '''

    if destination is None:
        destination = source.parent / os.path.splitext(source.stem)[0]

    if not destination.exists():
        os.mkdir(destination)
        cmd = "tar xzf {} -C {}".format(source, destination)
        subprocess.run(cmd, shell=True)

    return destination


def prepare_config(config_template, config_path, levels, notify_novelty):
    ''' Prepare a configuration file from a config template and a set of level paths. '''

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
    ''' List directories from base path which conform to pattern. '''
    return list(p for p in base_path.glob(pattern) if p.is_dir())


def diff_directories(a, b):
    ''' Return the difference between two lists of directories. '''
    if b is None:
        return None

    difference = set(b) - set(a)
    difference = {d for d in difference if any(d.iterdir())}
    if len(difference) == 1:
        return difference.pop()

    return None


@contextlib.contextmanager
def run_agent(config, agent, agent_stats=list()):
    ''' Run science birds and the hydra agent. '''
    try:
        env = sb.ScienceBirds(None, launch=True, config=config)
        yield env

        if agent == AgentType.Hydra:
            hydra = HydraAgent(env, agent_stats)
            hydra.main_loop(max_actions=10000)
        elif agent == AgentType.RepairingHydra:
            hydra = RepairingHydraSBAgent(env, agent_stats)
            hydra.main_loop(max_actions=10000)
        elif agent == AgentType.Baseline:
            ground_truth = ClientNaiveAgent(env.id, env.sb_client)
            ground_truth.run()
        elif agent == AgentType.Datalab:
            datalab = sb.DatalabAgent()
            datalab.run()
        elif agent == AgentType.Eaglewings:
            eaglewings = sb.EaglewingsAgent()
            eaglewings.run()
    finally:
        env.kill()


def get_bird_count(level_path):
    ''' Given the path to a level XML, return a dictionary with bird count per type of bird. '''
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
        birds_section = content[start:end+8]

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

def compute_stats(results_path, agent, agent_stats = list()):
    ''' Inspect evaluation directory from science birds and generate a stats dict. '''
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

                level_stats = {'level': level_path,
                               'score': score,
                               'status': status,
                               'birds_remaining': int(row['birdsRemaining']),
                               'birds_start': int(row['birdsAtStart']),
                               'pigs_remaining': int(row['pigsRemaining']),
                               'pigs_start': int(row['pigsAtStart']),
                               'birds': birds}

                # Get agent stats
                if i<len(agent_stats):
                    agent_stats_for_level = agent_stats[i]
                    for key in agent_stats_for_level:
                        level_stats[key]= agent_stats_for_level[key]

                stats['levels'].append(level_stats)

    stats['overall'] = {'passed': passed, 'failed': failed,
                        'avg_score': 0 if len(scores) == 0 else numpy.average(scores),
                        'agent': agent.name,
                        'birds': bird_scores}

    return stats


def run_sb_stats(extract=False, seed=None):
    ''' Run science birds agent stats. '''
    novelties = {NOVELTY: [TYPE]}
    run_performance_stats(novelties, agent_type=AGENT, seed=seed, samples=SAMPLES)


def run_performance_stats(novelties: dict,
                          agent_type: AgentType,
                          seed: int = None,
                          samples: Optional[int] = SAMPLES,
                          notify_novelty: Optional[bool] = None,
                          bin_path: pathlib.Path = SB_BIN_PATH,
                          levels_path: pathlib.Path = SB_BIN_PATH,
                          stats_base_path: pathlib.Path = STATS_BASE_PATH,
                          template: pathlib.Path = SB_CONFIG_PATH / 'test_config.xml',
                          config: pathlib.Path = SB_CONFIG_PATH / 'stats_config.xml'):
    ''' Run science birds agent stats. '''
    if seed is not None:
        random.seed(seed)

    for novelty, types in novelties.items():
        for novelty_type in types:
            pattern = 'Levels/novelty_level_{}/type{}/Levels/*.xml'.format(novelty, novelty_type)
            levels = list(levels_path.glob(pattern))

            number_samples = len(levels)
            if samples is not None:
                number_samples = min(number_samples, samples)
            levels = random.sample(levels, number_samples)

            prepare_config(template, config, levels, notify_novelty)
            pre_directories = glob_directories(bin_path, 'Agent*')
            post_directories = None

            agent_stats = list()
            with run_agent(config.name, agent_type, agent_stats) as env: # TODO: Typo?
                post_directories = glob_directories(SB_BIN_PATH, 'Agent*')

            results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is None:
                post_directories = glob_directories(SB_BIN_PATH, 'Agent*')
                results_directory = diff_directories(pre_directories, post_directories)

            if results_directory is not None:
                stats = compute_stats(results_directory, agent_type, agent_stats)
                filename = "stats_novelty{}_type{}_agent{}.json".format(novelty, novelty_type, agent_type.name)
                with open(stats_base_path / filename, 'w') as f:
                    json.dump(stats, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    run_sb_stats(seed=0)

