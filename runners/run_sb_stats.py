import pathlib
import os
import subprocess
import random
import contextlib
import csv
import json
import time
import xml.etree.ElementTree as ET

import numpy

import worlds.science_birds as sb
from agent.hydra_agent import HydraAgent
import settings

SB_DATA_PATH = pathlib.Path(settings.ROOT_PATH) / 'data' / 'science_birds'
SB_BIN_PATH = pathlib.Path(settings.SCIENCE_BIRDS_BIN_DIR) / 'linux'
SB_CONFIG_PATH = SB_DATA_PATH / 'config'
ANU_LEVELS_PATH = SB_DATA_PATH / 'ANU_Levels.tar.gz'
STATS_PATH = pathlib.Path(__file__).parent / 'stats.json'

NOVELTY = 0
TYPE = 1
SAMPLES = 2

def extract_levels(source, destination=None):
    ''' Extract ANU levels. '''

    if destination is None:
        destination = source.parent / os.path.splitext(source.stem)[0]

    if not destination.exists():
        os.mkdir(destination)
        cmd = "tar xzf {} -C {}".format(source, destination)
        subprocess.run(cmd, shell=True)

    return destination


def prepare_config(config_template, config_path, levels):
    ''' Prepare a configuration file from a config template and a set of level paths. '''

    tree = ET.parse(config_template)
    xpath = './trials/trial/game_level_set'
    level_set = tree.getroot().find(xpath)

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
    if len(difference) == 1:
        return difference.pop()

    return None


@contextlib.contextmanager
def run_hydra(config):
    ''' Run science birds and the hydra agent. '''
    try:
        env = sb.ScienceBirds(None, launch=True, config=config)
        yield env
        hydra = HydraAgent(env)
        hydra.main_loop()
    finally:
        env.kill()


def compute_stats(results_path):
    ''' Inspect evaluation directory from science birds and generate a stats dict. '''
    stats = {'levels': [], 'overall': None}

    passed = 0
    failed = 0
    scores = []

    evaluation_data = list(results_path.glob('*_EvaluationData.csv'))
    if len(evaluation_data) == 1:
        evaluation_data = evaluation_data[0]
        with open(evaluation_data) as f:
            data = csv.DictReader(f)
            for row in data:
                status = row['LevelStatus']
                if 'Pass' in status:
                    passed += 1
                else:
                    failed += 1

                score = float(row['Score'])
                scores.append(score)

                level_stats = {'level': row['levelName'],
                               'score': score,
                               'status': status,
                               'birds_remaining': int(row['birdsRemaining']),
                               'birds_start': int(row['birdsAtStart']),
                               'pigs_remaining': int(row['pigsRemaining']),
                               'pigs_start': int(row['pigsAtStart'])}
                stats['levels'].append(level_stats)

    stats['overall'] = {'passed': passed, 'failed': failed,
                        'avg_score': 0 if len(scores) == 0 else numpy.average(scores)}

    return stats


def run_sb_stats():
    ''' Run science birds agent stats. '''
    config_name = 'stats_config.xml'

    extracted = extract_levels(ANU_LEVELS_PATH)

    levels = list(extracted.glob('Levels/novelty_level_{}/type{}/Levels/*.xml'.format(NOVELTY, TYPE)))
    levels = random.sample(levels, SAMPLES)

    template = SB_CONFIG_PATH / 'test_config.xml'
    config = SB_CONFIG_PATH / config_name
    prepare_config(template, config, levels)

    pre_directories = glob_directories(SB_BIN_PATH, 'Agent*')
    post_directories = None

    with run_hydra(config_name) as env:
        post_directories = glob_directories(SB_BIN_PATH, 'Agent*')

    results = diff_directories(pre_directories, post_directories)

    if results is not None:
        stats = compute_stats(results)

        with open(STATS_PATH, 'w') as f:
            json.dump(stats, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    random.seed(2)
    run_sb_stats()

