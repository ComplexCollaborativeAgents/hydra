from os import listdir, path
import pickle
import random

from agent.consistency.episode_log import ScienceBirdsObservation
from agent.sb_hydra_agent import RepairingSBHydraAgent

# SB_NON_NOVEL_OBS_DIR = '/home/klenk/Downloads/non_novel/'
# SB_NOVEL_OBS_DIR = '/home/klenk/Downloads/novel/'
#
# SB_NON_NOVEL_TESTS = listdir(SB_NON_NOVEL_OBS_DIR)
# SB_NOVEL_TESTS = listdir(SB_NOVEL_OBS_DIR)

# if __name__ == '__main__':
#     hydra = RepairingHydraSBAgent()
#     ret = []
#     false_positives = 0
#     non_novel = random.choices(SB_NON_NOVEL_TESTS,k=90)
#     for ob_file in non_novel:
#         sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))
#         if hydra.should_repair(sb_ob):
#             false_positives += 1
#         ret.append(hydra.novelty_likelihood)
#     print('Current Novelty Likelihood {}'.format(ret[-1]))
#     print('false positives: {}'.format(false_positives))
#
#     false_negatives = 0
#     novel = random.choices(SB_NOVEL_TESTS,k=90)
#     for ob_file in novel:
#         sb_ob : ScienceBirdsObservation = pickle.load(open(path.join(SB_NON_NOVEL_OBS_DIR, ob_file), "rb"))
#         if not hydra.should_repair(sb_ob):
#             false_negative += 1
#         ret.append(hydra.novelty_likelihood)
#     print('Current Novelty Likelihood {}'.format(ret[-1]))
#     print('false positives: {}'.format(false_positives))
#     print(ret)
import re
import sys



def obs_prob(line):
    nums = re.findall('[0-9.]+', line)
    if nums:
        return float(nums[-3])

def last_prob(line):
    nums = re.findall('[0-9.]+', line)
    if nums:
        return float(nums[-2])

def second_last_prob(line):
    nums = re.findall('[0-9.]+', line)
    if nums:
        return float(nums[-1])


def solved(line):
    return 'True' in line if len(line) > 20 else None


def summarize_detections(fname,thresholds=[.5,.55,.6,.65,.7,.75]):
    obs_probs = []
    last_probs = []
    sols = []
    with open(fname,'r') as fin:
        lines = fin.readlines()
        last_probs.append(second_last_prob(lines[0]))
        for line in lines:
            o = obs_prob(line)
            if o:
                obs_probs.append(o)
            o = last_prob(line)
            if o:
                last_probs.append(o)
            o = solved(line)
            if o is not None:
                sols.append(o)
    for threshold in thresholds:
        obs_positive_rate = sum(1 for i in obs_probs if i and i > threshold)/len(obs_probs)
        probs_positive_rate = sum(1 for i in last_probs if i and i > threshold)/len(last_probs)
        failure_rate = sum(1 for i in sols if not i )/len(sols)
        print('Detection rate for threshold {}: {}'.format(threshold, obs_positive_rate * probs_positive_rate * probs_positive_rate * failure_rate))


if __name__ == '__main__':
    print('non_novel')
    summarize_detections('/home/klenk/tmp/non_novel_w_last.txt')
    print('\n\t2_6')
    summarize_detections('/home/klenk/tmp/level_2_6.txt')
    print('\n\t2_7')
    summarize_detections('/home/klenk/tmp/level_2_7.txt')
    print('\n\t2_8')
    summarize_detections('/home/klenk/tmp/level_2_8.txt')
    print('\n\t2_9')
    summarize_detections('/home/klenk/tmp/level_2_9.txt')
    print('\n\t2_10')
    summarize_detections('/home/klenk/tmp/level_2_10.txt')
    print('\n\t3_6')
    summarize_detections('/home/klenk/tmp/level_3_6.txt')
    print('\n\t3_7')
    summarize_detections('/home/klenk/tmp/level_3_7.txt')