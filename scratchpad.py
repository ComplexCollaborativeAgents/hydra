import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

import settings
from agent.planning.nyx.nyx import runner

# prob_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/r01.pddl'
# dom_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/cartpole.pddl'

# runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])
# runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])
from runners.run_sb_stats import AgentType
from settings import EXPERIMENT_NAME

experiment_names = ['bfs0', 'dfs0', 'gbfs2', 'gbfs5', 'gbfs11'] # , 'helpful_actions']

folder = 'runners/latest_data/working heuristic' #25 levels'

NOVELTY = 0
level_nums = [222, 225, 236, 245, 246, 253, 254, 257, 555]  # 226,243, 252,

novelties = {NOVELTY: level_nums}

overall_stats = dict()

for experiment_name in experiment_names:
    overall_stats[experiment_name] = dict()

for novelty, types in novelties.items():
    for novelty_type in types:

        for experiment_name in experiment_names:

            filename = "stats_{}_novelty{}_type{}_agent{}".format(experiment_name, novelty, novelty_type,
                                                                  AgentType.RepairingHydra.name)
            filename = "{}/{}.json".format(folder, filename)
            with open(filename, 'r') as jf:
                stats = json.load(jf)
                stats_per_level = {'passed': 0, 'solved': 0, 'default shot': 0}
                sum_time = 0
                nodes_opened = []
                for entry in stats['levels']:
                    if entry.get('Default action used'):
                        stats_per_level['default shot'] += 1
                    else:
                        stats_per_level['solved'] += 1
                        if entry['status'] == 'Pass':
                            stats_per_level['passed'] += 1
                    if entry.get('planning times'):
                        sum_time += sum(entry['planning times']) / sum(entry['birds'].values())
                        nodes_opened.append(sum(entry['expanded nodes']) / sum(entry['birds'].values()))
                # stats_per_level['avg planning time'] = sum_time / len(stats['levels'])
                if sum_time != 0:
                    stats_per_level['median expanded nodes'] = np.median(nodes_opened)
                    stats_per_level['avg nodes per second'] = sum(nodes_opened) / sum_time
                else:
                    stats_per_level['expanded nodes'] = 0
                    stats_per_level['avg nodes per second'] = 0
                overall_stats[experiment_name][novelty_type] = stats_per_level

plt.title('Levels solved and passed ')
print('passed that were also solved\n, ' + ','.join(str(lev) for lev in level_nums))
for experiment_name in experiment_names:
    passed = [overall_stats[experiment_name][i]['passed'] for i in level_nums]
    print(experiment_name + ', ' + ','.join(str(lev) for lev in passed))
    plt.plot([str(t) for t in level_nums], [passed[i] for i in range(len(passed))], '*')
ax = plt.gca()
ax.set_xlabel('level type')
ax.legend(experiment_names)
plt.show()


plt.figure()
plt.title('Levels solved')
print('\nsolved\n, ' + ','.join(str(lev) for lev in level_nums))
for experiment_name in experiment_names:
    solved = [overall_stats[experiment_name][i]['solved'] for i in level_nums]
    print(experiment_name + ', ' + ','.join(str(lev) for lev in solved))
    plt.plot([str(t) for t in level_nums], [solved[i] for i in range(len(solved))], '*')
ax = plt.gca()
ax.set_xlabel('level type')
ax.legend(experiment_names)
plt.show()


plt.figure()
plt.title('Median nodes expanded')
print('\nmedian nodes expanded\n, ' + ','.join(str(lev) for lev in level_nums))
for experiment_name in experiment_names:
    expanded = [overall_stats[experiment_name][i]['median expanded nodes'] for i in level_nums]
    print(experiment_name + ', ' + ','.join(str(lev) for lev in expanded))
    plt.plot([str(t) for t in level_nums], [expanded[i] for i in range(len(expanded))], '*')
ax = plt.gca()
ax.set_xlabel('level type')
ax.legend(experiment_names)
plt.show()


plt.figure()
plt.title('Nodes per second')
print('\nnodes per second\n, ' + ','.join(str(lev) for lev in level_nums))
for experiment_name in experiment_names:
    ex_per_sec = [overall_stats[experiment_name][i]['avg nodes per second'] for i in level_nums]
    print(experiment_name + ', ' + ','.join(str(lev) for lev in ex_per_sec))
    plt.plot([str(t) for t in level_nums], [ex_per_sec[i] for i in range(len(ex_per_sec))], '*')
ax = plt.gca()
ax.set_xlabel('level type')
ax.legend(experiment_names)
plt.show()


# plt.plot([str(t) for t in level_nums], [passed[i] / 50 for i in range(len(passed))], 'b*',
#          # [str(t) for t in TYPE], [overall_stats[i]['avg planning time']/30 for i in TYPE], 'y.',
#          [str(t) for t in level_nums], [overall_stats[i]['solved'] / 50 for i in level_nums], 'g^')
# ax = plt.gca()
# ax.set_xlabel('level type')
# ax.set_ylabel('percent of max value')
# ax.legend(['pass %', 'planning time %', 'default shot %'])
# plt.figtext(0.5, 0, f'overall levels passed: {sum(passed)}', wrap=True, horizontalalignment='center')
# plt.title(experiment_name)
# print([str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE])
# plt.figure()
# plt.plot(
#          [str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE], 'r*')
# plt.figure()
# plt.plot([str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE], '.')
# plt.show()





