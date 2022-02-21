import json
import pathlib
import matplotlib.pyplot as plt

import settings
from agent.planning.nyx.nyx import runner

# prob_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/r01.pddl'
# dom_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/cartpole.pddl'

# runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])
# runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])
from runners.run_sb_stats import EXPERIMENT_NAME, AgentType

experiment_name = 'bfs2'

folder = 'runners/latest_data'

NOVELTY = 0
TYPE = [222, 225, 226, 236, 243, 245, 246,  252, 253, 254, 257, 555]

novelties = {NOVELTY: TYPE}

overall_stats = dict()

for novelty, types in novelties.items():
    for novelty_type in types:

        filename = "stats_{}_novelty{}_type{}_agent{}".format(experiment_name, novelty, novelty_type,
                                                              AgentType.RepairingHydra.name)
        filename = "{}/{}.json".format(folder, filename)
        with open(filename, 'r') as jf:
            stats = json.load(jf)
            stats_per_level = {'won': stats['overall']['passed'], 'lost': stats['overall']['failed'], 'default shot': 0}
            sum_time = 0
            sum_nodes = 0
            for entry in stats['levels']:
                if entry.get('Default action used'):
                    stats_per_level['default shot'] += 1
                if entry.get('planning times'):
                    sum_time += sum(entry['planning times']) / sum(entry['birds'].values())
                    sum_nodes += sum(entry['expanded nodes']) / sum(entry['birds'].values())
            stats_per_level['avg planning time'] = sum_time / len(stats['levels'])
            if sum_time != 0:
                stats_per_level['avg nodes per second'] = sum_nodes / sum_time
            else:
                stats_per_level['avg nodes per second'] =0
            overall_stats[novelty_type] = stats_per_level

passed = [overall_stats[i]['won'] for i in TYPE]

plt.plot([str(t) for t in TYPE], [passed[i]/50 for i in range(len(passed))], 'b*',
         [str(t) for t in TYPE], [overall_stats[i]['avg planning time']/30 for i in TYPE], 'y.',
         [str(t) for t in TYPE], [overall_stats[i]['default shot']/50 for i in TYPE], 'g^')
ax = plt.gca()
ax.set_xlabel('level type')
ax.set_ylabel('percent of max value')
ax.legend(['pass %', 'planning time %', 'default shot %'])
plt.figtext(0.5, 0, f'overall levels passed: {sum(passed)}', wrap=True, horizontalalignment='center')
plt.title(experiment_name)
print([str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE])
plt.figure()
plt.plot(
         [str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE], 'r*')
# plt.figure()
# plt.plot([str(t) for t in TYPE], [overall_stats[i]['avg nodes per second'] for i in TYPE], '.')
plt.show()





