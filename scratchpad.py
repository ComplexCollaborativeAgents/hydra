from agent.planning.nyx.nyx import runner

prob_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/r01.pddl'
dom_path = '/mnt/c/Users/yonsher/PycharmProjects/hydra/cartpole.pddl'

runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])
# runner(dom_path, prob_path, ['-vv', '-to:300', '-noplan', '-search:astar', '-custom_heuristic:7', '-th:4', '-t:0.02'])

# ===== NYX Planning Configuration ================ 4 seconds? Cartpole heuristic
#
# 	* domain: /mnt/c/Users/yonsher/PycharmProjects/hydra/cartpole.pddl
# 	* problem: /mnt/c/Users/yonsher/PycharmProjects/hydra/r01.pddl
# 	* plan: /mnt/c/Users/yonsher/PycharmProjects/hydra/plan_r01.pddl
# 	* search algorithm: GBFS
# 	* time discretisation: 0.02
# 	* time horizon: 4.0
#
# 	* model parse time: 0.0024s
# [  0.96] ==> states explored: 10000
# 			10414.47 states/sec
#
# ===== Solution Info =============================
#
# 	* time: 1.044
# 	* explored states: 10917
# 	* plan length: 6 (105)
# 	* plan duration: 1.98
#
# =================================================



# ===== NYX Planning Configuration ================ 1.5 seconds, interval heuristic
#
# 	* domain: /mnt/c/Users/yonsher/PycharmProjects/hydra/cartpole.pddl
# 	* problem: /mnt/c/Users/yonsher/PycharmProjects/hydra/r01.pddl
# 	* plan: /mnt/c/Users/yonsher/PycharmProjects/hydra/plan_r01.pddl
# 	* search algorithm: A* (BGFS is same)
# 	* time discretisation: 0.02
# 	* time horizon: 4.0
#
# 	* model parse time: 0.0028s
# [ 12.04] ==> states explored: 10000
# 			830.71 states/sec
#
# ===== Solution Info =============================
#
# 	* time: 17.947
# 	* explored states: 14627
# 	* plan length: 54 (128)
# 	* plan duration: 1.48
#
# =================================================
#
