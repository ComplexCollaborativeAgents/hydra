
# from agent.planning.nyx.PDDL import PDDL_Parser
from agent.planning.nyx.planner import Planner
import agent.planning.nyx.syntax.constants as constants
import settings
import sys
import time
import os, shutil
from datetime import datetime
import gc
sys.dont_write_bytecode = True



def process_arguments(cl_arguments):

    for arg in cl_arguments:

        if arg == '-h':
            print(constants.HELP_TEXT)
            exit(1)
        elif arg == '-val':
            constants.VALIDATE = True
            continue
        elif arg == '-p':
            constants.PRINT_ALL_STATES = True
            continue
        elif arg == '-noplan':
            constants.NO_PLAN = True
            continue
        elif arg == '-vv':
            constants.VERY_VERBOSE_OUTPUT = True
            continue
        elif arg == '-v':
            constants.VERBOSE_OUTPUT = True
            continue

        arg_list = arg.split(':')

        if arg_list[0] == '-t':
            constants.DELTA_T = float(arg_list[1])
            constants.TIME_PASSING_ACTION.duration = constants.DELTA_T
        elif arg_list[0] == '-th':
            constants.TIME_HORIZON = float(arg_list[1])
        elif arg_list[0] == '-pi':
            constants.PRINT_INFO = float(arg_list[1])
        elif arg_list[0] == '-to':
            constants.TIMEOUT = float(arg_list[1])
        elif arg_list[0] == '-search':
            constants.SEARCH_BFS = False
            if arg_list[1] == 'bfs':
                constants.SEARCH_BFS = True
                constants.SEARCH_ALGO_TXT = "BFS"
            elif arg_list[1] == 'dfs':
                constants.SEARCH_DFS = True
                constants.SEARCH_ALGO_TXT = "DFS"
            elif arg_list[1] == 'gbfs':
                constants.SEARCH_GBFS = True
                constants.SEARCH_ALGO_TXT = "GBFS"
            elif arg_list[1] == 'astar':
                constants.SEARCH_ASTAR = True
                constants.TRACK_G = True
                constants.SEARCH_ALGO_TXT = "A*"
            else:
                print('\nERROR: Unrecognized Heuristic Argument\nCall with -h flag for help')
                exit(1)
        else:
            print('\nERROR: Unrecognized Argument\nCall with -h flag for help')
            exit(1)

def print_config(dom, prob, pla):
    config_string = '\n\n===== NYX Planning Configuration ================\n' \
        '\n\t* domain: ' + str(dom) + \
        '\n\t* problem: ' + str(prob) + \
        '\n\t* plan: ' + str(pla) + \
        '\n\t* search algorithm: ' + str(constants.SEARCH_ALGO_TXT) + \
        '\n\t* time discretisation: ' + str(constants.DELTA_T) + \
        '\n\t* time horizon: ' + str(constants.TIME_HORIZON) + \
        ''
    print(config_string)

def print_solution_info(pln, plnr, ttime):

    non_temporal_count = 0

    for pai in pln:
        if not pai[0] == constants.TIME_PASSING_ACTION:
            non_temporal_count += 1

    config_string = '\n===== Solution Info =============================\n' \
        '\n\t* time: ' + str(round(ttime,3)) + \
        '\n\t* explored states: ' + str(plnr.explored_states) + \
        '\n\t* plan length: ' + str(non_temporal_count) + ' (' + str(len(pln)) + ')' + \
        '\n\t* plan duration: ' + str(plnr.reached_goal_state.time)
    print(config_string)


def runner(dom_file, prob_file, args_list: []):
    start_time = time.time()
    domain = dom_file
    problem = prob_file
    plan_file = os.path.dirname(prob_file) + "/plan_" + os.path.basename(prob_file)

    process_arguments(args_list)

    print_config(domain, problem, plan_file)

    my_plnr = Planner()
    goal_state = my_plnr.solve(domain, problem)

    total_time = time.time() - start_time

    my_plan = []

    if my_plnr.reached_goal_state is not None:
        my_plan = my_plnr.get_trajectory(my_plnr.reached_goal_state)
        print_solution_info(my_plan, my_plnr, total_time)
    else:
        print('\n=================================================\n')
        print('\tNo Plan Found!')
        print('\t\tTime: ' + str(round(total_time,3)))
        print('\t\tStates Explored: ' + str(my_plnr.explored_states))
        print('\n=================================================\n')
        # shutil.copy(prob_file, os.path.dirname(prob_file)+"/trace/problems/cartpole_prob_" +
        #             str(len([name for name in os.listdir(os.path.dirname(prob_file)+"/trace/problems/") if os.path.isfile(os.path.join(os.path.dirname(prob_file)+"/trace/problems/", name))])) + ".pddl")

    # if constants.NO_PLAN:
    #     print('\n=================================================\n')
    #     exit(1)

    if not constants.NO_PLAN:
        print('\n===== Plan ======================================\n')

    count = 0

    if constants.VERY_VERBOSE_OUTPUT and not constants.NO_PLAN:
        print('\nInitial State:')
        print(my_plnr.initial_state)
        print('')


    open(plan_file, 'w').close()

    plan_f = open(plan_file, 'a')

    for pair in my_plan:

        if (not (constants.VERBOSE_OUTPUT or constants.VERY_VERBOSE_OUTPUT)) and pair[0] == constants.TIME_PASSING_ACTION:
            continue

        # print('' + str("{:10.3f}".format(my_plnr.visited_hashmap[pair[1].predecessor_hashed].state.time)) + ':\t' + str(pair[0].name), end='')
        # print(str(pair[0].parameters), end='') if pair[0].parameters else print('', end='')
        # print('\t[' + str(pair[0].duration)+']')

        str1 = '' + str("{:10.3f}".format(my_plnr.visited_hashmap[pair[1].predecessor_hashed].state.time)) + ':\t' + str(pair[0].name)
        str2 = str(pair[0].parameters).replace('\'', '').replace(',)',')') if pair[0].parameters else ''
        str3 = '\t[' + str(pair[0].duration) + ']'

        if not constants.NO_PLAN:
            print(str1, end='')
            print(str2, end='')
            print(str3)

        plan_f.write(str1 + str2.replace(',','').replace('(',' ').replace(')','') + str3 + '\n')

        if constants.VERY_VERBOSE_OUTPUT:
            if not constants.NO_PLAN:
                print(str(pair[1]) + '\n')
            plan_f.write(str(pair[1]) + '\n')
        count += 1

    print('\n=================================================\n')

    plan_f.close()


    del my_plnr
    gc.collect()

#-----------------------------------------------
# Main
#-----------------------------------------------
if __name__ == '__main__':
    # runner(sys.argv[1], sys.argv[2], sys.argv[3:])
    runner("%s/cartpole_domain.pddl" % str(settings.CARTPOLE_PLANNING_DOCKER_PATH),
               "%s/cartpole_prob_35.pddl" % (str(settings.CARTPOLE_PLANNING_DOCKER_PATH)+"/trace/problems"),
               ['-search:gbfs', '-th:4', '-t:%s' % str(settings.CP_DELTA_T)])
