import os
import sys
from typing import Optional, List

from agent.planning.nyx.PDDL import PDDL_Parser
from agent.planning.nyx.syntax import constants
from agent.planning.nyx.syntax.plan import Plan


def process_arguments(cl_arguments):

    for arg in cl_arguments:

        if arg == '-h':
            print(constants.SIMULATOR_HELP)
            exit(1)
        elif arg == '-vv':
            constants.VERY_VERBOSE_OUTPUT = True
            continue
        elif arg == '-v':
            constants.VERBOSE_OUTPUT = True
            continue
        elif arg == '-e':
            constants.EXPAND_TIME_PASSING = True
            continue
        elif arg == '-f':
            constants.FULL_TRACE = True
            continue
        elif arg == '-dblevent':
            constants.DOUBLE_EVENT_CHECK = True
            continue

        arg_list = arg.split(':')

        if arg_list[0] == '-t':
            constants.set_delta_t(float(arg_list[1]))
        else:
            print('\nERROR: Unrecognized Argument\nCall with -h flag for help')
            exit(1)


def simulator(domain_file: str, problem_file: str, plan_file: str,
              args_list: Optional[List[str]] = None):
    if args_list:
        process_arguments(args_list)

    trace_file = os.path.dirname(problem_file) + "/trace_" + os.path.basename(problem_file)

    parser = PDDL_Parser(domain_file, problem_file)
    grounded_instance = parser.grounded_instance
    plan = Plan.from_file(plan_file, grounded_instance, expand_time_passing=constants.EXPAND_TIME_PASSING)
    trace = plan.simulate(grounded_instance.init_state, grounded_instance)

    if constants.VERBOSE_OUTPUT or constants.VERY_VERBOSE_OUTPUT:
        trace.print(extended=constants.FULL_TRACE)

    trace.to_file(trace_file, extended=constants.FULL_TRACE)


if __name__ == '__main__':
    simulator(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
