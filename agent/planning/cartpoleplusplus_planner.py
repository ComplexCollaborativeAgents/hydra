from pip._internal.utils.misc import captured_output

from agent.planning.pddlplus_parser import PddlProblemExporter
from utils.state import InvokeBasicRL
# this will likely just be calling an executable
import settings
from os import path, chdir
import subprocess
import re
from agent.planning.pddl_plus import *
from agent.planning.cartpoleplusplus_pddl_meta_model import *
from agent.planning.nyx import nyx
import datetime
import time
import copy
from agent.hydra_agent import HydraPlanner


class CartPolePlusPlusPlanner(HydraPlanner):
    domain_file = None
    problem = None  # current state of the world
    SB_OFFSET = 1

    def __init__(self, meta_model=CartPolePlusPlusMetaModel()):
        super().__init__(meta_model)
        self.current_problem_prefix = None

    def make_plan(self, state, prob_complexity=0):
        '''
        The plan should be a list of actions that are either executable in the environment
        or invoking the RL agent
        '''

        pddl = self.meta_model.create_pddl_problem(state)
        if prob_complexity == 1:
            self.write_problem_file(self.meta_model.create_simplified_problem(pddl))
        elif prob_complexity == 2:
            self.write_problem_file(self.meta_model.create_super_simplified_problem(pddl))
        else:
            self.write_problem_file(pddl)
        return self.get_plan_actions()

    def execute(self, plan, policy_learner):
        '''Converts the symbolic action into an environment action'''
        assert False
        if isinstance(plan[0], InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)
        if isinstance(plan[0], SBShoot):
            return None

    def write_problem_file(self, pddl_problem):
        pddl_problem_file = "%s/cartpole_prob.pddl" % str(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH)
        exporter = PddlProblemExporter()
        exporter.to_file(pddl_problem, pddl_problem_file)
        if settings.DEBUG:
            self.current_problem_prefix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            cmd = "mkdir -p {}/trace/problems".format(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH)
            subprocess.run(cmd, shell=True)
            exporter.to_file(pddl_problem,
                             "{}/trace/problems/{}_cartpole_problem.pddl".format(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH,
                                                                                 self.current_problem_prefix))

    def get_plan_actions(self, count=0):
        nyx.runner("%s/cartpole_plus_plus_domain.pddl" % str(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH),
                   "%s/cartpole_prob.pddl" % str(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH),
                   ['-vv', '-to:%s' % str(settings.CP_TIMEOUT), '-np:6', '-noplan', '-dblevent', '-search:gbfs', '-custom_heuristic:1',
                    '-th:%s' % str(self.meta_model.constant_numeric_fluents['time_limit']),
                    '-t:%s' % str(settings.CP_DELTA_T)])

        plan_actions = self.extract_actions_from_plan_trace(
            "%s/plan_cartpole_prob.pddl" % str(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH))

        if len(plan_actions) > 0:
            if (plan_actions[0].action_name == "syntax error") and (count < 1):
                return self.get_plan_actions(count + 1)
            else:
                return plan_actions
        else:
            return []

    ''' Parses the given plan trace file and outputs the plan '''

    def extract_actions_from_plan_trace(self, plane_trace_file: str):
        plan_actions = PddlPlusPlan()
        lines_list = open(plane_trace_file).readlines()
        with open(plane_trace_file) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "No Plan Found!" in line:
                    plan_actions.append(TimedAction("out of memory", 1.0))
                    # if the planner ran out of memory:
                    # change the goal to killing a single pig to make the problem easier and try again with one fewer pig
                    return plan_actions
                if "time-passing" in line:
                    action_angle_time = (line.split(':')[1].split('[')[0].replace('(', '').replace(')', '').strip(),
                                         float(str(lines_list[i + 6].split('\"[\'f_x\']\",')[1].split(']')[0])),
                                         float(str(lines_list[i + 6].split('\"[\'f_y\']\",')[1].split(']')[0])),
                                         float(line.split(':')[0].strip()))
                    # print (str(action_angle_time) + "\n")

                    # TODO: CHECK WHETHER TEXTUAL DESCRIPTIONS CAN BE DROPPED IN FAVOUR OF NUMERIC ACTION INDEXES
                    action_name = "do_nothing dummy_obj"
                    if action_angle_time[1] != 0.0 and action_angle_time[2] == 0.0:
                        if action_angle_time[1] > 0.0:
                            action_name = "move_cart_left dummy_obj"
                        if action_angle_time[1] < 0.0:
                            action_name = "move_cart_right dummy_obj"
                    elif action_angle_time[1] == 0.0 and action_angle_time[2] != 0.0:
                        if action_angle_time[2] < 0.0:
                            action_name = "move_cart_backward dummy_obj"
                        if action_angle_time[2] > 0.0:
                            action_name = "move_cart_forward dummy_obj"
                    elif action_angle_time[1] == 0.0 and action_angle_time[2] == 0.0:
                        action_name = "do_nothing dummy_obj"
                    else:
                        action_name = "QUEEEEE?"

                    plan_actions.append(TimedAction(action_name, action_angle_time[3]))
                if "syntax error" in line:
                    plan_actions.append(TimedAction("syntax error", 0.0))
                    break

                if "0 goals found" in line:
                    break

        return plan_actions

        ''' Parses the given plan trace file and outputs the plan '''

    def extract_state_values_from_trace(self, plane_trace_file: str):
        plan_values = []
        lines_list2 = open(plane_trace_file).readlines()
        with open(plane_trace_file) as plan_trace_file:
            for ix, linex in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "time-passing" in linex:
                    copy_line = copy.copy(lines_list2[ix + 5])
                    copy_line2 = copy.copy(lines_list2[ix + 5])
                    copy_line3 = copy.copy(lines_list2[ix + 5])
                    copy_line4 = copy.copy(lines_list2[ix + 5])
                    copy_line5 = copy.copy(lines_list2[ix + 5])
                    copy_line6 = copy.copy(lines_list2[ix + 5])
                    copy_line7 = copy.copy(lines_list2[ix + 5])
                    copy_line8 = copy.copy(lines_list2[ix + 5])

                    state_values = (float(str(copy_line.split('\"[\'pos_x\']\",')[1].split(']')[0])),
                                    float(str(copy_line2.split('\"[\'pos_y\']\",')[1].split(']')[0])),
                                    float(str(copy_line3.split('\"[\'pos_x_dot\']\",')[1].split(']')[0])),
                                    float(str(copy_line4.split('\"[\'pos_y_dot\']\",')[1].split(']')[0])),
                                    float(str(copy_line.split('\"[\'theta_x\']\",')[1].split(']')[0])),
                                    float(str(copy_line2.split('\"[\'theta_y\']\",')[1].split(']')[0])),
                                    float(str(copy_line3.split('\"[\'theta_x_dot\']\",')[1].split(']')[0])),
                                    float(str(copy_line4.split('\"[\'theta_y_dot\']\",')[1].split(']')[0])),
                                    (round(float(linex.split(':')[0].strip()) + 0.02, 4)))
                    # print (str(state_values) + "\n")
                    # print("COPY LINEEEEEE: "+str(copy_line))
                    plan_values.append(state_values)

        return plan_values

    def run_val(self):

        # chdir("%s" % settings.PLANNING_DOCKER_PATH)

        unobscured_plan_list = []

        # COPY DOMAIN FILE TO VAL DIRECTORY FOR VALIDATION.
        cmd = 'cp {}/sb_domain.pddl {}/val_domain.pddl'.format(str(settings.SB_PLANNING_DOCKER_PATH),
                                                               str(settings.VAL_DOCKER_PATH))
        subprocess.run(cmd, shell=True)

        # COPY PROBLEM FILE TO VAL DIRECTORY FOR VALIDATION.
        cmd = 'cp {}/sb_prob.pddl {}/val_prob.pddl'.format(str(settings.SB_PLANNING_DOCKER_PATH),
                                                           str(settings.VAL_DOCKER_PATH))
        subprocess.run(cmd, shell=True)

        with open("%s/docker_plan_trace.txt" % str(settings.SB_PLANNING_DOCKER_PATH)) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if " pa-twang " in line:
                    # print(str(lines_list[i]))
                    # print(float(str(lines_list[i+1].split('angle:')[1].split(',')[0])))
                    unobscured_plan_list.append(line)

        # COPY ACTIONS DIRECTLY INTO A TEXT FILE FOR VALIDATION WITH VAL.
        val_plan = open("%s/val_plan.pddl" % str(settings.VAL_DOCKER_PATH), "w")
        for acn in unobscured_plan_list:
            val_plan.write(acn)
        val_plan.close()

        chdir("%s" % settings.VAL_DOCKER_PATH)

        completed_process = subprocess.run(('docker', 'build', '-t', 'val_from_dockerfile', '.'), capture_output=True)
        out_file = open("docker_build_trace.txt", "wb")
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr) > 0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()

        completed_process = subprocess.run(
            ('docker', 'run', 'val_from_dockerfile', 'val_domain.pddl', 'val_prob.pddl', 'val_plan.pddl'),
            capture_output=True)
        out_file = open("docker_validation_trace.txt", "wb")
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr) > 0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()
