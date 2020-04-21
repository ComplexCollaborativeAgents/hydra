from pip._internal.utils.misc import captured_output

from agent.planning.pddlplus_parser import PddlProblemExporter, PddlDomainExporter
from agent.planning.pddl_plus import PddlPlusProblem, PddlPlusDomain, PddlPlusPlan, TimedAction
from utils.state import InvokeBasicRL
# this will likely just be calling an executable
import settings
from os import path, chdir
import subprocess
import re

class Planner():
    domain_file = None
    problem = None # current state of the world
    SB_OFFSET = 1


    def __init__(self):
        pass

    def make_plan(self,state):
        '''
        The plan should be a list of actions that are either executable in the environment
        or invoking the RL agent
        '''
        pddl = state.translate_initial_state_to_pddl_problem()
        self.write_problem_file(pddl)
        return self.get_plan_actions()

    def execute(self,plan,policy_learner):
        '''Converts the symbolic action into an environment action'''
        assert False
        if isinstance(plan[0],InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)
        if isinstance(plan[0],SBShoot):
            return None

    ''' Runs the planner on the given problem and domain, return the plan '''
    def plan(self, pddl_problem : PddlPlusProblem, pddl_domain : PddlPlusDomain):
        self.write_problem_file(pddl_problem)
        self.write_domain_file(pddl_domain)
        return self.get_plan_actions()


    def write_problem_file(self, pddl_problem : PddlPlusProblem):
        pddl_problem_file = "%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH)
        exporter = PddlProblemExporter()
        exporter.to_file(pddl_problem, pddl_problem_file)

    def write_domain_file(self, pddl_domain : PddlPlusDomain):
        pddl_domain_file = "%s/sb_domain.pddl" % str(settings.PLANNING_DOCKER_PATH)
        exporter = PddlDomainExporter()
        exporter.to_file(pddl_domain, pddl_domain_file)

    def get_plan_actions(self,count=0):
        plan_actions = []

        chdir("%s"  % settings.PLANNING_DOCKER_PATH)
        completed_process = subprocess.run(('docker', 'build', '-t', 'upm_from_dockerfile', '.'), capture_output=True)
        out_file = open("docker_build_trace.txt", "wb")
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr);
        out_file.close()

        completed_process = subprocess.run(('docker', 'run', 'upm_from_dockerfile', 'sb_domain.pddl', 'sb_prob.pddl', '>', 'docker_plan_trace.txt'), capture_output=True)
        out_file = open("docker_plan_trace.txt", "wb")
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr);
        out_file.close()

        plan_trace_file = "%s/docker_plan_trace.txt" % str(settings.PLANNING_DOCKER_PATH)
        return self.extract_actions_from_plan_trace(plan_trace_file, count)


    ''' Extracts a PddlPlusPlan object from a plan trace. TODO: Currently assumes domain is grounded'''
    def extract_plan_from_plan_trace(self, plan_trace_file_name :str, grounded_domain: PddlPlusDomain) -> PddlPlusPlan:
        ACTION_REGEX = re.compile(r"^(\S*):.*\((.*)\).*$")
        plan = PddlPlusPlan()
        plan_trace_file = open(plan_trace_file_name,"r")
        for i, line in enumerate(plan_trace_file):
            if "Out of memory" in line:
                return None
            elif ACTION_REGEX.match(line):
                groups = ACTION_REGEX.search(line)
                time = float(groups[1].strip())
                action_str = groups[2].strip() # Removing white spaces in the brackets
                assert action_str.startswith("pa-twang") # Assert AB action is a twang TODO: Remove this when things get messier
                action_obj = None
                for action in grounded_domain.actions:
                    if action.name==action_str:
                        action_obj = action
                        break

                if action_obj is None:
                    raise ValueError("Action name %s is not in the grounded domain" % action_str)

                timed_action = TimedAction(action_obj, time)
                plan.append(timed_action)
        return plan

    ''' Parses the given plan trace file and outputs the plan '''
    def extract_actions_from_plan_trace(self, plane_trace_file : str, count=0):
        plan_actions = []
        lines_list = open(plane_trace_file).readlines()
        with open(plane_trace_file) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "Out of memory" in line:
                    plan_actions.append(("out of memory", -999))
                    return plan_actions
                if " pa-twang " in line:
                    plan_actions.append((line.split(':')[1].split('[')[0].replace('(', '').replace(')', '').strip(),
                                         float(str(lines_list[i + 1].split('angle:')[1].split(',')[0]))))
        self.run_val()
        print("\nACTIONS: " + str(plan_actions))
        if len(plan_actions) > 0:
            return plan_actions
        elif (count < 10):
            return self.get_plan_actions(count + 1)
        else:
            return []

    ''' Parses the expected trace of the given plan trace '''
    def extract_trace_from_plan_trace(self, plane_trace_file : str, count=0):
        trace = []
        lines_list = open(plane_trace_file).readlines()
        with open(plane_trace_file) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                if line.startswith("; TIME"):
                    parts = line.split(",")
                    key_value = parts[0].split(":")
                    assert key_value[0]=="; TIME"
                    trace_line = dict()
                    trace_line["t"]=float(key_value[1])

                    parts = parts[1:]
                    for part in parts:
                        if len(part.strip())>0:
                            key_value = part.split(":")

                            trace_line[key_value[0].strip()]=key_value[1].strip()
                    trace.append(trace_line)
        return trace


    def run_val(self):

        # chdir("%s" % settings.PLANNING_DOCKER_PATH)

        unobscured_plan_list = []

        # COPY DOMAIN FILE TO VAL DIRECTORY FOR VALIDATION.
        cmd = 'cp {}/sb_domain.pddl {}/val_domain.pddl'.format(str(settings.PLANNING_DOCKER_PATH), str(settings.VAL_DOCKER_PATH))
        subprocess.run(cmd, shell=True)

        # COPY PROBLEM FILE TO VAL DIRECTORY FOR VALIDATION.
        cmd = 'cp {}/sb_prob.pddl {}/val_prob.pddl'.format(str(settings.PLANNING_DOCKER_PATH), str(settings.VAL_DOCKER_PATH))
        subprocess.run(cmd, shell=True)

        with open("%s/docker_plan_trace.txt" % str(settings.PLANNING_DOCKER_PATH)) as plan_trace_file:
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
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr);
        out_file.close()

        completed_process = subprocess.run(('docker', 'run', 'val_from_dockerfile', 'val_domain.pddl', 'val_prob.pddl', 'val_plan.pddl'), capture_output=True)
        out_file = open("docker_validation_trace.txt", "wb")
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr);
        out_file.close()