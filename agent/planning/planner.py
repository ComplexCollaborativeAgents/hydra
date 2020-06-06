from pip._internal.utils.misc import captured_output

from agent.planning.pddlplus_parser import PddlProblemExporter
from utils.state import InvokeBasicRL
# this will likely just be calling an executable
import settings
from os import path, chdir
import subprocess
import re
from agent.planning.pddl_plus import *
from agent.planning.pddl_meta_model import *

class Planner():
    domain_file = None
    problem = None # current state of the world
    SB_OFFSET = 1


    def __init__(self, meta_model = MetaModel()):
        self.meta_model = meta_model

    def make_plan(self,state,prob_complexity=0):
        '''
        The plan should be a list of actions that are either executable in the environment
        or invoking the RL agent
        '''

        # # CHANGE THE PLANNER MEMORY LIMIT
        # f = open(path.join(settings.PLANNING_DOCKER_PATH, "run_script.sh"), 'r')
        # filedata = f.read()
        # f.close()
        # newdata = re.sub(r'\bm\d*\b', 'm'+str(settings.PLANNER_MEMORY_LIMIT), str(filedata))
        # f = open(path.join(settings.PLANNING_DOCKER_PATH, "run_script.sh"), 'w')
        # f.write(newdata)
        # f.close()

        pddl = self.meta_model.create_pddl_problem(state)
        if prob_complexity==1:
            self.write_problem_file(self.meta_model.create_simplified_problem(pddl))
        elif prob_complexity==2:
            self.write_problem_file(self.meta_model.create_super_simplified_problem(pddl))
        else:
            self.write_problem_file(pddl)
        return self.get_plan_actions()

    def execute(self,plan,policy_learner):
        '''Converts the symbolic action into an environment action'''
        assert False
        if isinstance(plan[0],InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)
        if isinstance(plan[0],SBShoot):
            return None


    def write_problem_file(self, pddl_problem):
        pddl_problem_file = "%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH)
        exporter = PddlProblemExporter()
        exporter.to_file(pddl_problem, pddl_problem_file)

    def get_plan_actions(self,count=0):
        chdir("%s"  % settings.PLANNING_DOCKER_PATH)
        completed_process = subprocess.run(('docker', 'build', '-t', 'upm_from_dockerfile', '.'), capture_output=True)
        out_file = open("docker_build_trace.txt", "wb")
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()

        completed_process = subprocess.run(('docker', 'run', '--rm', 'upm_from_dockerfile', 'sb_domain.pddl',
                                            'sb_prob.pddl', str(settings.PLANNER_MEMORY_LIMIT), str(settings.DELTA_T),
                                            '>', 'docker_plan_trace.txt'), capture_output=True)
        out_file = open("docker_plan_trace.txt", "wb")
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()
        subprocess.run(['docker', 'image', 'prune', '--force'])
        plan_actions =  self.extract_actions_from_plan_trace("%s/docker_plan_trace.txt" % str(settings.PLANNING_DOCKER_PATH))
        if len(plan_actions) > 0:
            return plan_actions
        elif (count <1):
            return self.get_plan_actions(count+1)
        else:
            return []

    ''' Parses the given plan trace file and outputs the plan '''
    def extract_actions_from_plan_trace(self, plane_trace_file: str):
        plan_actions = []
        lines_list = open(plane_trace_file).readlines()
        with open(plane_trace_file) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "Out of memory" in line:
                    plan_actions.append(("out of memory", 20.0))
                    # if the planner ran out of memory:
                    # change the goal to killing a single pig to make the problem easier and try again with one fewer pig
                    return plan_actions

                if " pa-twang " in line:
                    plan_actions.append((line.split(':')[1].split('[')[0].replace('(', '').replace(')', '').strip(),
                                         float(str(lines_list[i + 1].split('angle:')[1].split(',')[0]))))

                if "syntax error" in line:
                    break
        return  plan_actions

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
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()

        completed_process = subprocess.run(('docker', 'run', 'val_from_dockerfile', 'val_domain.pddl', 'val_prob.pddl', 'val_plan.pddl'), capture_output=True)
        out_file = open("docker_validation_trace.txt", "wb")
        out_file.write(completed_process.stdout)
        if len(completed_process.stderr)>0:
            out_file.write(str.encode("\n Stderr: \n"))
            out_file.write(completed_process.stderr)
        out_file.close()
