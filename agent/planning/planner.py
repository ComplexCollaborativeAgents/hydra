from utils.state import InvokeBasicRL
# this will likely just be calling an executable
import settings
from os import path
import subprocess

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
        return [InvokeBasicRL(state)]

    def execute(self,plan,policy_learner):
        '''Converts the symbolic action into an environment action'''
        if isinstance(plan[0],InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)


    def write_problem_file(self, prob_string):
        pddl_problem_file = open("%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH), "w+")
        pddl_problem_file.write(prob_string)
        pddl_problem_file.close()

    def get_plan_actions(self):
        plan_actions = []
        subprocess.call(
            "cd %s; docker build -t upm_from_dockerfile . > docker_build_trace.txt;docker run upm_from_dockerfile sb_domain.pddl sb_prob.pddl > docker_plan_trace.txt;" % (
                settings.PLANNING_DOCKER_PATH), shell=True)
        angle_theta_str = 0
        lines_list = open("%s/docker_plan_trace.txt" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))).readlines()

        with open("%s/docker_plan_trace.txt" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if " pa-twang " in line:
                    plan_actions.append((line.split(':')[1].split('[')[0].replace('(','').replace(')','').strip(), float(str(lines_list[i+1].split(',')[1].split('angle:')[1]))))
                    # print(str(lines_list[i]))
                    # print(float(str(lines_list[i+1].split(',')[1].split('angle:')[1])))

        print("\nACTIONS: " + str(plan_actions))
        print('Adjusted Angle: ' + str(plan_actions[0][1]*1.05) + "\n")

        return plan_actions

    # angle_theta = (float(angle_theta_str)*10+60)*1.0
    # print("\nRelease Angle = " + str(angle_theta) + "\n")