from utils.state import InvokeBasicRL
# this will likely just be calling an executable
import settings
from os import path, chdir
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
        self.write_problem_file(state.translate_state_to_pddl())
        return self.get_plan_actions()

    def execute(self,plan,policy_learner):
        '''Converts the symbolic action into an environment action'''
        assert False
        if isinstance(plan[0],InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)
        if isinstance(plan[0],SBShoot):
            return None


    def write_problem_file(self, prob_string):
        pddl_problem_file = open("%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH), "w+")
        pddl_problem_file.write(prob_string)
        pddl_problem_file.close()

    def get_plan_actions(self,count=0):
        plan_actions = []

        chdir("%s"  % settings.PLANNING_DOCKER_PATH)
        completed_process = subprocess.run(('docker', 'build', '-t', 'upm_from_dockerfile', '.'), capture_output=True)
        out_file = open("docker_build_trace.txt", "wb")
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write("\n Stderr: \n")
            out_file.write(completed_process.stderr);
        out_file.close()

        completed_process = subprocess.run(('docker', 'run', 'upm_from_dockerfile', 'sb_domain.pddl', 'sb_prob.pddl'), capture_output=True)
        out_file = open("docker_plan_trace.txt", "wb")
        out_file.write(completed_process.stdout);
        if len(completed_process.stderr)>0:
            out_file.write("\n Stderr: \n")
            out_file.write(completed_process.stderr);
        out_file.close()

        lines_list = open("%s/docker_plan_trace.txt" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))).readlines()

        with open("%s/docker_plan_trace.txt" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "Out of memory" in line:
                    plan_actions.append(("out of memory", -999))
                    return plan_actions
                if " pa-twang " in line:
                    # print(str(lines_list[i]))
                    # print(float(str(lines_list[i+1].split('angle:')[1].split(',')[0])))
                    plan_actions.append((line.split(':')[1].split('[')[0].replace('(','').replace(')','').strip(), float(str(lines_list[i+1].split('angle:')[1].split(',')[0]))))


        print("\nACTIONS: " + str(plan_actions))
        if len(plan_actions) > 0:
            return plan_actions
        elif (count <10):
            return self.get_plan_actions(count+1)
        else:
            return []