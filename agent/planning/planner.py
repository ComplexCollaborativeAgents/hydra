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

    def translate_state_to_pddl(self):

        bird_params = ''
        pig_params = ''
        block_params = ''
        goal_conds = ''
        groundOffset = self.cur_sling.bottom_right[1]

        vision = self.sb_client._updateReader('groundTruth')

        prob_instance = '(define (problem angry_birds_prob)\n'
        prob_instance += '(:domain angry_birds_scaled)\n'
        prob_instance += '(:objects '

        for bird, bird_objs in vision.find_birds().items():
            for bo in bird_objs:
                prob_instance += '{}_{} '.format(bird, bo.id)
                bird_params += '    (not (bird_dead {}_{}))\n'.format(bird,bo.id)
                bird_params += '    (not (bird_released {}_{}))\n'.format(bird, bo.id)
                bird_params += '    (= (x_bird {}_{}) {})\n'.format(bird, bo.id,  round((min(bo.points[1]) + max(bo.points[1]))/2)-15 )
                bird_params += '    (= (y_bird {}_{}) {})\n'.format(bird, bo.id, round(abs(((min(bo.points[0]) + max(bo.points[0]))/2) - groundOffset)-15))
                bird_params += '    (= (v_bird {}_{}) 280)\n'.format(bird, bo.id)
                bird_params += '    (= (vy_bird {}_{}) 0)\n'.format(bird, bo.id)
                goal_conds += ' (not (bird_dead {}_{}))'.format(bird, bo.id)

        prob_instance += '- bird '

        for po in vision.find_pigs_mbr():
            prob_instance += '{}_{} '.format('pig', po.id)
            pig_params += '    (not (pig_dead {}_{}))\n'.format('pig', po.id)
            pig_params += '    (= (x_pig {}_{}) {})\n'.format('pig', po.id, min(po.points[1]))
            pig_params += '    (= (y_pig {}_{}) {})\n'.format('pig', po.id, abs(min(po.points[0]) - groundOffset))
            pig_params += '    (= (margin_pig {}_{}) {})\n'.format('pig', po.id, round(abs(max(po.points[1]) - min(po.points[1]))*0.5))
            goal_conds += ' (pig_dead {}_{})'.format('pig', po.id)

        prob_instance += '- pig '

        if vision.find_blocks() != None:
            for block, block_objs in vision.find_blocks().items():
                for blo in block_objs:
                    prob_instance += '{}_{} - block '.format(block, blo.id)
                    bird_params += '    (not (block_destroyed {}_{}))\n'.format(block,blo.id)
                    bird_params += '    (= (x_block {}_{}) {})\n'.format(block, blo.id,  min(blo.points[1]))
                    bird_params += '    (= (y_block {}_{}) {})\n'.format(block, blo.id, abs(min(blo.points[0]) - groundOffset))
                    bird_params += '    (= (block_height {}_{}) {})\n'.format(block, blo.id, abs(max(blo.points[0]) - min(blo.points[0])))
                    bird_params += '    (= (block_width {}_{}) {})\n'.format(block, blo.id, abs(max(blo.points[1]) - min(blo.points[1])))
        else:
            prob_instance += 'dummy_block - block '

        if vision.find_hill_mbr() != None:
            for pla in vision.find_hill_mbr():
                prob_instance += '{}_{} - platform '.format('hill', pla.id)
                bird_params += '    (= (x_platform {}_{}) {})\n'.format('hill', pla.id,  min(pla.points[1]))
                bird_params += '    (= (y_platform {}_{}) {})\n'.format('hill', pla.id, abs(min(pla.points[0]) - groundOffset))
                bird_params += '    (= (platform_height {}_{}) {})\n'.format('hill', pla.id, abs(max(pla.points[0]) - min(pla.points[0])))
                bird_params += '    (= (platform_width {}_{}) {})\n'.format('hill', pla.id, abs(max(pla.points[1]) - min(pla.points[1])))
        else:
            prob_instance += 'dummy_platform - platform '
        # prob_instance += '- platform '

        prob_instance += ')\n' #close objects

        init_params = '(:init '
        init_params += '(= (gravity) 139.0)\n    (= (angle) 0)\n    (= (angle_rate) 10)\n    (bird_in_slingshot)\n    (not (angle_adjusted))\n'

        init_params += bird_params
        init_params += pig_params

        init_params += ')\n' # close init

        prob_instance += init_params

        prob_instance += '(:goal (and {}))\n'.format(goal_conds)

        prob_instance += '(:metric minimize(total-time))\n'
        prob_instance += ')\n' # close define
        # print(prob_instance)

        return prob_instance

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
        with open("%s/docker_plan_trace.txt" % str(path.join(settings.ROOT_PATH, 'agent', 'planning', 'docker_scripts'))) as plan_trace_file:
            for i, line in enumerate(plan_trace_file, 120):
                if " pa-twang " in line:
                    plan_actions.append((line.split(':')[1].split('[')[0].replace('(','').replace(')','').strip(), float(line.split(':')[0])))

        return plan_actions

    # angle_theta = (float(angle_theta_str)*10+60)*1.0
    # print("\nRelease Angle = " + str(angle_theta) + "\n")