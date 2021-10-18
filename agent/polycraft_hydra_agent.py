import random
import datetime

from agent.consistency.observation import HydraObservation
from agent.planning.polycraft_meta_model import PolycraftMetaModel
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_interface.client.polycraft_interface import TiltDir
import worlds.polycraft_interface.client.polycraft_interface as poly
from agent.planning.pddlplus_parser import *
from agent.hydra_agent import HydraAgent, HydraPlanner, MetaModelRepair
from worlds.polycraft_world import *
from agent.planning.nyx import nyx

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")

import json


class PolycraftObservation(HydraObservation):
    ''' An object that represents an observation of the SB game '''

    def __init__(self):
        self.states = [] # A sequence of polycraft states
        self.actions = []  # A sequence of polycraft actions
        self.rewards = [] # The reward obtained from performing each action
        self.actions_success = [] # Whether the performed action was successful or not

    def get_initial_state(self):
        return self.states[0]

    def get_pddl_states_in_trace(self, meta_model: PolycraftMetaModel = PolycraftMetaModel()) -> list: # TODO: Refactor and move this to the meta model?
        ''' Returns a sequence of PDDL states that are the observed intermediate states '''
        observed_state_seq = []
        for state in self.states:
            pddl = meta_model.create_pddl_state(state)
            observed_state_seq.append(pddl)
        return observed_state_seq

    def get_pddl_plan(self, meta_model: PolycraftMetaModel = PolycraftMetaModel):
        ''' Returns a PDDL+ plan object with a single action that is the action that was performed '''
        raise NotImplementedError()

    def print(self):
        for i in len(self.states):
            print(f'State[{i}] {str(self.states[i])}')
            print(f'Action[{i}] {str(self.actions[i])}')
            print(f'Success[{i}] {str(self.actions_success[i])}')

class PolycraftPlanner(HydraPlanner):
    ''' Planner for the polycraft domain'''

    def __init__(self, meta_model = PolycraftMetaModel(), planning_path = settings.POLYCRAFT_PLANNING_DOCKER_PATH):
        super().__init__(meta_model)
        self.current_problem_prefix = None
        self.planning_path = planning_path
        self.pddl_problem_file = "%s/polycraft_prob.pddl" % str(self.planning_path)
        self.pddl_domain_file = "%s/polycraft_domain.pddl" % str(self.planning_path)
        self.pddl_plan_file = "%s/plan_polycraft_prob.pddl" % str(self.planning_path)

        self.delta_t = settings.POLYCRAFT_DELTA_T
        self.timeout = settings.POLYCRAFT_TIMEOUT

    def make_plan(self,state):
        if settings.NO_PLANNING:
            self.current_problem_prefix = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # need a prefix for observations
            return []
        pddl_problem = self.meta_model.create_pddl_problem(state)
        pddl_domain = self.meta_model.create_pddl_domain(state)
        self.write_pddl_file(pddl_problem, pddl_domain)
        return self.get_plan_actions()


    def write_pddl_file(self, pddl_problem : PddlPlusProblem, pddl_domain: PddlPlusDomain):
        problem_exporter = PddlProblemExporter()
        problem_exporter.to_file(pddl_problem, self.pddl_problem_file)
        domain_exporter = PddlDomainExporter()
        domain_exporter.to_file(pddl_domain, self.pddl_domain_file)
        if settings.DEBUG:
            self.current_problem_prefix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            cmd = "mkdir -p {}/trace/problems".format(self.planning_path)
            subprocess.run(cmd, shell=True)
            problem_exporter.to_file(pddl_problem, "{}/trace/problems/{}_problem.pddl".format(self.planning_path,
                                                                                      self.current_problem_prefix))
            domain_exporter.to_file(pddl_domain, "{}/trace/problems/{}_domain.pddl".format(self.planning_path,
                                                                                      self.current_problem_prefix))

    def get_plan_actions(self,count=0):

        plan_actions = []

        try:
            nyx.runner(self.pddl_domain_file,
                       self.pddl_problem_file,
                       ['-vv', '-to:%s' % str(self.timeout), '-noplan', '-search:bfs', '-custom_heuristic:2', '-th:10',
                        # '-th:%s' % str(self.meta_model.constant_numeric_fluents['time_limit']),
                        '-t:%s' % str(self.delta_t)])

            plan_actions = self.extract_actions_from_plan_trace(self.pddl_plan_file)

        except Exception as e_inst:
            print(e_inst)

        # print(plan_actions)

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
                if "pa-twang" in line:
                    action_angle_time = (line.split('\t')[1].strip(),
                                         # float(str(lines_list[i + 1].split('angle:')[1].split(',')[0])),
                                         float(line.split(':')[0]))
                    plan_actions.append(TimedAction(action_angle_time[0], action_angle_time[1]))

                ## TAP UPDATE
                # if "bird_action" in line:
                #     action_angle_time = (line.split(':')[1].split('[')[0].replace('(', '').replace(')', '').strip(),
                #                          float(str(lines_list[i + 1].split('angle:')[1].split(',')[0])),
                #                          float(line.split(':')[0]))
                #     plan_actions.append(TimedAction(action_angle_time[0], action_angle_time[2]))

                if "syntax error" in line:
                    plan_actions.append(TimedAction("syntax error", 0.0))
                    break
        return  plan_actions

    # def run_val(self):
    #
    #     # chdir("%s" % settings.PLANNING_DOCKER_PATH)
    #
    #     unobscured_plan_list = []
    #
    #     # COPY DOMAIN FILE TO VAL DIRECTORY FOR VALIDATION.
    #     cmd = 'cp {}/sb_domain.pddl {}/val_domain.pddl'.format(str(settings.SB_PLANNING_DOCKER_PATH), str(settings.VAL_DOCKER_PATH))
    #     subprocess.run(cmd, shell=True)
    #
    #     # COPY PROBLEM FILE TO VAL DIRECTORY FOR VALIDATION.
    #     cmd = 'cp {}/sb_prob.pddl {}/val_prob.pddl'.format(str(settings.SB_PLANNING_DOCKER_PATH), str(settings.VAL_DOCKER_PATH))
    #     subprocess.run(cmd, shell=True)
    #
    #     with open("%s/docker_plan_trace.txt" % str(settings.SB_PLANNING_DOCKER_PATH)) as plan_trace_file:
    #         for i, line in enumerate(plan_trace_file):
    #             # print(str(i) + " =====> " + str(line))
    #             if " pa-twang " in line:
    #                 # print(str(lines_list[i]))
    #                 # print(float(str(lines_list[i+1].split('angle:')[1].split(',')[0])))
    #                 unobscured_plan_list.append(line)
    #
    #     # COPY ACTIONS DIRECTLY INTO A TEXT FILE FOR VALIDATION WITH VAL.
    #     val_plan = open("%s/val_plan.pddl" % str(settings.VAL_DOCKER_PATH), "w")
    #     for acn in unobscured_plan_list:
    #         val_plan.write(acn)
    #     val_plan.close()
    #
    #
    #     chdir("%s" % settings.VAL_DOCKER_PATH)
    #
    #     completed_process = subprocess.run(('docker', 'build', '-t', 'val_from_dockerfile', '.'), capture_output=True)
    #     out_file = open("docker_build_trace.txt", "wb")
    #     out_file.write(completed_process.stdout)
    #     if len(completed_process.stderr)>0:
    #         out_file.write(str.encode("\n Stderr: \n"))
    #         out_file.write(completed_process.stderr)
    #     out_file.close()
    #
    #     completed_process = subprocess.run(('docker', 'run', 'val_from_dockerfile', 'val_domain.pddl', 'val_prob.pddl', 'val_plan.pddl'), capture_output=True)
    #     out_file = open("docker_validation_trace.txt", "wb")
    #     out_file.write(completed_process.stdout)
    #     if len(completed_process.stderr)>0:
    #         out_file.write(str.encode("\n Stderr: \n"))
    #         out_file.write(completed_process.stderr)
    #     out_file.close()







class PolycraftHydraAgent(HydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self, planner: HydraPlanner = PolycraftPlanner(),
                 meta_model_repair: MetaModelRepair = None):
        super().__init__(planner, meta_model_repair)

    def explore_level(self, env: Polycraft):
        ''' Perform exploratory actions to get familiar with the current level. This is needed before calling the planner '''
        env.populate_current_recipes()

        current_state = env.get_current_state()

        # Try to interact with all other agents
        for entity_id, entity_attr in current_state.entities.items():
            if entity_attr['type']=='EntityTrader':
                # Move to trader
                env.move_to_entity(entity_id)

                # Interact with it
                env.interact(entity_id)

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        logger.info("World state summary is: {}".format(str(world_state)))

        plan = self.planner.make_plan(world_state)

        return plan[0]

    def should_repair(self, observation):
        ''' Choose if the agent should repair its meta model based on the given observation '''
        raise NotImplementedError()

    def repair_meta_model(self, observation):
        ''' Call the repair object to repair the current meta model '''
        raise NotImplementedError()


class PolycraftRandomAgent(PolycraftHydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        logger.info("World state summary is: {}".format(str(world_state)))

        # Choose random action types
        chosen_action = self._choose_random(world_state)

        return chosen_action

    def _choose_random(self, world_state: PolycraftState):
        ''' Choose a random action from the list of possible actions '''
        world_state.get_available_actions()
        action_class = random.choice([PolyTP, PolyEntityTP, PolyTurn, PolyTilt, PolyTilt, PolyBreak, PolyInteract,
                                      PolySelectItem, PolyUseItem, PolyPlaceItem, PolyCollect, PolyDeleteItem, PolyTradeItems,
                                      PolyCraftItem])

        if action_class == PolyTP:
            coordinate = random.choice(list(world_state.game_map.keys())).split(",")
            return PolyTP(coordinate)
        elif action_class == PolyEntityTP:
            npcs = [npc for npc in world_state.entities.keys()]
            if len(npcs) > 0:
                npc = random.choice(npcs)
                return PolyEntityTP(npc)
            else:
                return PolyNoAction()
        elif action_class == PolyTurn:
            dir = random.choice(range(0, 360, 15))
            return PolyTurn(dir)
        elif action_class == PolyTilt:
            angle = random.choice(list(TiltDir))    # Choose from DOWN, FORWARD, and UP
            return PolyTilt(angle)
        elif action_class == PolyBreak:
            return PolyBreak()
        elif action_class == PolySelectItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolySelectItem(item_name)
            else:
                return PolyNoAction()   # No items to select
        elif action_class == PolyUseItem:
            return PolyUseItem()
        elif action_class == PolyPlaceItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolyPlaceItem(item_name)
            else:
                return PolyNoAction()   # No items to place
        elif action_class == PolyDeleteItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolyDeleteItem(item_name)
            else:
                return PolyNoAction()   # No items to place
        elif action_class == PolyCollect:
            return PolyCollect()
        elif action_class == PolyInteract:
            npcs = [npc for npc in world_state.entities.keys()]
            if len(npcs) > 0:
                npc = random.choice(npcs)
                return PolyInteract(npc)
            else:
                return PolyNoAction()   # Do nothing.
        elif action_class == PolyTradeItems:
            # Choose trade from list of possible trades
            if len(world_state.trades) > 0:
                random_trade = random.choice(world_state.trades)
            else:
                return PolyNoAction()
            return PolyTradeItems(random_trade['entity_id'], random_trade['input'])
        elif action_class == PolyCraftItem:
            if len(world_state.trades) > 0:
                random_recipe = random.choice(world_state.recipes)
            else:
                return PolyNoAction()
            return PolyCraftItem(random_recipe)
        else:
            raise ValueError("Bad action class {}".format(action_class))



class PolycraftDoingAgent(PolycraftHydraAgent):
    ''' An agent that mines every available block '''
    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        logger.info("World state summary is: {}".format(str(world_state)))
        actions = world_state.get_available_actions()
        return random.choice(actions)




class PolycraftDoNothingAgent(PolycraftHydraAgent):
    ''' An agent that does nothing '''
    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        return PolyNoAction()


class PolycraftTPAllAgent(PolycraftHydraAgent):
    ''' An agent that performs a prescribed list of actions, after which it terminates '''
    def __init__(self):
        super().__init__()

        self.cells_visited = set()

    def choose_action(self, world_state: PolycraftState):
        cells_to_visit = []
        for coords, block in world_state.game_map.items():
            if block['isAccessible'] and coords not in self.cells_visited: # TODO: Explore why everything seems inaccessible
                cells_to_visit.append(coords)

        if len(cells_to_visit)==0:
            return PolyGiveUp()
        else:
            coords = cells_to_visit.pop(0)
            self.cells_visited.add(coords)
            return PolyTP(coords, dist=1)


class PolycraftTPAgent(PolycraftHydraAgent):
    ''' An agent that moves around between blocks and other entities '''
    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        logger.info("World state summary is: {}".format(str(world_state)))

        actions = []
        for coords, block in world_state.game_map.items():
            if block['isAccessible']:
                actions.append(PolyTP(coords))

        return random.choice(actions)


class PolycraftManualAgent(PolycraftHydraAgent):
    ''' An agent that queries the user for actions USED FOR DEBUGGING '''
    def __init__(self):
        super().__init__()
        self.command_seq = 0
        # self.commands = [PolyTP("43,17,42"), PolyTilt(TiltDir.FORWARD), PolyTurn(90), PolyTurn(90), PolyTurn(90), PolyTurn(90), PolyTurn(45), PolyTurn(90), PolyTurn(90), PolyTurn(90), PolyTurn(90)]
        self.commands = [PolyMoveToAndBreak("43,17,42"), PolyCraftItem(["minecraft:log", "0", "0", "0"])]

    def choose_action(self, world_state: PolycraftState):
        if self.command_seq<len(self.commands):
            cmd = self.commands[self.command_seq]
            self.command_seq=self.command_seq+1
        else:
            return PolyGiveUp()

        return cmd