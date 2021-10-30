import datetime

from agent.consistency.observation import HydraObservation
from agent.planning.polycraft_planning.tasks import *
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_interface.client.polycraft_interface import TiltDir
from agent.planning.pddlplus_parser import *
from agent.hydra_agent import HydraAgent, HydraPlanner, MetaModelRepair
from worlds.polycraft_world import *
from agent.planning.nyx import nyx
import agent.planning.nyx.heuristic_functions as nyx_heuristics

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")

import re

RE_EXTRACT_ACTION_PATTERN = re.compile(r" *(\d*\.\d*):\t*(.*)\t\[0\.0\]")  # Pattern used to extract action time and name from planner output

class PolycraftObservation(HydraObservation):
    ''' An object that represents an observation of the SB game '''

    def __init__(self):
        self.states = [] # A sequence of polycraft states
        self.actions = []  # A sequence of polycraft actions
        self.rewards = [] # The reward obtained from performing each action

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

class PolycraftPlanner(HydraPlanner):
    ''' Planner for the polycraft domain'''

    def __init__(self, meta_model = PolycraftMetaModel(active_task=PolycraftTask.CRAFT_POGO.create_instance()), planning_path = settings.POLYCRAFT_PLANNING_DOCKER_PATH):
        super().__init__(meta_model)
        self.current_problem_prefix = None
        self.planning_path = planning_path
        self.pddl_problem_file = "%s/polycraft_prob.pddl" % str(self.planning_path)
        self.pddl_domain_file = "%s/polycraft_domain.pddl" % str(self.planning_path)
        self.pddl_plan_file = "%s/plan_polycraft_prob.pddl" % str(self.planning_path)

        self.delta_t = settings.POLYCRAFT_DELTA_T
        self.timeout = settings.POLYCRAFT_TIMEOUT

    def make_plan(self, state:PolycraftState):
        if settings.NO_PLANNING:
            self.current_problem_prefix = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # need a prefix for observations
            return []
        self.initial_state = state
        self.pddl_problem = self.meta_model.create_pddl_problem(state)
        self.pddl_domain = self.meta_model.create_pddl_domain(state)
        self.write_pddl_file(self.pddl_problem, self.pddl_domain)
        nyx_heuristics.active_heuristic = self.meta_model.get_nyx_heuristic(state)

        try:
            nyx.runner(self.pddl_domain_file,
                       self.pddl_problem_file,
                       ['-vv', '-to:%s' % str(self.timeout), '-noplan', '-search:gbfs', '-custom_heuristic:3', '-th:10',
                        # '-th:%s' % str(self.meta_model.constant_numeric_fluents['time_limit']),
                        '-t:%s' % str(self.delta_t)])
            plan_actions = self.extract_actions_from_plan_trace(self.pddl_plan_file)
            if len(plan_actions) > 0:
                return plan_actions
            else:
                return []
        except Exception as e_inst:
            logger.error(f"Exception while running planner. {e_inst}", stack_info=True)
            logger.exception(e_inst)
            print(e_inst)
        return []


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

    def _get_poly_action_from_pddl_action(self, pddl_action_line:str, action_generators:list)->PolycraftAction:
        ''' Generates a PolycraftAction using the given list of action generators based on the given
        pddl action line. '''

        # Extract action name and parameters
        matches = list(RE_EXTRACT_ACTION_PATTERN.finditer(pddl_action_line))
        assert (len(matches) == 1)
        assert (len(matches[0].groups()) == 2)
        action_time = matches[0].groups()[0].strip()
        action_name_and_params = matches[0].groups()[1].strip()
        action_parts = action_name_and_params.split(" ")
        action_name = action_parts[0].strip()

        # Find appropriate action generator
        logger.info(f"Parsing line {pddl_action_line} for action {action_name}")
        selected_action_gen = None
        for action_gen in action_generators:
            if action_gen.pddl_name.replace(":","_")==action_name:
                selected_action_gen = action_gen
                break
        assert(selected_action_gen is not None)
        pddl_action = selected_action_gen.to_pddl(self.meta_model)

        # Handle parameters if needed
        binding = dict()
        if len(action_parts)>1:
            params_parts = action_parts[1:]
            pddl_parameters_list = pddl_action.parameters[0]
            assert(len(params_parts)*3 == len(pddl_parameters_list))
            for i, param in enumerate(params_parts):
                param_value = param.strip()
                param_name = pddl_parameters_list[i*3].strip()
                param_type = pddl_parameters_list[i * 3+2].strip()
                binding[param_name]=self._translate_pddl_object_to_poly(param_value, param_type)

        return selected_action_gen.to_polycraft(binding)

    def _translate_pddl_object_to_poly(self, pddl_obj_name, pddl_obj_type):
        ''' Translate the name of a pddl object to its corresponding name in polycraft, based on its type
            For cell type objects, this means convert from cell_x_y_z to x,y,z
        '''
        assert(pddl_obj_type in [type.name for type in PddlType]) # Currently we only have objects of type cell
        return ",".join(pddl_obj_name.split("_")[1:])

    def extract_actions_from_plan_trace(self, plane_trace_file: str):
        ''' Parses the given plan trace file and outputs the plan '''
        pddl_action_names = [action.name for action in self.pddl_domain.actions]
        action_generators = self.meta_model.create_action_generators(self.initial_state)
        plan_actions = []
        with open(plane_trace_file) as plan_trace_file:
            for i, line in enumerate(plan_trace_file):
                # print(str(i) + " =====> " + str(line))
                if "No Plan Found!" in line:
                    logger.info("No plan found")
                    return None

                action_in_plan = None
                for action_name in pddl_action_names:
                    if action_name in line:
                        action_in_plan = action_name
                        break

                if action_in_plan:
                    plan_actions.append(self._get_poly_action_from_pddl_action(line, action_generators))
        return  plan_actions


class PolycraftHydraAgent(HydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self, planner: HydraPlanner = PolycraftPlanner(),
                 meta_model_repair: MetaModelRepair = None):
        super().__init__(planner, meta_model_repair)
        self.active_plan = None
        self.active_action = None
        self.current_observation = None
        self.current_state = None # Maintains the agent's knowledge about the current state
        self.exploration_tasks = list()
        self.exploration_rate = 1 # Number of failed actions to endure before trying one exploration task
        self.failed_actions_in_level = 0 # Count how many actions have failed in a given level

    def start_level(self, env: Polycraft):
        ''' Initialize datastructures for a new level and perform exploratory actions to get familiar with the current level.
        These actions are needed before calling the planner  '''

        # Explore the level
        env.init_state_information()
        env.populate_current_recipes()
        current_state = env.get_current_state()
        # Try to interact with all other agents
        for entity_id, entity_attr in current_state.entities.items():
            if entity_attr['type']=='EntityTrader':
                # If trader not accessible, mark getting to it as a possilbe exploration task
                trader_cell = coordinates_to_cell(entity_attr['pos'])
                if current_state.game_map[trader_cell]['isAccessible']==False:
                    reach_cell_task = PolycraftTask.MAKE_CELL_ACCESSIBLE.create_instance()
                    reach_cell_task.cell = trader_cell
                    self.exploration_tasks.append(reach_cell_task)
                    continue

                # Move to trader
                tp_action = PolyEntityTP(entity_id,dist=1)
                current_state, step_cost = env.act(tp_action)

                if tp_action.success==False:
                    trader_cell = coordinates_to_cell(entity_attr['pos'])
                    cell_accessible = current_state.game_map[trader_cell]['isAccessible']
                    logger.info(f"Entity {entity_id} is at cell {trader_cell} whose accessibility is {cell_accessible}, but TP_TO failed.")

                    attempts = 0
                    max_attempts = 4
                    while cell_accessible and attempts < max_attempts:
                        tp_action = PolyEntityTP(entity_id, dist=1)
                        current_state, step_cost = env.act(tp_action)
                        if tp_action.success:
                            break # Managed to reach the trader
                        attempts = attempts+1
                        trader_cell = coordinates_to_cell(entity_attr['pos'])
                        cell_accessible = current_state.game_map[trader_cell]['isAccessible']

                if tp_action.success==False:
                    logger.info(f"Entity {entity_id} not reached. Added it as an exploration task")
                    trader_cell = coordinates_to_cell(entity_attr['pos'])
                    if current_state.game_map[trader_cell]['isAccessible'] == True:
                        logger.info("Bug in polycraft: accessible trader is not acessible")
                    task = PolycraftTask.MAKE_CELL_ACCESSIBLE.create_instance()
                    task.cell = trader_cell
                    self.exploration_tasks.append(task)
                    continue

                # Interact with it
                interact_action = PolyInteract(entity_id)
                current_state, step_cost = env.act(interact_action)
                assert(interact_action.success)
                env.current_trades[entity_id] = interact_action.response['trades']['trades']


        # Add doors to other rooms as exploration tasks
        for door_cell in current_state.get_cells_of_type(BlockType.WOODER_DOOR.value):
            task = PolycraftTask.OPEN_DOOR.create_instance()
            task.door_cell = door_cell
            self.exploration_tasks.append(task)

        # Initialize the current observation object
        current_state = env.get_current_state()
        self.current_observation = PolycraftObservation() # Start a new observation object for this level
        self.current_observation.states.append(current_state)
        self.observations_list.append(self.current_observation)
        self.env = env
        self.active_action = None
        self.active_plan = None
        self.failed_actions_in_level = 0

    def _choose_exploration_task(self, world_state: PolycraftState):
        ''' Choose an exploration task to perform '''
        assert(len(self.exploration_tasks)>0)

        # Prefer to open unopened doors
        open_door_tasks = []
        for exploration_task in self.exploration_tasks:
            if isinstance(exploration_task, PolycraftTask.OPEN_DOOR.value):
                open_door_tasks.append(exploration_task)
        if len(open_door_tasks)>0:
            return random.choice(open_door_tasks)

        # No open door tasks? choose a random exploration task
        return random.choice(self.exploration_tasks)

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        # If no active plan - need to create one (this should happen in the beginning of a level)
        if self.active_plan is None:
            self._update_current_state(world_state) # TODO: Maybe this is redundant
            self.active_plan = self.planner.make_plan(self.current_state)
        else: # Check if the active plan is working or if we need to replan
            assert(len(self.current_observation.actions)>0)
            last_action = self.current_observation.actions[-1]
            if last_action.success==False:
                logger.info("Last action failed, replanning...")
                self.failed_actions_in_level = self.failed_actions_in_level+1
                self.active_plan = self.replan(world_state)
                self.active_action = None
            else:
                logger.info("Continue to perform the current plan")

        # If no plan found, choose default action
        if self.active_plan is None or (len(self.active_plan)==0 and self.active_action is None):
            logger.info("No active plan or action has been assigned: choose a default action")
            return self._choose_default_action(world_state)

        # Perform the next action in the plan
        if self.active_action is not None and self.active_action.is_done()==False: # If there is an active action tat is not done, continue doing it
            return self.active_action

        assert(len(self.active_plan)>0)
        action_to_do = self.active_plan.pop(0)
        if isinstance(action_to_do, MacroAction):
            self.active_action = action_to_do
        return action_to_do

    def _should_explore(self, world_state:PolycraftState):
        ''' Consider choosing an exploration action'''
        if len(self.exploration_tasks)>0 and \
                self.failed_actions_in_level % self.exploration_rate == 1:
            return True
        else:
            return False

    def replan(self, world_state:PolycraftState):
        ''' Create a new plan after the active plan failed '''
        if self._should_explore(world_state):
            task = self._choose_exploration_task(world_state)
        else:
            task = PolycraftTask.CRAFT_POGO.create_instance()
        self.meta_model.set_active_task(task)

        # Create new plan from the current state to achieve the current task
        self.planner.make_plan(world_state)

    def _choose_default_action(self, world_state: PolycraftState):
        ''' Choose a default action. Current policy: try to mine something if available.
         Otherwise, try to collect an item. Otherwise do a no-op. '''

        # Try to mine a block
        type_to_cells = world_state.get_type_to_cells()
        # Note: world state may contain new types of cells we do not know. So instead of marking which cells to mine,
        # we only mark which known types we shouldn't
        non_minable_types = [BlockType.AIR.value,
                             BlockType.BEDROCK.value,
                             BlockType.CRAFTING_TABLE.value,
                             BlockType.PLASTIC_CHEST.value,
                             BlockType.TREE_TAP.value,
                             BlockType.WOODER_DOOR.value]
        minable_types = [block_type for block_type in type_to_cells.keys() if block_type not in non_minable_types]
        # Find a minable cell type that is accessible
        while len(minable_types)>0:
            type_index = random.choice(range(len(minable_types)))
            type_to_mine = minable_types.pop(type_index)
            cells = world_state.get_cells_of_type(type_to_mine, only_accessible=True)
            if len(cells)>0:
                cell = random.choice(cells)
                return TeleportToBreakAndCollect(cell)

        # Try to collect an item
        entity_items = world_state.get_entities_of_type(EntityType.ITEM.value)
        while len(entity_items)>0:
            entity_index = random.choice(range(len(entity_items)))
            entity_to_collect = entity_items.pop(entity_index)
            entity_attr = world_state.entities[entity_to_collect]
            entity_cell = coordinates_to_cell(entity_attr["pos"])
            if world_state.game_map[entity_cell]["isAccessible"]:
                return PolyEntityTP(entity_to_collect)

        return PolyNoAction()

    def do(self, action: PolycraftAction, env : Polycraft):
        ''' Perform the given aciton in the given environment '''
        if isinstance(action, MacroAction):
            action.set_current_state(env.get_current_state())
        self.current_observation.actions.append(action)
        next_state, step_cost =  env.act(action)  # Note this returns step cost for the action
        self._update_current_state(next_state)
        self.current_observation.states.append(self.current_state)
        self.current_observation.rewards.append(step_cost)
        return next_state, step_cost

    def _update_current_state(self, new_state: PolycraftState):
        ''' Updates the current state object with the information from the new state.
        Needed because sometimes agents leave/enter rooms.'''
        old_current_state = self.current_state
        self.current_state = new_state
        if old_current_state is not None:
            for cell in old_current_state.game_map:
                if cell not in self.current_state.game_map:
                    self.current_state.game_map[cell] = old_current_state.game_map[cell]

    def do_batch(self, batch_size:int, state:PolycraftState, env:Polycraft):
        ''' Runs a batch of actions from the given state using the given environment.
        Halt after batch_size actions or if the level has been finished. '''
        iteration = 0
        while state.terminal == False and \
                state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value) == 0 and \
                iteration < batch_size:
            action = self.choose_action(state)
            after_state, step_cost = self.do(action, env)
            state = after_state
            iteration = iteration + 1
        return state, step_cost

    def set_meta_model(self, meta_model: MetaModel):
        ''' Update the meta model of the agent '''
        self.meta_model = meta_model
        if isinstance(self.planner, PolycraftPlanner):
            self.planner.meta_model = meta_model

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
            return PolyCraftItem.create_action(random_recipe)
        else:
            raise ValueError("Bad action class {}".format(action_class))


class PolycraftDoNothingAgent(PolycraftHydraAgent):
    ''' An agent that does nothing '''
    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        return PolyNoAction()


class PolycraftManualAgent(PolycraftHydraAgent):
    ''' An agent that queries the user for actions USED FOR DEBUGGING '''
    def __init__(self):
        super().__init__()
        self.command_seq = 0
        self.commands = []

    def choose_action(self, world_state: PolycraftState):
        if self.command_seq<len(self.commands):
            cmd = self.commands[self.command_seq]
            self.command_seq=self.command_seq+1
        else:
            return PolyGiveUp()

        return cmd