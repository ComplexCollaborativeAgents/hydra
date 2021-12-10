import datetime
import random

from agent.consistency.observation import HydraObservation
from agent.planning.polycraft_planning.tasks import *
from agent.planning.polycraft_planning.actions import *
from worlds.polycraft_actions import PolyNoAction, PolyEntityTP, PolyInteract, PolyGiveUp
from agent.planning.pddlplus_parser import *
from agent.hydra_agent import HydraAgent, HydraPlanner, MetaModelRepair
from worlds.polycraft_world import *
from agent.planning.nyx import nyx
import agent.planning.nyx.heuristic_functions as nyx_heuristics

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")

import re

RE_EXTRACT_ACTION_PATTERN = re.compile(r" *(\d*\.\d*):\t*(.*)\t\[0\.0\]")  # Pattern used to extract action time and name from planner output
SAVE_FAILED_PLANS_STATES  = True # If true then whenever a plan fails, we store its domain and problem files

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
        logger.info(f"Planning to achieve task {self.meta_model.active_task}")

        # For debug purposes, print a summary of the current state
        logger.info(f"Summary of the current state\n {state.summary()}")


        try:
            nyx.constants.MAX_GENERATED_NODES = settings.POLYCRAFT_MAX_GENERATED_NODES
            nyx.runner(self.pddl_domain_file,
                       self.pddl_problem_file,
                       ['-vv', '-to:%s' % str(self.timeout), '-noplan', '-search:gbfs', '-custom_heuristic:3', '-th:10',
                        # '-th:%s' % str(self.meta_model.constant_numeric_fluents['time_limit']),
                        '-t:%s' % str(self.delta_t)])
            plan_actions = self.extract_actions_from_plan_trace(self.pddl_plan_file)
            if len(plan_actions) > 0:
                return plan_actions
            else:
                if SAVE_FAILED_PLANS_STATES:
                    saved_state_file = os.path.join(settings.ROOT_PATH, f"polycraft_failed_{self.meta_model.active_task}")
                    logger.info(f"Saving the state we failed to plan for in file {saved_state_file}")
                    with open(saved_state_file, "wb") as out_file:
                        pickle.dump(self.initial_state, out_file)
                return None
        except Exception as e_inst:
            logger.error(f"Exception while running planner. {e_inst}", stack_info=True)
            logger.exception(e_inst)
            print(e_inst)
        return None


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
        logger.info(f'Extracting plan action from  line "{pddl_action_line.strip()}"')
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

        return selected_action_gen.to_pddl_polycraft(binding)

    def _translate_pddl_object_to_poly(self, pddl_obj_name, pddl_obj_type):
        ''' Translate the name of a pddl object to its corresponding name in polycraft, based on its type
            For cell type objects, this means convert from cell_x_y_z to x,y,z
        '''
        assert(pddl_obj_type in [type.name for type in PddlType]) # Currently we only have objects of type cell
        return ",".join(pddl_obj_name.split("_")[1:])

    def extract_actions_from_plan_trace(self, plane_trace_file: str):
        ''' Parses the given plan trace file and outputs the plan '''
        pddl_action_names = [action.name for action in self.pddl_domain.actions]
        action_generators = self.meta_model.get_action_generators(self.initial_state)
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
        self.exploration_rate = 10 # Number of failed actions to endure before trying one exploration task
        self.active_plan = None
        self.current_observation = None
        self.current_state = None # Maintains the agent's knowledge about the current state
        self.failed_actions_in_level = 0 # Count how many actions have failed in a given level
        self.actions_since_planning = 0 # Count how many actions have been performed since we planned last

    def start_level(self, env: Polycraft):
        ''' Initialize datastructures for a new level and perform exploratory actions to get familiar with the current level.
        These actions are needed before calling the planner  '''

        # Explore the level
        env.init_state_information()
        env.populate_current_recipes()
        env.populate_door_to_room_cells()

        current_state = env.get_current_state()

        # Try to interact with all other agents
        for entity_id, entity_attr in current_state.entities.items():
            if entity_attr['type']=='EntityTrader':
                # If trader not accessible, mark getting to it as a possilbe exploration task
                trader_cell = coordinates_to_cell(entity_attr['pos'])
                if current_state.game_map[trader_cell]['isAccessible']==False:
                    continue

                # Move to trader
                tp_action = PolyEntityTP(entity_id,dist=1)
                current_state, step_cost = env.act(current_state, tp_action)

                if tp_action.success==False:
                    trader_cell = coordinates_to_cell(entity_attr['pos'])
                    cell_accessible = current_state.game_map[trader_cell]['isAccessible']
                    logger.info(f"Entity {entity_id} is at cell {trader_cell} whose accessibility is {cell_accessible}, but TP_TO failed.")
                else:
                    # Interact with it
                    interact_action = PolyInteract(entity_id)
                    current_state, step_cost = env.act(current_state, interact_action)
                    assert(interact_action.success)
                    env.current_trades[entity_id] = interact_action.response['trades']['trades']

        # Initialize the current observation and active plan objects
        self.current_state = env.get_current_state()
        self.current_observation = PolycraftObservation() # Start a new observation object for this level
        self.current_observation.states.append(current_state)
        self.observations_list.append(self.current_observation)
        self.env = env
        self.failed_actions_in_level = 0 # Count how many actions have failed in a given level
        self.actions_since_planning = 0 # Count how many actions have been performed since we planned last
        self.active_plan = None

    def _choose_exploration_task(self, world_state: PolycraftState):
        ''' Choose an exploration task to perform '''
        exploration_tasks = []
        # Explore other rooms
        for door_cell, room_cells in world_state.door_to_room_cells.items():
            if len(room_cells)==1: # Room not explored
                task = PolycraftTask.EXPLORE_DOOR.create_instance()
                task.door_cell = door_cell
                exploration_tasks.append(task)

        # Prefer to open door to unexplored rooms
        if len(exploration_tasks)>0:
            return random.choice(exploration_tasks)

        # Reach unreachable traders
        for entity_id, entity_attr in world_state.entities.items():
            if entity_attr['type']==EntityType.TRADER.value:
                # If trader not accessible, mark getting to it as a possilbe exploration task
                trader_cell = coordinates_to_cell(entity_attr['pos'])
                if world_state.get_known_cells()[trader_cell]['isAccessible']==False:
                    reach_cell_task = PolycraftTask.MAKE_CELL_ACCESSIBLE.create_instance()
                    reach_cell_task.cell = trader_cell
                    exploration_tasks.append(reach_cell_task)

        # Open safe
        safe_cells= world_state.get_cells_of_type(BlockType.SAFE.value)
        for safe_cell in safe_cells:
            safe_ok = True
            for old_action in self.current_observation.actions:
                if isinstance(old_action, OpenSafeAndCollect):
                    if old_action.cell==safe_cell and old_action.safe_opened:
                        logger.info(f"Safe {safe_cell} has already been opened - no need to re-explore it")
                        safe_ok=False
                        break
            if safe_ok:
                exploration_tasks.append(CollectFromSafeTask(safe_cell))

        # No open door tasks? choose a random exploration task
        return random.choice(exploration_tasks)

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        if len(self.current_observation.actions)==0:
            if self.active_plan is None or len(self.active_plan)==0:
                logger.info("Running planner to generate initial plan")
                self.active_plan = self.plan()
            else:
                logger.info("Initial plan set externally (should be used only for testing and debugging")

        if self.should_replan():
            logger.info("Replanning...")
            self.active_plan = self.replan(world_state)
            self.actions_since_planning = 0
        else:
            logger.info(f"Continue to perform the current plan. Next action is {self.active_plan[0]}")

        # If no plan found, choose default action
        if self.active_plan is None:
            logger.info("No active plan or action has been assigned: choose a default action")
            return self._choose_default_action(world_state)

        # Perform the next action in the plan
        assert(len(self.active_plan)>0)
        return self.active_plan.pop(0)

    def plan(self, active_task = None)->list:
        ''' Generate a plan for the active task '''
        if active_task is not None:
            self.set_active_task(active_task)
        self._adapt_to_novel_objects(self.current_state)
        return self.planner.make_plan(self.current_state)

    def should_replan(self):
        ''' Decide if we need to update the active plan'''
        if self.active_plan is None or len(self.active_plan)==0:
            return True
        if len(self.current_observation.actions) > 0:
            last_action = self.current_observation.actions[-1]
            if last_action.success == False:
                return True
        return False

    def _should_explore(self, world_state:PolycraftState):
        ''' Consider choosing an exploration action'''
        if self.failed_actions_in_level > 0 and \
                self.failed_actions_in_level % self.exploration_rate == 0:
            return True
        else:
            return False

    def replan(self, world_state:PolycraftState):
        ''' Create a new plan after the active plan failed '''
        if self._should_explore(world_state)==False:
            task = PolycraftTask.CRAFT_POGO.create_instance()
            plan = self.plan(active_task=task)
            if plan is not None:
                logger.info(f"Found a plan for main task ({task})")
                return plan
            else:
                logger.info(f"Failed to find a plan for {task}. Try to explore.")

        # Either decided to explore or failed to find a plan to craft the pogo: explore
        for i in range(settings.POLYCRAFT_MAX_EXPLORATION_PLANNING_ATTEMPTS):
            task = self._choose_exploration_task(world_state)
            if task.is_feasible(world_state)==False:
                logger.info(f"Chosen exploration task {task} but it is not feasible in the current state")
                return None
            plan = self.plan(active_task=task)
            if plan is not None:
                logger.info(f"Found a plan for exploration task {task}")
                return plan
            else:
                logger.info(f"Failed to find a plan for exploration task {task}")
        logger.info("No plan found for any task :(")
        return None

    def set_active_task(self, task: PolycraftTask):
        ''' Sets the active task, generate a plan to achieve it and updates the active plan '''
        self.meta_model.set_active_task(task)

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
        possible_default_actions = [WaitForLogs()]
        for type_to_mine in minable_types:
            cells = world_state.get_cells_of_type(type_to_mine, only_accessible=True)
            if len(cells)>0:
                cell = random.choice(cells)
                possible_default_actions.append(TeleportToBreakAndCollect(cell))

        # Try to collect an item
        entity_items = world_state.get_entities_of_type(EntityType.ITEM.value)
        while len(entity_items)>0:
            entity_index = random.choice(range(len(entity_items)))
            entity_to_collect = entity_items.pop(entity_index)
            entity_attr = world_state.entities[entity_to_collect]
            entity_cell = coordinates_to_cell(entity_attr["pos"])
            if world_state.game_map[entity_cell]["isAccessible"]:
                possible_default_actions.append(PolyEntityTP(entity_to_collect))
                break

        return random.choice(possible_default_actions)

    def do(self, action: PolycraftAction, env : Polycraft):
        ''' Perform the given aciton in the given environment '''
        self.current_observation.actions.append(action)
        if self.meta_model.active_task.is_feasible(self.current_state)==False:
            logger.error(f"Current task not feasible - False action") # TODO: Design choice: what to do this in this case

        next_state, step_cost =  env.act(self.current_state, action)  # Note this returns step cost for the action

        if action.success == False:
            logger.info(f"Action{action} failed: {action.response}")
            self.failed_actions_in_level = self.failed_actions_in_level+1
        else:
            self.actions_since_planning = self.actions_since_planning +1
            logger.info(f"Action{action} finished successfully!")
        self.current_state = next_state
        self.current_observation.states.append(self.current_state)
        self.current_observation.rewards.append(step_cost)
        return next_state, step_cost

    def _update_current_state(self, new_state: PolycraftState):
        ''' Updates the current state object with the information from the new state.
        Needed because sometimes agents leave/enter rooms.'''
        for cell_id, cell_attr in new_state.game_map.items():
            for door_cell_id, room_game_map in self.current_state.door_to_room_cells.items():
                if cell_id in room_game_map:
                    room_game_map[cell_id] = cell_attr
        self.current_state = new_state

    def do_batch(self, batch_size:int, state:PolycraftState, env:Polycraft, time_limit=0):
        ''' Runs a batch of actions from the given state using the given environment.
        Halt after batch_size actions or if the level has been finished. '''
        iteration = 0
        start_time = time.time()
        while state.terminal == False and \
                state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value) == 0 and \
                iteration < batch_size and \
                (time_limit==0 or (time.time()-start_time<time_limit)):
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

    def _adapt_to_novel_objects(self, state):
        ''' Computes the novel blocks, items, and entities in the current state'''
        for block_type in state.get_type_to_cells():
            if block_type not in self.meta_model.block_type_to_idx:
                logger.info(f"Novel block type detected - {block_type}")
                self.meta_model.introduce_novel_block_type(block_type)
        for item_type in state.get_item_to_count():
            if item_type not in self.meta_model.item_type_to_idx:
                logger.info(f"Novel item type detected - {item_type}")
                self.meta_model.introduce_novel_inventory_item_type(item_type)
        for entity, entity_attr in state.entities.items():
            entity_type = entity_attr["type"]
            if entity_type not in [entity.value for entity in EntityType]:
                logger.info(f"Novel entity type detected - {entity_type}")
                self.meta_model.introduce_novel_entity_type(entity_type)

    def novelty_detection(self, report_novelty=True, only_current_state=True):
        ''' Computes the likelihood that the current observation is novel '''
        if only_current_state==False:
            novelties = set()
            last_observation = self.observations_list[-1]
            for i, state in enumerate(last_observation.states):
                novelties.update(self._detect_unknown_objects(state))
        else:
            novelties = self._detect_unknown_objects(self.current_state)

        if len(novelties)>0:
            novelty_characterization = "\n".join(novelties)
            novelty_likelihood = 1.0
            if report_novelty:
                self.env.poly_client.REPORT_NOVELTY(level="1", confidence= f"{novelty_likelihood}", user_msg = novelty_characterization)
        else:
            novelty_characterization = ""
            novelty_likelihood = 0.0
        return novelty_likelihood, novelty_characterization

    def _detect_unknown_objects(self, state):
        ''' Returns a list of unknown object novelties identified in the given state '''
        novelties = set()
        for block_type in state.get_type_to_cells():
            if block_type not in [type.value for type in BlockType]:
                novelties.add(f"Block type {block_type} unknown")
        for item_type in state.get_item_to_count():
            if item_type not in [type.value for type in ItemType]:
                novelties.add(f"Item type {item_type} is unknown")
        for entity, entity_attr in state.entities.items():
            entity_type = entity_attr["type"]
            if entity_type not in [entity.value for entity in EntityType]:
                novelties.add(f"Entity type {entity_type} unknown")
        return novelties

    def should_repair(self, observation):
        ''' Choose if the agent should repair its meta model based on the given observation '''
        raise NotImplementedError()

    def repair_meta_model(self, observation):
        ''' Call the repair object to repair the current meta model '''
        raise NotImplementedError()

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