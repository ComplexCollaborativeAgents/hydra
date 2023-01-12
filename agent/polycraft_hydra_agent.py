import datetime
import time
from typing import List, Set, Tuple

from agent.consistency.nyx_pddl_simulator import NyxPddlPlusSimulator
from agent.consistency.polycraft_episode_log import PolycraftEpisodeLog
from agent.planning.polycraft_planning.tasks import *
from agent.planning.polycraft_planning.polycraft_macro_actions import *
from agent.repair.polycraft_repair import PolycraftMetaModelRepair
from utils.stats import PolycraftAgentStats, PolycraftDetectionStats
from worlds.polycraft_actions import PolyNoAction, PolyEntityTP, PolyInteract, PolyGiveUp
from agent.planning.pddlplus_parser import *
from agent.hydra_agent import HydraAgent, HydraPlanner
from worlds.polycraft_world import *
from agent.planning.nyx import nyx
import agent.planning.nyx.heuristic_functions as nyx_heuristics
import re

# logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")


RE_EXTRACT_ACTION_PATTERN = re.compile(
    r" *(\d*\.\d*):\t*(.*)\t\[0\.0\]")  # Pattern used to extract action time and name from planner output
SAVE_FAILED_PLANS_STATES = True  # If true then whenever a plan fails, we store its domain and problem files


class PolycraftPlanner(HydraPlanner):
    """ Planner for the polycraft domain"""

    def __init__(self, meta_model:PolycraftMetaModel,
                 planning_path:str=settings.POLYCRAFT_PLANNING_DOCKER_PATH):
        super().__init__(meta_model)
        self.pddl_domain = None
        self.pddl_problem = None
        self.initial_state = None
        self.current_problem_prefix = None
        self.planning_path = planning_path
        self.pddl_problem_file = "%s/polycraft_prob.pddl" % str(self.planning_path)
        self.pddl_domain_file = "%s/polycraft_domain.pddl" % str(self.planning_path)
        self.pddl_plan_file = "%s/plan_polycraft_prob.pddl" % str(self.planning_path)

        self.delta_t = settings.POLYCRAFT_DELTA_T
        self.timeout = settings.POLYCRAFT_TIMEOUT

    def make_plan(self, state: PolycraftState, problem_complexity:int=0) -> List[PolycraftAction]:
        """Generate a sequence of Polycraft actions as a plan

        Args:
            state (PolycraftState): Polycraft World state object
            problem_complexity (int, optional): Degree of complexity the planner will consider the problem with (CURRENTLY UNUSED). Defaults to 0.

        Returns:
            List[PolycraftAction]: 
        """
        if settings.NO_PLANNING:
            self.current_problem_prefix = datetime.datetime.now().strftime(
                "%y%m%d_%H%M%S")  # need a prefix for observations
            return []
        self.initial_state = state
        self.pddl_problem = self.meta_model.create_pddl_problem(state)
        self.pddl_domain = self.meta_model.create_pddl_domain(state)
        self.write_pddl_file(self.pddl_problem, self.pddl_domain)
        nyx_heuristics.active_heuristic = self.meta_model.get_nyx_heuristic(state, self.meta_model)
        logger.info(f"Planning to achieve task {self.meta_model.active_task}")

        # For debug purposes, print a summary of the current state
        logger.info(f"Summary of the current state\n {state.summary()}")

        try:
            nyx.constants.MAX_GENERATED_NODES = settings.POLYCRAFT_MAX_GENERATED_NODES
            _, self.explored_states = nyx.runner(self.pddl_domain_file,
                                                 self.pddl_problem_file,
                                                 ['-vv', '-to:%s' % str(self.timeout), '-noplan', '-search:gbfs',
                                                  '-custom_heuristic:3', '-th:10',
                                                  # '-th:%s' % str(self.meta_model.constant_numeric_fluents['time_limit']),
                                                  '-t:%s' % str(self.delta_t)])
            plan_actions = self.extract_actions_from_plan_trace(self.pddl_plan_file)
            if len(plan_actions) > 0:
                return plan_actions
            else:
                if SAVE_FAILED_PLANS_STATES:
                    saved_state_file = os.path.join(settings.ROOT_PATH,
                                                    f"polycraft_failed_{self.meta_model.active_task}")
                    logger.info(f"Saving the state we failed to plan for in file {saved_state_file}")
                    with open(saved_state_file, "wb") as out_file:
                        pickle.dump(self.initial_state, out_file)
                return []
        except Exception as e_inst:
            logger.error(f"Exception while running planner. {e_inst}", stack_info=True)
            logger.exception(e_inst)
            print(e_inst)
        return []

    def write_pddl_file(self, pddl_problem: PddlPlusProblem, pddl_domain: PddlPlusDomain):
        """Write pddl problem + domain to file

        Args:
            pddl_problem (PddlPlusProblem): PDDL Problem object that contains the objects and goal.
            pddl_domain (PddlPlusDomain): PDDL Domain object that contains predicates, functions, constants, etc.
        """

        problem_exporter = PddlProblemExporter()
        problem_exporter.to_file(pddl_problem, self.pddl_problem_file)
        domain_exporter = PddlDomainExporter()
        domain_exporter.to_file(pddl_domain, self.pddl_domain_file)
        if settings.DEBUG:
            self.current_problem_prefix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            cmd = "mkdir -p {}/trace/problems".format(self.planning_path)
            subprocess.run(cmd, shell=True)
            problem_exporter.to_file(pddl_problem,
                                     "{}/trace/problems/{}_problem.pddl".format(self.planning_path,
                                                                                self.current_problem_prefix))
            domain_exporter.to_file(pddl_domain,
                                    "{}/trace/problems/{}_domain.pddl".format(self.planning_path,
                                                                              self.current_problem_prefix))

    def _get_poly_action_from_pddl_action(self, pddl_action_line: str, action_generators: List[PddlPolycraftActionGenerator]) -> PolycraftAction:
        """Generates a PolycraftAction using the given list of action generators based on the given
        pddl action line. 

        Args:
            pddl_action_line (str): raw PDDL action string
            action_generators (List[PddlPolycraftActionGenerator]): List of relevant action generators

        Returns:
            PolycraftAction: translated Polycraft action
        """

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
            if action_gen.pddl_name.replace(":", "_") == action_name:
                selected_action_gen = action_gen
                break
        assert (selected_action_gen is not None)
        pddl_action = selected_action_gen.to_pddl(self.meta_model)

        # Handle parameters if needed
        binding = dict()
        if len(action_parts) > 1:
            params_parts = action_parts[1:]
            pddl_parameters_list = pddl_action.parameters[0]
            assert (len(params_parts) * 3 == len(pddl_parameters_list))
            for i, param in enumerate(params_parts):
                param_value = param.strip()
                param_name = pddl_parameters_list[i * 3].strip()
                param_type = pddl_parameters_list[i * 3 + 2].strip()
                binding[param_name] = self._translate_pddl_object_to_poly(param_value, param_type)

        return selected_action_gen.to_pddl_polycraft(binding)

    def _translate_pddl_object_to_poly(self, pddl_obj_name:str, pddl_obj_type:PddlType) -> str:
        """Translate the name of a pddl object to its corresponding name in polycraft, based on its type
            For cell type objects, this means convert from cell_x_y_z to x,y,z

        Args:
            pddl_obj_name (str): Name of the PDDL object
            pddl_obj_type (PddlType): PddlType of the object

        Returns:
            str: Polycraft object string
        """
        assert (pddl_obj_type in [tp.name for tp in PddlType])  # Currently, we only have objects of type cell
        return ",".join(pddl_obj_name.split("_")[1:])

    def extract_actions_from_plan_trace(self, plan_trace_file: str) -> List[PolycraftAction]:
        """Parses the given plan trace file and outputs the plan

        Args:
            plan_trace_file (str): filepath of the plan trace

        Returns:
            List[PolycraftAction]: List of actions from the plan trace
        """
        pddl_action_names = [action.name for action in self.pddl_domain.actions]
        action_generators = self.meta_model.active_task.create_relevant_actions(self.initial_state, self.meta_model)
        plan_actions = []
        with open(plan_trace_file) as fp:
            for i, line in enumerate(fp):
                if "No Plan Found!" in line:
                    logger.info("No plan found")
                    return []

                action_in_plan = None
                for action_name in pddl_action_names:
                    if action_name in line:
                        action_in_plan = action_name
                        break

                if action_in_plan:
                    plan_actions.append(self._get_poly_action_from_pddl_action(line, action_generators))
        return plan_actions


class PolycraftHydraAgent(HydraAgent):
    """ A Hydra agent for Polycraft of all the Hydra agents """
    meta_model: PolycraftMetaModel
    planner: PolycraftPlanner
    meta_model_repair: PolycraftMetaModelRepair

    current_state: PolycraftState
    current_log: PolycraftEpisodeLog
    current_stats: PolycraftAgentStats
    current_detection: PolycraftDetectionStats

    agent_stats: List[PolycraftAgentStats]
    novelty_stats: List[PolycraftDetectionStats]

    #TODO: Investigate necessity of having a "consistency" member variable when it is included in meta_model_repair
    def __init__(self):
        super().__init__()
        self.meta_model = PolycraftMetaModel(active_task=PolycraftTask.CRAFT_POGO.create_instance())
        self.planner = PolycraftPlanner(self.meta_model)
        self.meta_model_repair = PolycraftMetaModelRepair(self.meta_model)

        self.exploration_rate = 10  # Number of failed actions to endure before trying one exploration task
        self.current_log = None
        self.current_state = None  # Maintains the agent's knowledge about the current state
        self.new_level_time = 600  # If the timeout for a game has passed, start a new game.
        
        self.novelty_reported = False  # have we reported novelty?
        self.novelty_explored = False  # after exploring a new type of brick\item\entity, should attempt to repair.
        
        self.objects_to_explore = []  # New objects that might be worth exploring
        self.unknown_traders = []  # Traders we haven't asked for recipes yet

        self._init_new_episode_stats()

    def _init_new_episode_stats(self):
        """ Initialize a new set of stats objects for the new episode
        """
        self.current_stats = PolycraftAgentStats(episode_start_time=time.time())
        self.current_detection = PolycraftDetectionStats()

    def episode_init(self, world: Polycraft):
        """Perform setup for the agent at the beginning of an episode. 
            Initialize datastructures for a new level and perform exploratory actions to get familiar with the current level.
            These actions are needed before calling the planner.

        Args:
            world (Polycraft): reference to domain world object
        """

        self._init_new_episode_stats()

        self.current_stats.episode_start_time = time.time()

        # Explore the level
        world.init_state_information()
        world.populate_current_recipes()
        world.populate_door_to_room_cells()

        current_state = world.get_current_state()

        # Try to interact with all other agents
        for entity_id in current_state.entities.keys():
            if self._is_entity_white_listed(current_state, entity_id):
                self._interact_with_enttiy(entity_id, current_state, world)

        self.current_state = world.get_current_state()
        self.current_log = PolycraftEpisodeLog(self.meta_model)  # Start a new observation object for this level
        self.current_log.states.append(self.current_state)
        self.episode_logs.append(self.current_log)

        self.active_plan = []
        self.set_active_task(PolycraftTask.CRAFT_POGO.create_instance())

    def episode_end(self) -> PolycraftDetectionStats:
        """Cleans up agent and prepares it for the next episode

        Returns:
            PolycraftDetectionStats: The novelty detection stats for the latest episode
        """
        return super().episode_end()

    def _interact_with_enttiy(self, entity_id: str, current_state: PolycraftState, world: Polycraft):
        """Interacts with an entity. If the entity is a trader, attempt to log possible trades. 

        Args:
            entity_id (str): ID of the entity to interact with
            current_state (PolycraftState): Polycraft World state object
            world (Polycraft): Polycraft world environment
        """
        # Move to entity
        tp_action = PolyEntityTP(entity_id, dist=1)
        current_state, step_cost = world.act(current_state, tp_action)

        if not tp_action.success:
            trader_cell = coordinates_to_cell(current_state.entities[entity_id]['pos'])
            cell_accessible = current_state.game_map[trader_cell]['isAccessible']
            logger.info(
                f"Entity {entity_id} is at cell {trader_cell} whose accessibility is {cell_accessible}, "
                f"but TP_TO failed.")
        else:
            # Interact with it
            interact_action = PolyInteract(entity_id)
            current_state, step_cost = world.act(current_state, interact_action)
            assert interact_action.success
            if interact_action.response.get('trades') is not None:
                # Discovered possible trades!
                world.current_trades[entity_id] = interact_action.response['trades']['trades']
            elif current_state.entities[entity_id]['type'] == 'EntityTrader':
                # Entity is a trader, but might be busy. Mark for trying again later.
                self.unknown_traders.append(entity_id)

    def _is_entity_white_listed(self, state: PolycraftState, entity_id: str):
        return state.entities[entity_id] and state.entities[entity_id]["type"] and "trader" in state.entities[entity_id]["type"].lower()

    def _choose_exploration_action(self, world_state: PolycraftState) -> PolycraftAction:
        """If there are new objects to explore, choose one.

        Args:
            world_state (PolycraftState): Polycraft World state object

        Returns:
            PolycraftAction: Exploration action to take (can be a macro action)
        """
        exploration_actions = []
        # Try out new objects
        for obj in self.objects_to_explore:
            if obj in world_state.get_type_to_cells():
                cells = world_state.get_cells_of_type(obj, only_accessible=True)
                if cells:
                    cell_to_break = cells[0]
                    exploration_actions.append((obj, TeleportToBreakAndCollect(cell_to_break)))
            elif obj in world_state.get_item_to_count():
                exploration_actions.append((obj, SelectAndUse(obj)))
            else:
                # Must be an entity
                exploration_actions.append((obj, TeleportToAndInteract(obj,
                                                                       coordinates_to_cell(
                                                                           world_state.entities[obj]['pos']))))

        # Prefer new objects
        if len(exploration_actions) > 0:
            obj, action = random.choice(exploration_actions)
            self.objects_to_explore.remove(obj)
            self.novelty_explored = True
            return action
        # else
        return None

    def _choose_exploration_task(self, world_state: PolycraftState) -> PolycraftTask:
        """Choose an exploration task to perform

        Args:
            world_state (PolycraftState): Polycraft World state object

        Returns:
            PolycraftTask: Exploration task for the agent to focus on
        """
        exploration_tasks = []

        # Explore other rooms
        for door_cell, room_cells in world_state.door_to_room_cells.items():
            if len(room_cells) == 1:  # Room not explored
                task = PolycraftTask.EXPLORE_DOOR.create_instance()
                task.door_cell = door_cell
                exploration_tasks.append(task)

        # Prefer to open door to unexplored rooms
        if len(exploration_tasks) > 0:
            return random.choice(exploration_tasks)

        # Reach unreachable traders
        for entity_id, entity_attr in world_state.entities.items():
            if entity_attr['type'] == EntityType.TRADER.value:
                # If trader not accessible, mark getting to it as a possible exploration task
                trader_cell = coordinates_to_cell(entity_attr['pos'])
                if not world_state.get_known_cells()[trader_cell]['isAccessible']:
                    reach_cell_task = PolycraftTask.MAKE_CELL_ACCESSIBLE.create_instance()
                    reach_cell_task.cell = trader_cell
                    exploration_tasks.append(reach_cell_task)

        # Open safe
        safe_cells = world_state.get_cells_of_type(BlockType.SAFE.value)
        if len(self.episode_logs) > 2:
            # Only go to safe if we haven't checked in last three tasks?
            for safe_cell in safe_cells:
                safe_ok = True
                # The following lines removed to accommodate the "thief" novelty
                for back_index in range(-1, -4, -1):
                    for old_action in self.episode_logs[back_index].actions:
                        if str(old_action).find('OpenSafeAndCollect') > -1:
                            logger.info(f'Safe {old_action.cell} already accessed, and is {old_action.safe_opened}')
                            if old_action.cell == safe_cell and old_action.safe_opened:
                                logger.info(f"Safe {safe_cell} has already been opened - no need to re-explore it")
                                safe_ok = False
                                break
                if safe_ok:
                    exploration_tasks.append(CollectFromSafeTask(safe_cell))

        # No open door tasks? choose a random exploration task
        logger.info(f"possible exploration tasks: {exploration_tasks}")
        return random.choice(exploration_tasks)

    def level_timed_out(self, world: Polycraft) -> bool:
        """Checks whether the level should time out, returns True if we should give up. 

        Args:
            world (Polycraft): Polycraft World object

        Returns:
            bool: Whether or not the time limit has been reached
        """
        if time.time() - self.current_stats.episode_start_time > self.new_level_time:
            if not self.novelty_reported:
                world.poly_client.REPORT_NOVELTY(level="1", confidence="50",
                                                    user_msg='Something that made the agent plan for too long. ')
                self.novelty_reported = True
            logger.info("Level timed out!")
            return True
        return False

    def choose_action(self, world_state: PolycraftState, world: Polycraft) -> PolycraftAction:
        """Choose which action to perform in the given state

        Args:
            world_state (PolycraftState): Polycraft World state object
            world (Polycraft): Polycraft World object

        Returns:
            PolycraftAction: action to take
        """

        # Update current state with most recent values
        self._update_current_state(world_state)

        if self.level_timed_out(world):
            self.active_plan = []
            return PolyNoAction()

        if len(self.active_plan) == 0:
            logger.info("No current plan, plan to create pogostick")
            self.set_active_task(PolycraftTask.CRAFT_POGO.create_instance())
            # plan logic utilizes current state - but current state is not updated until the action is complete
            self.active_plan = self.plan_logic(world_state, world)

        if self.level_timed_out(world):
            self.active_plan = []
            return PolyNoAction()

        # If no plan found, choose default action
        if len(self.active_plan) == 0:
            logger.info("No active plan or action has been assigned: choose a default action")
            return self._choose_default_action(world_state)
        else:
            logger.info(f"Continue to perform the current plan. Next action is {self.active_plan[0]}")

        # Perform the next action in the plan
        assert (len(self.active_plan) > 0)
        return self.active_plan.pop(0)

    def plan(self, world: Polycraft, active_task:PolycraftTask=None) -> List[PolycraftAction]:
        """Generate a plan for the active task

        Args:
            world (Polycraft): Polycraft World object
            active_task (PolycraftTask, optional): Task used to generate planned actions. Defaults to None.

        Returns:
            List[PolycraftAction]: List of Polycraft Actions that comprise the plan
        """
        # Planning is the most time-intensive process, so check before and after to make sure we're not out.
        if active_task is not None:
            self.set_active_task(active_task)
        self._detect_unknown_objects(self.current_state, world)

        start_time = time.perf_counter()
        plan = self.planner.make_plan(self.current_state)
        self.current_stats.planning_time = time.perf_counter() - start_time
        self.current_stats.explored_states = self.planner.explored_states
        self.current_stats.plan_action_length = len(plan)
        return plan

    def _should_explore(self, world_state: PolycraftState) -> bool:
        """Consider choosing an exploration action

        Args:
            world_state (PolycraftState): Polycraft World state object

        Returns:
            bool: Whether or not the agent should explore
        """
        if self.current_stats.failed_actions > 0 and \
                self.current_stats.failed_actions % self.exploration_rate == 0:
            self.current_stats.failed_actions = 0  # reset since we are changing task.
            logger.info("Exploration chosen")
            return True
        else:
            logger.info("Proceeding to main task")
            return False

    def plan_logic(self, world_state: PolycraftState, world: Polycraft) -> List[PolycraftAction]:
        """Create a new plan after the active plan failed

        Args:
            world_state (PolycraftState): Polycraft World state object
            world (Polycraft): Polycraft world

        Returns:
            List[PolycraftAction]: List of Polycraft Actions for the new plan
        """
        if self.should_repair(world_state, world):
            logger.info("Repairing model!")
            self.repair_meta_model(world_state, world)

        if not self._should_explore(world_state):
            task = PolycraftTask.CRAFT_POGO.create_instance()
            plan = self.plan(world, active_task=task)
            if len(plan) > 0:
                logger.info(f"Found a plan for main task ({task})")
                return plan
            else:
                logger.info(f"Failed to find a plan for {task}. Try to explore.")

        # Either decided to explore or failed to find a plan to craft the pogo: explore
        for i in range(settings.POLYCRAFT_MAX_EXPLORATION_PLANNING_ATTEMPTS):
            # First, explore any trades we've missed.
            if len(self.unknown_traders) > 0:
                self._interact_with_enttiy(self.unknown_traders.pop(), world_state, world)
                break
            action = self._choose_exploration_action(world_state)
            if action is not None:
                return [action]
            # else
            task = self._choose_exploration_task(world_state)
            if not task.is_feasible(world_state):
                logger.info(f"Chosen exploration task {task} but it is not feasible in the current state")
                return None
            plan = self.plan(world, active_task=task)
            # After exploring try to create pogostick again
            if len(plan) > 0:
                logger.info(f"Found a plan for exploration task {task}")
                return plan
            else:
                logger.info(f"Failed to find a plan for exploration task {task}")
        logger.info("No plan found for any task :(")
        return []

    def set_active_task(self, task: PolycraftTask):
        """Sets the active task, generate a plan to achieve it and updates the active plan

        Args:
            task (PolycraftTask): The new active task
        """
        if task != self.meta_model.active_task:# task changed, start a new observation set
            self.current_log = PolycraftEpisodeLog(self.meta_model)
            self.current_log.states.append(self.current_state)
            self.episode_logs.append(self.current_log)
        self.meta_model.set_active_task(task)

    def _choose_default_action(self, world_state: PolycraftState) -> PolycraftAction:
        """Choose a default action. Current policy: try to mine something if available.
         Otherwise, try to collect an item. Otherwise, do a no-op.

        Args:
            world_state (PolycraftState): Polycraft World state object

        Returns:
            PolycraftAction: the default action to take
        """

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
        # possible_default_actions = [WaitForLogs()]
        possible_default_actions = []
        for type_to_mine in minable_types:
            cells = world_state.get_cells_of_type(type_to_mine, only_accessible=True)
            if len(cells) > 0:
                cell = random.choice(cells)
                possible_default_actions.append(TeleportToBreakAndCollect(cell))

        # Try to collect an item
        entity_items = world_state.get_entities_of_type(EntityType.ITEM.value)
        while len(entity_items) > 0:
            entity_index = random.choice(range(len(entity_items)))
            entity_to_collect = entity_items.pop(entity_index)
            entity_attr = world_state.entities[entity_to_collect]
            entity_cell = coordinates_to_cell(entity_attr["pos"])
            if world_state.game_map[entity_cell]["isAccessible"]:
                possible_default_actions.append(PolyEntityTP(entity_to_collect))
                break

        logger.info(f"DEFAULT ACTIONS POSSIBLE -- {possible_default_actions}")
        return random.choice(possible_default_actions)

    def do(self, action: PolycraftAction, world: Polycraft) -> Tuple[PolycraftState, int]:
        """Perform the given action in the given environment

        Args:
            action (PolycraftAction): PolycraftAction to perform
            world (Polycraft): Polycraft World object

        Returns:
            Tuple[PolycraftState, int]: The state and step cost after the performed action
        """
        self.current_log.actions.append(action)
        if not self.meta_model.active_task.is_feasible(self.current_state):
            logger.error(
                f"Current task not feasible - False action")  # TODO: Design choice: what to do this in this case

        next_state, step_cost = world.act(self.current_state, action)  # Note this returns step cost for the action
        action.start_at = self.current_log.time_so_far

        if not action.success:
            logger.info(f"Action{action} failed: {action.response}")
            self.current_stats.failed_actions = self.current_stats.failed_actions + 1
        else:
            self.current_stats.actions_since_planning += 1
            logger.info(f"Action{action} finished successfully!")
        self.current_state = next_state
        self.current_log.states.append(self.current_state)
        self.current_log.rewards.append(step_cost)

        if self.current_state.is_terminal():
            self.current_stats.success = self.current_state.passed

        return next_state, step_cost

    def _update_current_state(self, new_state: PolycraftState):
        """Updates the current state object with the information from the new state.
        Needed because sometimes agents leave/enter rooms and entities may appear/disappear.

        Args:
            new_state (PolycraftState): Polycraft World state object to update
        """
        # for cell_id, cell_attr in new_state.game_map.items():
        #     for door_cell_id, room_game_map in self.current_state.door_to_room_cells.items():
        #         if cell_id in room_game_map:
        #             room_game_map[cell_id] = cell_attr
        # self.current_state = new_state

        self.current_state.entities = copy.copy(new_state.entities)

    def do_batch(self, batch_size: int, state: PolycraftState, world: Polycraft, time_limit=0) -> Tuple[PolycraftState, int]:
        """Runs a batch of actions from the given state using the given environment.
        Halt after batch_size actions or if the level has been finished.

        Args:
            batch_size (int): Number of iterations.
            state (PolycraftState): Polycraft World state object
            world (Polycraft): Polycraft World object
            time_limit (int, optional): Time limit before process exits. Defaults to 0.

        Returns:
            Tuple[PolycraftState, int]: The state and step cost after the batch actions 
        """
        iteration = 0
        start_time = time.time()
        step_cost = 0
        while not state.terminal and \
                state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value) == 0 and \
                iteration < batch_size and \
                (time_limit == 0 or (time.time() - start_time < time_limit)):
            action = self.choose_action(state)
            after_state, step_cost = self.do(action, world)
            state = after_state
            iteration = iteration + 1
        return state, step_cost

    def set_meta_model(self, meta_model: PolycraftMetaModel):
        """Update the metamodel of the agent

        Args:
            meta_model (PolycraftMetaModel): Meta Model to set
        """
        self.meta_model = meta_model
        if isinstance(self.planner, PolycraftPlanner):
            self.planner.meta_model = meta_model

    def _detect_if_current_episode_is_novel(self, state: PolycraftState, world: Polycraft, only_current_state:bool = True) -> Tuple[float, str]:
        """Computes the likelihood that the current observation is novel

        Args:
            state (PolycraftState): Polycraft World state object
            world (Polycraft): World object for reporting novelty
            only_current_state (bool, optional): Detects novelty only in the current state. Defaults to True.

        Returns:
            Tuple[float, str]: Novelty likelihood and novelty characterization
        """
        stats = PolycraftDetectionStats()
        novelty_likelihood = 0.0
        
        if not only_current_state:
            novelties = set()
            last_observation = self.current_log
            for i, logged_state in enumerate(last_observation.states):
                novelties.update(self._detect_unknown_objects(logged_state, world))
        else:
            novelties = self._detect_unknown_objects(state, world)

        if len(novelties) > 0:
            novelty_characterization = "\n".join(novelties)
            stats.pddl_prob = 1.0
            stats.novelty_detected = True
            # new objects detected - no need to report them twice
            report_novelty = False
        else:
            novelty_characterization = ""
            self.meta_model_repair.current_delta_t = settings.POLYCRAFT_DELTA_T
            self.meta_model_repair.current_meta_model = self.meta_model
            logger.info(state.summary())
            if len(self.current_log.states) > 0:
                curr_inconsistency = self.meta_model_repair.consistency_estimator.consistency_from_observations(
                    self.meta_model, NyxPddlPlusSimulator(), self.current_log, settings.POLYCRAFT_DELTA_T)
                logger.info(f'Computed inconsistency: {curr_inconsistency}')
                if curr_inconsistency > settings.POLYCRAFT_CONSISTENCY_THRESHOLD:
                    novelty_likelihood = curr_inconsistency / settings.POLYCRAFT_CONSISTENCY_THRESHOLD
                    stats.novelty_detected = True
                    novelty_characterization = f'Plan simulation does not match observations. ' \
                                               f'Mismatch value: {curr_inconsistency}'
                else:
                    stats.pddl_prob = 0.0
                    stats.novelty_detected = False
            else:
                stats.pddl_prob = 0.0
                stats.novelty_detected = False

        return novelty_likelihood, novelty_characterization

    def report_novelty(self, state: PolycraftState, world: Polycraft, report_novelty:bool = True):
        """
        Ideally should be called only by *dispatcher* to report the novelty
        acts as an encapsulation for _detect_novelty internal function
        Reports the detected novelty

        Args:
            state (PolycraftState): State in which novelty has to be verified
            world (Polycraft): World object for reporting novelty
            report_novelty (bool, optional): Whether or not to report novelty if applicable. Defaults to True.

        No Return arguments. Updates agent objects for novelty characterization stats
        and novelty reporting for the tournament.
        """
        novelty_likelihood, novelty_characterization = self._detect_if_current_episode_is_novel(state, world)

        if stats.novelty_detection and report_novelty and not self.novelty_reported:
            world.poly_client.REPORT_NOVELTY(level="0", confidence=f"{novelty_likelihood}",
                                            user_msg=novelty_characterization)
            self.novelty_reported = True

        # Update the PolycraftDetectionStats object
        stats.pddl_prob = novelty_likelihood
        stats.novelty_characterization['characterization'] = novelty_characterization
        self.current_detection = stats


    def _detect_unknown_objects(self, state: PolycraftState, world: Polycraft) -> Set[str]:
        """Returns a set of unknown object novelties identified in the given state

        Args:
            state (PolycraftState):Polycraft World state object
            world (Polycraft): Polycraft World object

        Returns:
            Set[str]: Set of strings with novelties
        """
        novelties = set()
        for block_type in state.get_type_to_cells():
            if block_type not in [tp for tp in self.meta_model.block_type_to_idx]:
                logger.info(f"Novel block type detected - {block_type}")
                self.meta_model.introduce_novel_block_type(block_type)
                novelties.add(f"{block_type}")
                self.objects_to_explore.append(block_type)
        for item_type in state.get_item_to_count():
            if item_type not in [tp for tp in self.meta_model.item_type_to_idx]:
                novelties.add(f"{item_type}")
                logger.info(f"Novel item type detected - {item_type}")
                self.meta_model.introduce_novel_inventory_item_type(item_type)
                self.objects_to_explore.append(item_type)
        for entity, entity_attr in state.entities.items():
            entity_type = entity_attr["type"]
            if entity_type not in [entity.value for entity in EntityType]:
                novelties.add(f"{entity_type}")
                logger.info(f"Novel entity type detected - {entity_type}")
                self.meta_model.introduce_novel_entity_type(entity_type)
                self.objects_to_explore.append(entity_type)
        if len(novelties) > 0 and not self.novelty_reported:
            self.current_detection.novelty_characterization['unknown_object'] = novelties
            
            world.poly_client.REPORT_NOVELTY(level="1", confidence=f"{100}", user_msg=str(novelties))
            self.novelty_reported = True
        return novelties

    def should_repair(self, state: PolycraftState, world: Polycraft) -> bool:
        """Choose if the agent should repair its metamodel based on the given observation

        Args:
            state (PolycraftState): World state object
            world (Polycraft): Polycraft World object

        Returns:
            bool: Whether or not to issue a repair
        """
        if self.novelty_explored:
            # There is a novelty we explored since last check, need to repair based on what we discoverd.
            self.novelty_explored = False  # done with this novelty
            return True
        elif self.current_stats.failed_actions > 0 and self.current_stats.failed_actions % self.exploration_rate == 0:
            return True
        self._detect_if_current_episode_is_novel(state, world, only_current_state=False)
        return self.current_detection.novelty_detected

    def repair_meta_model(self, state: PolycraftState, world: Polycraft):
        """ Call the repair object to repair the current metamodel """

        self._detect_unknown_objects(state, world)
        self.current_stats.repair_calls += 1
        try:
            # self.meta_model.set_active_task(CreatePogoTask())
            start_time = time.perf_counter()
            repair, consistency = self.meta_model_repair.repair(self.current_log,
                                                                delta_t=settings.POLYCRAFT_DELTA_T)
            repair_description = ["Repair %s, %.2f" % (fluent, repair[i])
                                  for i, fluent in enumerate(self.meta_model.repairable_constants)]
            logger.info(
                "Repair done! Consistency: %.2f, Repair:\n %s" % (consistency, repair_description))
            self.current_stats.repair_time = time.perf_counter() - start_time
        except:
            # TODO: fix this hack, catch correct exception
            import traceback
            traceback.print_exc()
            logger.info(f"No Valid repair found for inconcistency -- {repair}")


class PolycraftDoNothingAgent(PolycraftHydraAgent):
    """ An agent that does nothing """

    def __init__(self):
        super().__init__()

    def choose_action(self, world_state: PolycraftState):
        return PolyNoAction()


class PolycraftManualAgent(PolycraftHydraAgent):
    """ An agent that queries the user for actions USED FOR DEBUGGING """

    def __init__(self):
        super().__init__()
        self.command_seq = 0
        self.commands = []

    def choose_action(self, world_state: PolycraftState):
        if self.command_seq < len(self.commands):
            cmd = self.commands[self.command_seq]
            self.command_seq = self.command_seq + 1
        else:
            return PolyGiveUp()

        return cmd
