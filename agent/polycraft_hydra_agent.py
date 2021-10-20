import random
import datetime

from agent.consistency.observation import HydraObservation
from utils.polycraft_utils import *
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



class FixedPlanPlanner(HydraPlanner):
    ''' Planner for the polycraft domain that follows the following fixed plan:
    1) Mine diamonds and craft 2 diamons blocks
    2) Mine logs and make 2 sticks.
    3) Mine platinum and trade to titanium blocks
    4) Obtain the pallets somehow
    '''

    def __init__(self, meta_model = PolycraftMetaModel(), planning_path = settings.POLYCRAFT_PLANNING_DOCKER_PATH):
        super().__init__(meta_model)

    def make_plan(self,state: PolycraftState):
        # Stopping condition
        if state.count_items_of_type(ItemType.WOODEN_POGO_STICK.value)>0:
            logger.info("Already have the pogo stick! why plan?")
            return [PolyNoAction()]

        # Else, need to craft the pogo stick. Step 1: get recipe for it
        recipe_index = state.get_recipe_indices_for(ItemType.WOODEN_POGO_STICK.value)
        pogo_recipe = state.recipes[recipe_index]
        ingredients = dict()
        for input in pogo_recipe['inputs']:
            item_type = input['Item']
            quantity = input['stackSize']
            if item_type not in ingredients:
                ingredients[item_type]=quantity
            else:
                ingredients[item_type] = ingredients[item_type] + quantity


        # Step 2: comput what we're missing
        missing_ingredients = dict()
        for ingredient_type, ingredient_quantity in ingredients.items():
            count = state.count_items_of_type(ingredient_type)
            missing = ingredient_quantity-count
            if missing>0:
                missing_ingredients[ingredient_type] = missing

        # If all is here - craft the pogo stick!
        if len(missing_ingredients)==0:
            return [PolyCraftItem.create_action(pogo_recipe)]

        if ItemType.DIAMOND_BLOCK.value in missing_ingredients:
            return self._plan_for_diamond_blocks(state, missing_ingredients)
        if ItemType.STICK.value in missing_ingredients:
            return self._plan_for_sticks(state, missing_ingredients)
        if ItemType.BLOCK_OF_TITANIUM.value in missing_ingredients:
            return self._plan_for_blocks_of_titanium(state, missing_ingredients)
        if ItemType.SACK_POLYISOPRENE_PELLETS.value in missing_ingredients:
            return self._plan_for_sack_polyisoprene_pellets(state, missing_ingredients)

        logger.info("Ingredients missing that we have no fixed plan for. Missing ingredients are:")
        for ingredient_type, ingredient_quantity in missing_ingredients.items():
            logger.info(f'\t {ingredient_type} : {ingredient_quantity}')
        return None

    def _plan_for_diamond_blocks(self, state:PolycraftState, missing_ingredients:dict):
        ''' Mine diamonds and craft them toa  diamong block '''
        plan = []
        # Select iron pickaxe
        selected_item = state.get_selected_item()
        if ItemType.IRON_PICKAXE.value!=selected_item:
            plan.append(PolySelectItem(ItemType.IRON_PICKAXE.value))

        # Mine diamonds per desired block
        missing_diamond_blocks = missing_ingredients[ItemType.DIAMOND_BLOCK.value]
        missing_diamonds = missing_diamond_blocks*9
        missing_diamonds = missing_diamonds - state.count_items_of_type(ItemType.DIAMOND.value)
        if missing_diamonds>0:
            plan.append(CollectAndMineItem(ItemType.DIAMOND.value, missing_diamonds, [BlockType.DIAMOND_ORE.value]))

        # Craft
        diamond_block_recipe_indices = state.get_recipe_indices_for(poly.ItemType.DIAMOND_BLOCK.value)
        assert (len(diamond_block_recipe_indices) == 1)
        recipe = state.recipes[diamond_block_recipe_indices[0]]
        for i in range(missing_diamond_blocks):
            plan.append(PolyCraftItem.create_action(recipe))
        return plan

    def _plan_for_sticks(self, state:PolycraftState, missing_ingredients:dict):
        plan = []

        # Mine logs per desired block
        missing_sticks = missing_ingredients[ItemType.STICK.value]
        missing_logs = missing_sticks * 2
        missing_logs = missing_logs - state.count_items_of_type(BlockType.LOG.value)
        if missing_logs > 0:
            plan.append(CollectAndMineItem(BlockType.LOG.value, missing_logs, [BlockType.LOG.value]))

        # Craft
        stick_recipe_indices = state.get_recipe_indices_for(poly.ItemType.STICK.value)
        assert (len(stick_recipe_indices) == 1)
        recipe = state.recipes[stick_recipe_indices[0]]
        for i in range(missing_sticks):
            plan.append(PolyCraftItem.create_action(recipe))
        return plan

    def _plan_for_blocks_of_titanium(self, state:PolycraftState, missing_ingredients:dict):
        return None


    def _plan_for_sack_polyisoprene_pellets(self, state:PolycraftState, missing_ingredients:dict):
        return None

class PolycraftHydraAgent(HydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self, planner: HydraPlanner = PolycraftPlanner(),
                 meta_model_repair: MetaModelRepair = None):
        super().__init__(planner, meta_model_repair)

        self.active_plan = None

    def start_level(self, env: Polycraft):
        ''' Initialize datastructures for a new level and perform exploratory actions to get familiar with the current level.
        These actions are needed before calling the planner
        '''

        # Explore the level
        env.populate_current_recipes()
        current_state = env.get_current_state()
        # Try to interact with all other agents
        for entity_id, entity_attr in current_state.entities.items():
            if entity_attr['type']=='EntityTrader':
                # Move to trader
                env.move_to_entity(entity_id)

                # Interact with it
                env.interact(entity_id)

        # Initialize the current observation object
        current_state = env.get_current_state()
        self.current_observation = PolycraftObservation() # Start a new observation object for this level
        self.current_observation.states.append(current_state)
        self.observations_list.append(self.current_observation)


    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        # If no active plan - need to create one (this should happen in the beginning of a level)
        if self.active_plan is None:
            self.active_plan = self.planner.make_plan(world_state)
        else: # Check if the active plan is working or if we need to replan
            assert(len(self.current_observation.actions)>0)
            last_action = self.current_observation.actions[-1]
            if last_action.success==False:
                logger.info("Last action failed, replanning...")
                self.active_plan = self.planner.make_plan(world_state)
            else:
                logger.info("Continue to perform the current plan")

        # If no plan found, choose default action
        if self.active_plan is None or len(self.active_plan)>0:
            return self._choose_default_action()

        # Perform the next action in the plan
        assert(len(self.active_plan)>0)
        action = self.active_plan.pop(0)
        return action

    def _choose_default_action(self, world_state: PolycraftState):
        ''' Choose a default action. Current policy: try to mine something if available.
         Otherwise, try to collect an item.
         Otherwise do a no-op. '''

        # Try to mine a block
        type_to_cells = world_state.get_type_to_cells()
        non_air_types = [block_type for block_type in type_to_cells.keys() if block_type!=BlockType.AIR.value]
        while len(non_air_types)>0:
            type_index = random.choice(range(len(non_air_types)))
            type_to_mine = non_air_types.pop(type_index)
            cells = world_state.get_cells_of_type(type_to_mine, only_accessible=True)
            if len(cells)>0:
                cell = random.choice(cells)
                return PolyBreakAndCollect(cell)

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
        self.current_observation.actions.append(action)
        next_state, step_cost =  env.act(action)  # Note this returns step cost for the action
        self.current_observation.states.append(next_state)
        self.current_observation.rewards.append(step_cost)
        return next_state, step_cost

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
        self.commands = []

    def choose_action(self, world_state: PolycraftState):
        if self.command_seq<len(self.commands):
            cmd = self.commands[self.command_seq]
            self.command_seq=self.command_seq+1
        else:
            return PolyGiveUp()

        return cmd