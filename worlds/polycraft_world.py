import json
import os
import sys
import queue
import pickle
import subprocess
import time
import copy
import logging
from os import path
import threading
import psutil

import settings
import worlds.polycraft_interface.client.polycraft_interface as poly
# from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from utils.host import Host
from agent.planning.pddlplus_parser import PddlPlusProblem
from utils.state import State, Action, World


logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")
logger.setLevel(logging.DEBUG)


class PolycraftAction(Action):
    ''' Polycraft World Action '''

    def __init__(self):
        self.success = None

    def is_success(self, result: dict):
        try:
            return result['command_result']['result'] == "SUCCESS"
        except KeyError as err:
            return False

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        raise NotImplementedError("Subclasses of PolycraftAction should implement this")


class PolyNoAction(PolycraftAction):
    """ A no action (do nothing) """
        
    def __str__(self):
        return "<PolyNoAction>"
    
    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.CHECK_COST()


class PolyTP(PolycraftAction):
    """ Teleport to a position "dist" away from the given cell (cell name is its coordinates)"""
    def __init__(self, cell:str, dist: int = 0):
        super().__init__()
        self.cell = cell
        self.dist = dist

    def __str__(self):
        return "<PolyTP pos=({}) dist={} success={}>".format(self.cell, self.dist, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.TP_TO_POS(self.cell, distance=self.dist)
        self.success = self.is_success(result)
        return result


class PolyEntityTP(PolycraftAction):
    """ Teleport to a position "dist" away from the entity facing in direction d and with pitch p"""
    def __init__(self, entity_id: str, dist: int = 0):
        super().__init__()
        self.entity_id = entity_id
        self.dist = dist

    def __str__(self):
        return "<PolyEntityTP entity={} dist={} success={}>".format(self.entity_id, self.dist, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result =  poly_client.TP_TO_ENTITY(self.entity_id)
        self.success = self.is_success(result)
        return result


class PolyTurn(PolycraftAction):
    """ Turn the actor side to side in the y axis (vertical) in increments of 15 degrees """
    def __init__(self, direction: int):
        super().__init__()
        self.direction = direction

    def __str__(self):
        return "<PolyTurn dir={} success={}>".format(self.direction, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.TURN(self.direction)
        self.success = self.is_success(result)
        return result


class PolyTilt(PolycraftAction):
    """ Tilt the actor's focus up/down in the x axis (horizontal) in increments of 15 degrees """
    def __init__(self, pitch: str):
        super().__init__()
        self.pitch = pitch

    def __str__(self):
        return "<PolyTilt pitch={} success={}>".format(self.pitch, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.SMOOTH_TILT(self.pitch)
        self.success = self.is_success(result)
        return result


class PolyBreak(PolycraftAction):
    """ Break the block directly in front of the actor """

    def __str__(self):
        return "<PolyBreak success={}>".format(self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.BREAK_BLOCK()
        self.success = self.is_success(result)
        return result


class PolyInteract(PolycraftAction):
    """ Similarly to SENSE_RECIPES, this command returns the list of available trades with a particular entity (must be adjacent) """
    def __init__(self, entity_id: str):
        super().__init__()
        self.entity_id = entity_id

    def __str__(self):
        return "<PolyInteract entity={} success={}>".format(self.entity_id, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.INTERACT(self.entity_id)
        self.success = self.is_success(result)
        return result


class PolySense(PolycraftAction):
    """ Senses the actor's current inventory, all available blocks, recipes and entities that are in the same room as the actor """

    def __str__(self):
        return "<PolySense success={}>".format(self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.SENSE_ALL()
        self.success = self.is_success(result)
        return result


class PolySelectItem(PolycraftAction):
    """ Select an item by name within the actor's inventory to be the item that the actor is currently holding (active item).  Pass no item name to deselect the current selected item. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolySelectItem item={} success={}>".format(self.item_name, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.SELECT_ITEM(item_name=self.item_name)
        self.success = self.is_success(result)
        return result


class PolyUseItem(PolycraftAction):
    """ Perform the use action (use key on safe, open door) with the item that is currently selected.  Alternatively, pass the item in to use that item. """
    def __init__(self, item_name: str = ""):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyUseItem item={} success={}>".format(self.item_name, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.USE_ITEM(item_name=self.item_name)
        self.success = self.is_success(result)
        return result


class PolyPlaceItem(PolycraftAction):
    """ Place a block or item from the actor's inventory in the space adjacent to the block in front of the player.  This command may fail if there is no block available to place the item upon. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyPlaceItem item={} success={}>".format(self.item_name, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.PLACE(self.item_name)
        self.success = self.is_success(result)
        return result


class PolyCollect(PolycraftAction):
    """ Collect item from block in front of actor - use for collecting rubber from a tree tap. """

    def __str__(self):
        return "<PolyCollect success={}>".format(self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.COLLECT()
        self.success = self.is_success(result)
        return result

class PolyGiveUp(PolycraftAction):
    ''' An action in which the agent gives up'''
    def __str__(self):
        return "<PolyGiveUp success={}>".format(self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.GIVE_UP()
        self.success = self.is_success(result)
        return result

class PolyDeleteItem(PolycraftAction):
    """ Deletes the item in the player's inventory to prevent a fail state where the player is unable to pick up items due to having a full inventory """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def __str__(self):
        return "<PolyDeleteItem item={} success={}>".format(self.item_name, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.DELETE(self.item_name)
        self.success = self.is_success(result)
        return result


class PolyTradeItems(PolycraftAction):
    """
    Perform a trade action with an adjacent entity. Accepts up to 5 items, and can result in up to 5 items.
    "items" is a list of tuples with format ("item_name", quantity) -> (str, int)
    """
    def __init__(self, entity_id: str, items: list):
        super().__init__()
        self.entity_id = entity_id
        self.items = items

    def __str__(self):
        return "<PolyTradeItems entity={} items={} success={}>".format(self.entity_id, self.items, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.TRADE(self.entity_id, self.items)
        self.success = self.is_success(result)
        return result


class PolyCraftItem(PolycraftAction):
    """
    Craft an item using resources from the actor's inventory.
    "recipe" is a collapsed list of strings that represents a 2x2 or 3x3 matrix.
    For a 2x2, recipe would be a list with length 4
    for a 3x3, recipe would be a list with length 9
    Example:
        2x2: ["minecraft:plank", "0", "minecraft:plank", "0"] -> creates sticks
        3x3: ["minecraft:plank", "minecraft:plank", "0", "minecraft:plank", "minecraft:plank", "0", "0", "0", "0"]
        NOTE: "0" stands for a null/empty space in the matrix
    """
    def __init__(self, recipe: list):
        super().__init__()
        self.recipe = recipe

    def __str__(self):
        return "<PolyCraftItem recipe={} success={}>".format(self.recipe, self.success)

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        result = poly_client.CRAFT(self.recipe)
        self.success = self.is_success(result)
        return result


class PolycraftState(State):
    """ Current State of Polycraft """
    def __init__(self, step_num: int, facing_block: str,  location: dict, game_map: dict,
                 entities: dict, inventory: dict, current_item: str,
                 recipes: list, trades: list, terminal: bool, step_cost: float):
        super().__init__()

        self.id = step_num
        self.facing_block = facing_block    # the block the actor is currently facing
        self.location = location    # Formatted as {"pos": [x,y,z], "facing": DIR, yaw: ANGLE, pitch: ANGLE }
        self.game_map = game_map    # Formatted as {"xyz_string": {"name": "block_name", isAccessible: bool}, ...}
        self.entities = entities
        self.inventory = inventory  # Formatted as {SLOT_NUM:{ATTRIBUTES}, "selectedItem":{ATTRIBUTES}}
        self.current_item = current_item
        self.recipes = recipes
        self.trades = trades
        self.terminal = terminal
        self.step_cost = step_cost

    @staticmethod
    def create_current_state(poly_client : poly.PolycraftInterface):
        ''' Create the current state by calling a SENSE_ALL command '''
        # Call API
        sensed = poly_client.SENSE_ALL()

        # Extract values from SENSE_ALL
        step_num = sensed['step']
        facing_block = sensed['blockInFront']
        inventory = sensed['inventory']
        currently_selected = sensed['inventory']['selectedItem']
        pos = sensed['player']
        entities = sensed['entities']
        game_map = sensed['map']
        terminal = sensed['gameOver']
        return PolycraftState(step_num, facing_block, pos, game_map, entities,
                              inventory, currently_selected, None, None, terminal, -1)

    def __str__(self):
        return "< Step: {} | Action Cost: {} | Location: {} | Inventory: {} >".format(self.id, self.step_cost, self.location, self.inventory)

    def get_block_at(self, x: int, y: int, z: int) -> tuple:
        """ Helper function to get info of a block at coordinates xyz from the game map (type, accessibility) """
        coord_str = "{},{},{}".format(x, y, z)
        if coord_str not in self.game_map:
            return None, None
        
        return self.game_map[coord_str]["name"], self.game_map["isAccesible"]

    def get_available_actions(self):
        actions = []

        # TP to position
        for coords, block in self.game_map.items():
            if block['isAccessible']:
                x, y, z = coords.split(',')
                actions.append(PolyTP(coords))

        # TP to entity
        for entity in self.entities:
            actions.append(PolyEntityTP(entity))

        # Turn in a direction
        for direction in range(15, 360, 15):
            actions.append(PolyTurn(direction))

        # Tilt up/down
        for tilt_dir in poly.TiltDir:
            actions.append(PolyTilt(tilt_dir))

        # Break a block
        actions.append(PolyBreak())

        # Select an item from inventory
        for item_name in self.inventory.keys():
            actions.append(PolySelectItem(item_name))

        # Use an item (use an item from inventory as well)
        actions.append(PolyUseItem())
        for item_name in self.inventory.keys():
            actions.append(PolyUseItem(item_name=item_name))

        # Place an item from inventory
        for item_name in self.inventory.keys():
            actions.append(PolyPlaceItem(item_name))
            
        # Collect an item
        actions.append(PolyCollect())

        # Delete an item from inventory
        for item_name in self.inventory.keys():
            actions.append(PolyDeleteItem(item_name))

        # Make a trade for an item with an entity NOTE: May need to be adjacent TODO: decide whether or not to enforce adjacency in valid action?
        for trader, trades in self.trades.items():
            for trade in trades:
                actions.append(PolyTradeItems(trader, trade['inputs']))

        # Craft an item NOTE: will need to be adjacent to crafting bench for 3x3 crafts TODO: decide whether or not to enforce adjaceny in valid action?
        for recipe in self.recipes:
            # Create recipe list, filling zeros for empty slots
            slot_to_item = dict()
            for recipe_item in recipe['inputs']:
                item_name = recipe_item['Item']
                assert(recipe_item['stackSize']==1) # Currently supporting only one item per slot recipes
                slot = recipe_item['slot']
                slot_to_item[slot]=item_name

            # Infer if this is a 3x3 or 2x2 recipe
            if max(slot_to_item.keys())>3: # then this is  3x3 recipe
                recipe_size = 9
            else: # then this is a 2x2 recipe
                recipe_size = 4
            recipe_items = []
            for i in range(recipe_size):
                if i in slot_to_item:
                    recipe_items.append(slot_to_item[i])
                else:
                    recipe_items.append("0")

            actions.append(PolyCraftItem(recipe=recipe_items))

        return actions

    def summary(self):
        '''returns a summary of state'''
        return str(self)

    def serialize_current_state(self, level_filename: str):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename: str):
        return pickle.load(open(level_filename, 'rb'))

    def is_terminal(self) -> bool:
        return self.terminal

    def diff(self, other_state):
        ''' Return the differences between the states '''
        diff_dict = {}
        if self.facing_block!=other_state.facing_block:
            diff_dict['facing_block'] = {'self':self.facing_block, 'other':other_state.facing_block}

        if self.location!=other_state.location:
            location_diff_dict = dict()
            for loc_attr in self.location:
                if self.location[loc_attr]!=other_state.location[loc_attr]:
                    location_diff_dict[loc_attr] = {'self':self.location[loc_attr], 'other':other_state.location[loc_attr]}
            diff_dict['location']=location_diff_dict

        if self.game_map!=other_state.game_map:
            game_map_diff_dict = dict()
            for cell in self.game_map:
                if self.game_map[cell]!=other_state.game_map[cell]:
                    game_map_diff_dict[cell] = {'self': self.game_map[cell], 'other':other_state.game_map[cell]}
            diff_dict['game_map']=game_map_diff_dict

        if self.inventory!=other_state.inventory:
            inventory_diff_dict = dict()
            for item, item_attr in self.inventory.items():
                if item not in other_state.inventory:
                    inventory_diff_dict[item]={'self': item_attr, 'other':None}
                else:
                    if item_attr!=other_state.inventory[item]:
                        inventory_diff_dict[item] = {'self': item_attr, 'other': other_state.inventory[item]}
            diff_dict['inventory']=inventory_diff_dict

        if self.entities!=other_state.entities:
            entities_diff_dict = dict()
            for entity_id, entity_attr in self.entities.items():
                if entity_id not in other_state.entities:
                    entities_diff_dict[entity_id]={'self': entity_attr, 'other':None}
                else:
                    if entity_attr!=other_state.entities[entity_id]:
                        entities_diff_dict[entity_id] = {'self':entity_attr, 'other':other_state.entities[entity_id]}
            diff_dict['entities'] = entities_diff_dict

        if self.current_item!=other_state.current_item:
            diff_dict['current_item'] = {'self':self.current_item, 'other':other_state.current_item}

        if self.step_cost!=other_state.step_cost:
            diff_dict['step_cost'] = {'self': self.step_cost, 'other': other_state.step_cost}

        return diff_dict


class Polycraft(World):
    """
    Polycraft interface supplied by UTD
    There is one Polycraft world per session
    We will make calls through the Polycraft runtime
    """

    def __init__(self, launch: bool = False, client_config: str = None):
        self.id = 2229
        self.detached_server_mode = True # This means we do not listen to the server's output. This mode is used when launch parameter is False
        self.history = []

        # self.trajectory_planner = SimpleTrajectoryPlanner() # This is static to allow others to reason about it

        self.poly_server_process = None     # Subprocess running the polycraft instance
        self.poly_client = None     # polycraft client interface (see polycraft_interface.py)
        self.poly_listener = None   # Listener thread to the polycraft application
        self.poly_output_queue = queue.Queue()   # Queue that collects output from polycraft application subprocess

        # State information
        self.current_recipes = []
        self.current_trades = dict() # Maps entity id to its trades
        self.last_cmd_success = True
        self.ready_for_cmds = False
        
        if launch:
            logger.info("Launching Polycraft instance")
            self.launch_polycraft()
            self.detached_server_mode = False

        # Path to polycraft client interface config file (host name/port, buffer size, etc.)
        if client_config is None:
            client_config = str(path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client', 'server_client_config.json'))

        self.create_interface(client_config)

    def _read_polycraft_output(self, pipe, queue):
        """ Worker function for a separate daemon thread to listen to polycraft output.  Can be used for debugging. """
        
        logger.info("Entered Polycraft listener thread...")

        listening = True
        while listening and not pipe.stdout.closed:
            try:
                line = pipe.stdout.readline()
                if len(line)>0:
                    logger.debug(line)
                queue.put(line)
                sys.stdout.flush()
                pipe.stdout.flush()
            except UnicodeDecodeError as e:
                logger.error(e)
                try:
                    line = pipe.stdout.read().decode("utf-8")
                    queue.put(line)
                    sys.stdout.flush()
                    pipe.stdout.flush()
                except Exception as encoding_err:
                    logger.error("Could not handle output encoding: {}".format(encoding_err))
                    sys.stdout.flush()
                    pipe.stdout.flush()
            except Exception as unknown:
                logger.error("Unknown exception: {}".format(unknown))
                sys.stdout.flush()
                pipe.stdout.flush()

    def _get_polycraft_output(self):
        """ Check output queue, return next line output.  Can be used for debugging. """
        next_line = ""

        try:
            next_line = str(self.poly_output_queue.get(False, timeout=0.025))
            logger.debug(next_line)   # Turn off logging for now, the output from polycraft is large, and consists mostly of response messaging
            sys.stdout.flush()
            sys.stderr.flush()
        except queue.Empty:
            pass

        return next_line

    def kill(self, exit_program=False):
        ''' Perform cleanup '''

        # Disconnect client from polycraft
        if self.poly_client is not None:
            self.poly_client.disconnect_from_polycraft()
            self.poly_client = None

        if self.poly_server_process is not None:
            logger.info("Killing process groups: {}".format(self.poly_server_process.pid))
            try:
                os.killpg(self.poly_server_process.pid, 9)
            except os.error as err:
                logger.error("Error during process termination: {}".format(err))

        # Clean up any remaining processes
        procs = list(psutil.Process(os.getpid()).children(recursive=True))
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(procs, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

        self.poly_listener = None
        self.poly_output_queue = None

        if exit_program:
            exit()

    def launch_polycraft(self):
        """
        Start polycraft server using parameters from config dictionary
        See https://github.com/StephenGss/PAL/tree/release_2.0 for full list of parameters
        """

        logger.info('Launching Polycraft Server')
        # Needs to be run in the "pal/PolycraftAIGym" directory

        # Config needs to specify at the least:
        # * Headless mode

        params = []

        params.append(settings.POLYCRAFT_HEADLESS)  # NOTE: Polycraft must run headless if used with a subprocess
        params.append(settings.POLYCRAFT_SERVER_CMD)

        polycraft_cmd = " ".join(params)

        cmd = '{}'.format(polycraft_cmd)
        logger.info('Launching Polycraft using: {}'.format(cmd))

        self.poly_server_process = subprocess.Popen(cmd,
                                                    cwd=settings.POLYCRAFT_DIR,
                                                    stdout=subprocess.PIPE, 
                                                    stderr=subprocess.STDOUT, 
                                                    shell=True,
                                                    bufsize=1,
                                                    universal_newlines=True,
                                                    start_new_session=True)
        
        logger.info("Starting Polycraft process listener thread...")

        self.poly_listener = threading.Thread(target=self._read_polycraft_output, 
                                              args=(self.poly_server_process, self.poly_output_queue))
        self.poly_listener.daemon = True
        self.poly_listener.start()

        if self.poly_server_process.poll() is None:
            logger.debug('Launched Polycraft with pid: {}'.format(str(self.poly_server_process.pid)))
        else:
            logger.error('Could not launch Polycraft server - check the "polycraft_interface.log" located in the bin/pal folder')
            self.kill(exit_program=True)

        logger.info("Waiting for polycraft process to fully start up before sending commands...")

        # Wait for polycraft application to fully start up before sending commands
        while True:
            try:
                if "Minecraft finished loading" in self._get_polycraft_output():
                    logger.info("Polycraft application ready...")
                    break
            except KeyboardInterrupt as err:
                self.kill(exit_program=True)

    def load_hosts(self, server_host: Host, observer_host: Host):
        """ Holdover from ScienceBirds world - intention is to use Docker to run as if agent were being evaluated"""
        
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client', 'server_client_config.json')), 'r') as config:
            sc_json_config = json.load(config)

        server = Host(sc_json_config[0]['host'], sc_json_config[0]['port'])
        if 'DOCKER' in os.environ:
            server.hostname = 'docker-host'
        if server_host:
            server.hostname = server_host.hostname
            server.port = server_host.port

        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_observer_client_config.json')), 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)

        observer = Host(observer_sc_json_config[0]['host'], observer_sc_json_config[0]['port'])
        if 'DOCKER' in os.environ:
            observer.hostname = 'docker-host'
        if observer_host:
            observer.hostname = observer_host.hostname
            observer.port = observer_host.port

    def create_interface(self, settings_path: str):
        """
        Create polycraft interface - connect to server (server should be started first)
        """
        logger.info("Creating polycraft interface")
        
        try:
            self.poly_client = poly.PolycraftInterface(settings_path, logger=logger)
        except (ConnectionRefusedError, BrokenPipeError) as err:
            logger.error("Failed to connect to Polycraft server - shutting down.")
            self.kill(exit_program=True)


    def init_selected_level(self, s_level: str):
        """
        Initialize a specific level (accepts a string path)
        NOTE: at every end level, the gameOver boolean in the returned dictionary turns to True - we need to handle advancing to next level on our side
        """
        self.current_level = s_level
        self.ready_for_cmds = False
        try:
            self.poly_client.RESET(self.current_level)
            
            # Wait for level to load fully (if not loaded fully, SENSE_ALL will return nothing and other undefined behavior) TODO: make consistent with RunTournament.py
            if self.detached_server_mode==False:
                while True:
                    if "[EXP] game initialization completed" in self._get_polycraft_output():
                        self.ready_for_cmds = True
                        break
            else: # Detached server, using a heuristic of waiting a bit for it to load TODO: Is this bad?
                time.sleep(5) # Assumes 5 sec. is enough to load a level.

        except (BrokenPipeError, KeyboardInterrupt) as err:
            logger.error("Polycraft server connection interrupted ({})".format(err))
            self.kill(exit_program=True)

    def act(self, action: PolycraftAction) -> tuple:
        ''' returns the state and step cost / reward '''
        # Match action with low level command in polycraft_interface.py
        results = dict()

        if isinstance(action, PolycraftAction):
            try:
                results = action.do(self.poly_client)   # Perform each polycraft action's unique do command which uses the Polycraft API
                self.last_cmd_success = results['command_result']['result'] == "SUCCESS"    # Update if last command was successful or not
                logger.debug(str(action))
            except (BrokenPipeError, KeyboardInterrupt) as err:
                logger.error("Polycraft server connection interrupted ({})".format(err))
                self.kill()                
                raise err
        else:
            raise ValueError("Invalid action requested: {}".format(str(type(action))))

        # NOTE: Pulling state every action - incurs extra step cost

        return self.get_current_state(), results['command_result']['stepCost']

    def get_current_state(self) -> PolycraftState:
        """
        Query polycraft instance using low level interface and return State
        """
        # Call SENSE ALL API
        try:
            current_state = PolycraftState.create_current_state(self.poly_client)
        except (BrokenPipeError, KeyboardInterrupt) as err:
            self.kill()
            logger.error("Polycraft server connection interrupted (broken pipe or keyboard interrupt")
            raise err

        # Populate current recipes and trades
        current_state.trades = copy.copy(self.current_trades)
        current_state.recipes = copy.copy(self.current_recipes)

        # Obtain current cost
        step_cost = self.get_level_total_step_cost()
        current_state.step_cost = step_cost

        return current_state

    def populate_current_recipes(self):
        """
        Query polycraft instance to obtain the relevant recipes
        """
        response = self.poly_client.SENSE_RECIPES()
        assert(response['command_result']['result']=='SUCCESS')
        self.current_recipes = response['recipes']

    def interact(self, entity_id):
        """
        Query polycraft instance to interact with another agent
        """
        response = self.poly_client.INTERACT(entity_id)
        self.current_trades[entity_id] = response['trades']['trades']
        assert(response['command_result']['result']=='SUCCESS')

    def move_to_entity(self, entity_id):
        ''' Move adjacent to a given entity '''
        PolyEntityTP(entity_id, dist=1).do(self.poly_client)


    def get_level_total_step_cost(self) -> float:
        try:
            cost_dict = self.poly_client.CHECK_COST()
        except KeyboardInterrupt as err:
            self.kill(exit_program=True)
        
        try:
            msg = cost_dict['command_result']['message']
        except KeyError as err:
            logger.error("CHECK_COST message returned malformed: {}".format(cost_dict))
            raise err

        # extract numbers from message (as of 8/25/21, message has format "Total Cost Incurred: <cost>")
        try:
            parts = msg.split(': ')
            cost = float(parts[-1]) # extract the total cost
        except ValueError as err:
            logger.error("Could not parse CHECK_COST message for total cost: {}".format(msg))
            raise err

        return cost
