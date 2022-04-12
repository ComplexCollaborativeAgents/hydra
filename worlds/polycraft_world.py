import copy
import enum
import json
import logging
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from os import path

import numpy as np
import psutil
import settings
from agent.planning.pddl_plus import TimedAction
from utils.host import Host
from utils.state import Action, State, World

import worlds.polycraft_interface.client.polycraft_interface as poly

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")
logger.setLevel(logging.INFO)


# Useful constants

class ServerMode(enum.Enum):
    CLIENT = "Only control the client. Assumes server is up by a different process"
    SERVER = "Control the polycraft server and the agent client"
    TOURNAMENT = "Configured for the TA1 evaluation."


class BlockType(enum.Enum):
    AIR = "minecraft:air"
    BEDROCK = "minecraft:bedrock"
    BLOCK_OF_PLATINUM = "polycraft:block_of_platinum"
    CRAFTING_TABLE = "minecraft:crafting_table"
    DIAMOND_ORE = "minecraft:diamond_ore"
    LOG = "minecraft:log"
    PLASTIC_CHEST = "polycraft:plastic_chest"
    SAPLING = "minecraft:sapling"
    TREE_TAP = "polycraft:tree_tap"
    WOODER_DOOR = "minecraft:wooden_door"
    SAFE = "polycraft:safe"


class ItemType(enum.Enum):
    BLOCK_OF_PLATINUM = "polycraft:block_of_platinum"
    BLOCK_OF_TITANIUM = "polycraft:block_of_titanium"
    CRAFTING_TABLE = "minecraft:crafting_table"
    DIAMOND = "minecraft:diamond"
    DIAMOND_BLOCK = "minecraft:diamond_block"
    IRON_PICKAXE = "minecraft:iron_pickaxe"
    KEY = "polycraft:key"
    LOG = "minecraft:log"
    PLANKS = "minecraft:planks"
    SAPLING = "minecraft:sapling"
    STICK = "minecraft:stick"
    SACK_POLYISOPRENE_PELLETS = "polycraft:sack_polyisoprene_pellets"
    TREE_TAP = "polycraft:tree_tap"
    WOODEN_POGO_STICK = "polycraft:wooden_pogo_stick"


class EntityType(enum.Enum):
    POGOIST = "EntityPogoist"
    TRADER = "EntityTrader"
    ITEM = "EntityItem"


class PolycraftState(State):
    """ Current State of Polycraft """

    def __init__(self, step_num: int, facing_block: str, location: dict, game_map: dict,
                 entities: dict, inventory: dict, current_item: str,
                 recipes: list, trades: list, door_to_room_cells: dict, terminal: bool, step_cost: int = -1, viz: np.ndarray=None):
        super().__init__()

        self.step_num = step_num
        self.facing_block = facing_block  # the block the actor is currently facing
        self.location = location  # Formatted as {"pos": [x,y,z], "facing": DIR, yaw: ANGLE, pitch: ANGLE }
        self.game_map = game_map  # Formatted as {"xyz_string": {"name": "block_name", isAccessible: bool}, ...}
        self.entities = entities
        self.inventory = inventory  # Formatted as {SLOT_NUM:{ATTRIBUTES}, "selectedItem":{ATTRIBUTES}}
        self.current_item = current_item
        self.terminal = terminal
        self.step_cost = step_cost  # TODO Remove this
        self.door_to_room_cells = door_to_room_cells  # Maps a room id to the door through which to enter to it
        self.recipes = recipes
        self.trades = trades
        self.viz = viz # flattened array of integers representing the visualization of the player character at the given state

    def get_known_cells(self) -> dict:
        ''' Returns a game-map-like dictionary containing all the cells from all the rooms '''
        cell_to_attr = dict()
        for door, room_cells in self.door_to_room_cells.items():
            for cell, cell_attr in room_cells.items():
                cell_to_attr[cell] = cell_attr
        return cell_to_attr

    def get_cells_of_type(self, item_type: str, only_accessible=False):
        ''' returns a list of cells that are of the given type '''
        cells = []
        for cell, cell_attr in self.get_known_cells().items():
            if cell_attr["name"] == item_type:
                if only_accessible and cell_attr['isAccessible'] == False:
                    continue
                cells.append(cell)
        return cells

    def __str__(self):
        return "< Step: {} | Action Cost: {} | Location: {} | Inventory: {} >".format(self.step_num, self.step_cost,
                                                                                      self.location, self.inventory)

    def get_type_to_cells(self):
        ''' Returns a dictionary of cell type to the list of cells of that type '''
        type_to_cells = dict()
        for cell, cell_attr in self.get_known_cells().items():
            cell_type = cell_attr["name"]
            if cell_type not in type_to_cells:
                type_to_cells[cell_type] = list()
            type_to_cells[cell_type].append(cell)
        return type_to_cells

    def get_item_to_count(self):
        ''' Return a map of inventory item type to count, excluding the selected item '''
        item_to_count = dict()
        for entry, entry_attr in self.inventory.items():
            if entry == "selectedItem":  # Do not consider the selected item
                continue
            item_type = entry_attr["item"]
            quantity = entry_attr['count']
            if item_type not in item_to_count:
                item_to_count[item_type] = quantity
            else:
                item_to_count[item_type] = item_to_count[item_type] + quantity
        return item_to_count

    def get_inventory_entries_of_type(self, item_type: str):
        ''' Returns the inventory entries that contain an item of the given type '''
        entries = []
        for entry, entry_attr in self.inventory.items():
            if entry == "selectedItem":  # Do not consider the selected item
                continue
            if entry_attr["item"] == item_type:
                entries.append(entry)
        return entries

    def get_entities_of_type(self, entity_type: str):
        ''' Return all the entities of a given type '''
        entities_to_return = []
        for entity, entity_attr in self.entities.items():
            if entity_attr["type"] == entity_type:
                entities_to_return.append(entity)
        return entities_to_return

    def count_items_of_type(self, item_type: str):
        ''' Counts the number of items of a given type '''
        count = 0
        inventory_entries = self.get_inventory_entries_of_type(item_type)
        for entry in inventory_entries:
            entry_attr = self.inventory[entry]
            count = count + entry_attr["count"]
        return count

    def is_facing_type(self, item_type: str):
        ''' checks if Steve is facing a cell of the given type '''
        return self.facing_block["name"] == item_type

    def has_item(self, item_type: str, count: int = 1):
        ''' Checks if we have enough items of the given type '''
        return len(self.get_inventory_entries_of_type(item_type)) > count - 1

    def get_selected_item(self):
        ''' Returns the type of item currently selected '''
        if "selectedItem" not in self.inventory:
            return None
        return self.inventory["selectedItem"]["item"]

    def get_all_recipes_for(self, item_name) -> list():
        ''' Return all the recipe that output an item of the given type '''
        recipes = list()
        for i, recipe in enumerate(self.recipes):
            for recipe_output in recipe['outputs']:
                if recipe_output['Item'] == item_name:
                    recipes.append(recipe)
        return recipes

    def get_recipe_for(self, item_name) -> list():
        ''' Return a single recipe that output an item of the given type '''
        recipes = list()
        for i, recipe in enumerate(self.recipes):
            for recipe_output in recipe['outputs']:
                if recipe_output['Item'] == item_name:
                    return recipe

    def get_all_trades_for(self, input_items, output_item):
        ''' Return a list of (trader_id, trade) tuples for trades
        in which we obtain the output item using only items from the input items '''
        results = []
        for trader_entity_id, trades in self.trades.items():
            for trade in trades:
                outputs = trade['outputs']
                for output in outputs:
                    if output['Item'] == output_item:
                        has_inputs = True
                        for input in trade['inputs']:
                            if input['Item'] not in input_items:
                                has_inputs = False
                        if has_inputs:
                            results.append((trader_entity_id, trade))
                        break
        return results

    def get_trade_for(self, item_to_get, items_to_give):
        ''' Return a singe tuple (trader_id, trade) tuples for trades
        in which we obtain the output item using only items from the input items '''
        for trader_entity_id, trades in self.trades.items():
            for trade in trades:
                outputs = trade['outputs']
                for output in outputs:
                    if output['Item'] == item_to_get:
                        has_inputs = True
                        for input in trade['inputs']:
                            if input['Item'] not in items_to_give:
                                has_inputs = False
                        if has_inputs:
                            return (trader_entity_id, trade)
                        break
        logger.info(f"No trades found where input={items_to_give} and output={output}")
        return None

    def summary(self) -> str:
        '''returns a summary of state'''
        summary_lines = []
        # The inventory
        summary_lines.append("Inventory:")
        for item, count in self.get_item_to_count().items():
            summary_lines.append(f"\t {item} : {count}")
        summary_lines.append(f"Selected item: {self.get_selected_item()}")

        # The available block types
        summary_lines.append("Map contains cells of type:")
        for block_type, cells in self.get_type_to_cells().items():
            summary_lines.append(f"\t {block_type} : {len(cells)}")

        # The location of all entities
        summary_lines.append(f"Steve is located at {self.location}")
        summary_lines.append("Entities are located at:")
        for entity_id, entity_attr in self.entities.items():
            summary_lines.append(f"\t {entity_id}: type:{entity_attr['type']}, location {entity_attr['pos']}")

        return "\n".join(summary_lines)

    def serialize_current_state(self, level_filename: str):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename: str):
        return pickle.load(open(level_filename, 'rb'))

    def is_terminal(self) -> bool:
        return self.terminal

    def diff(self, other_state):
        ''' Return the differences between the states '''
        diff_dict = {}
        if self.facing_block != other_state.facing_block:
            diff_dict['facing_block'] = {'self': self.facing_block, 'other': other_state.facing_block}

        if self.location != other_state.location:
            location_diff_dict = dict()
            for loc_attr in self.location:
                if self.location[loc_attr] != other_state.location[loc_attr]:
                    location_diff_dict[loc_attr] = {'self': self.location[loc_attr],
                                                    'other': other_state.location[loc_attr]}
            diff_dict['location'] = location_diff_dict

        if self.game_map != other_state.game_map:
            game_map_diff_dict = dict()
            for cell in self.game_map:
                if self.game_map[cell] != other_state.game_map[cell]:
                    game_map_diff_dict[cell] = {'self': self.game_map[cell], 'other': other_state.game_map[cell]}
            diff_dict['game_map'] = game_map_diff_dict

        if self.inventory != other_state.inventory:
            inventory_diff_dict = dict()
            for item, item_attr in self.inventory.items():
                if item not in other_state.inventory:
                    inventory_diff_dict[item] = {'self': item_attr, 'other': None}
                else:
                    if item_attr != other_state.inventory[item]:
                        inventory_diff_dict[item] = {'self': item_attr, 'other': other_state.inventory[item]}
            diff_dict['inventory'] = inventory_diff_dict

        if self.entities != other_state.entities:
            entities_diff_dict = dict()
            for entity_id, entity_attr in self.entities.items():
                if entity_id not in other_state.entities:
                    entities_diff_dict[entity_id] = {'self': entity_attr, 'other': None}
                else:
                    if entity_attr != other_state.entities[entity_id]:
                        entities_diff_dict[entity_id] = {'self': entity_attr, 'other': other_state.entities[entity_id]}
            diff_dict['entities'] = entities_diff_dict

        if self.current_item != other_state.current_item:
            diff_dict['current_item'] = {'self': self.current_item, 'other': other_state.current_item}

        if self.step_cost != other_state.step_cost:
            diff_dict['step_cost'] = {'self': self.step_cost, 'other': other_state.step_cost}

        return diff_dict


class PolycraftAction(Action, TimedAction):
    ''' Polycraft World Action '''

    def __init__(self):
        TimedAction.__init__(self, type(self).__name__, 0.0)
        self.name = type(self).__name__
        self.success = None
        self.response = None  # The result returned by the server for doing this command. Initialized as None.

    def is_success(self, results: dict):
        # If results are not a nice dictionary
        if isinstance(results, dict) == False:
            return False
        try:
            return results['command_result']['result'] == "SUCCESS"
        except KeyError as err:
            return False

    def get_cost(self, results: dict):
        if isinstance(results, dict) == False:
            return 0  # TODO: Design choice: what is the cost of an action with a bad response
        try:
            return results['command_result']['stepCost']
        except KeyError as err:
            return 0  # TODO: Design choice: what is the cost of an action with a bad response

    def __str__(self):
        return f"<{self.name} success={self.success}>"

    def do(self, state: PolycraftState, env) -> dict:
        raise NotImplementedError("Subclasses of PolycraftAction should implement this")

    def can_do(self, state: PolycraftState, env) -> bool:
        ''' Checks if this action can be done in the given state at the given environment
            Useful for adding control measures to be considered during execution '''
        return True


class Polycraft(World):
    DUMMY_DOOR = "-1,-1,-1"  # This marks the "door cell" for the main room, for the self.door_to_cells dict.

    """
    Polycraft interface supplied by UTD
    There is one Polycraft world per session
    We will make calls through the Polycraft runtime
    """

    def __init__(self, client_config: str = None, polycraft_mode: ServerMode = ServerMode.SERVER):
        self.id = 2229
        self.world_mode = polycraft_mode
        self.poly_server_process = None  # Subprocess running the polycraft instance
        self.poly_client = None  # polycraft client interface (see polycraft_interface.py)
        self.poly_listener = None  # Listener thread to the polycraft application
        self.poly_output_queue = queue.Queue()  # Queue that collects output from polycraft application subprocess

        # State information
        self.init_state_information()

        if self.world_mode == ServerMode.SERVER:
            logger.info("Launching Polycraft instance")
            self.launch_polycraft()
        else:
            logger.info("Starting agent without launching Polycraft")

        # Path to polycraft client interface config file (host name/port, buffer size, etc.)
        if client_config is None:
            client_config = str(
                path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client', 'server_client_config.json'))

        self.create_interface(client_config)

    def init_state_information(self):
        ''' Initialize information about the current state in the current episode (level) '''

        # Represent the current knowledge of the world. TODO: Think about moving this to the agent.
        self.current_recipes = list()
        self.current_trades = dict()  # Maps entity id to its trades
        self.door_to_room_cells = dict()  # Maps door cell to dict of gamemap cell in the room of that door

        # Store information
        self.last_cmd_success = True
        self.ready_for_cmds = False

    def _read_polycraft_output(self, pipe, queue):
        """ Worker function for a separate daemon thread to listen to polycraft output.  Can be used for debugging. """

        logger.info("Entered Polycraft listener thread...")

        listening = True
        while listening and not pipe.stdout.closed:
            try:
                line = pipe.stdout.readline()
                if len(line) > 0:
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
            # logger.debug(
            #     next_line)  # Turn off logging for now, the output from polycraft is large, and consists mostly of response messaging
            # sys.stdout.flush()
            # sys.stderr.flush()
        except queue.Empty:
            pass

        return next_line

    def wait_for_server_output(self, output:str):
        """ Waits for a particular message to be recieved from the server before continuing. """
        while True:
            try:
                if output in self._get_polycraft_output():
                    break
            except KeyboardInterrupt as err:
                self.kill(exit_program=True)

    
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
            logger.error(
                'Could not launch Polycraft server - check the "polycraft_interface.log" located in the bin/pal folder')
            self.kill(exit_program=True)

        logger.info("Waiting for polycraft process to fully start up before sending commands...")

        # Wait for polycraft application to fully start up before sending commands
        self.wait_for_server_output("Minecraft finished loading")
        logger.info("Polycraft application ready...")

    def load_hosts(self, server_host: Host, observer_host: Host):
        """ Holdover from ScienceBirds world - intention is to use Docker to run as if agent were being evaluated"""

        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client',
                                'server_client_config.json')), 'r') as config:
            sc_json_config = json.load(config)

        server = Host(sc_json_config[0]['host'], sc_json_config[0]['port'])
        if 'DOCKER' in os.environ:
            server.hostname = 'docker-host'
        if server_host:
            server.hostname = server_host.hostname
            server.port = server_host.port

        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client',
                                'server_observer_client_config.json')), 'r') as observer_config:
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

            # LaunchTournament.py automatically calls START, do not do so if running without it!
            if self.world_mode != ServerMode.TOURNAMENT:
                self.poly_client.START()  # Send START command - will not perform any further actions unless done so

            # Wait for agent to fully join before sending commands
            logger.info("Waiting for agent to connect to polycraft server...")
            self.wait_for_server_output("joined the game")
            logger.info("Agent connected to polycraft server...")

            # self.poly_client.CHECK_COST()  # Give time for the polycraft instance to clear its buffer?


        except (ConnectionRefusedError, BrokenPipeError) as err:
            logger.error("Failed to connect to Polycraft server - shutting down.")
            self.kill(exit_program=True)

    def init_selected_level(self, s_level: str):
        """
        Initialize a specific level (accepts a string path)
        NOTE: at every end level, the gameOver boolean in the returned dictionary turns to True - we need to handle advancing to next level on our side
        """
        # Reset values from past level
        self.current_level = s_level
        self.ready_for_cmds = False
        self.current_trades.clear()
        self.current_recipes.clear()
        self.door_to_room_cells.clear()

        try:
            self.poly_client.RESET(self.current_level)

            # Wait for level to load fully (if not loaded fully, SENSE_ALL will return nothing and other undefined behavior) TODO: make consistent with RunTournament.py
            if self.world_mode == ServerMode.SERVER:
                logger.info("Telling Polycraft to load a new level: {}".format(self.current_level))
                self.wait_for_server_output("[EXP] game initialization completed")
                self.ready_for_cmds = True
            else:  # Detached server, using a heuristic of waiting a bit for it to load TODO: Is this bad?
                logger.info("Waiting for level to initialize...")
                time.sleep(5)  # Assumes 5 sec. is enough to load a level.

        except (BrokenPipeError, KeyboardInterrupt) as err:
            logger.error("Polycraft server connection interrupted ({})".format(err))
            self.kill(exit_program=True)

    def act(self, state: PolycraftState, action: PolycraftAction) -> tuple:
        ''' returns the state and step cost / reward '''
        # Match action with low level command in polycraft_interface.py
        results = None
        if isinstance(action, PolycraftAction):
            try:
                if action.can_do(state, self):
                    results = action.do(state,
                                        self)  # Perform each polycraft action's unique do command which uses the Polycraft API
                    action.response = results
                    action.success = action.is_success(results)
                    self.last_cmd_success = action.success  # Store if last command was successful or not
                else:
                    # Environment cannot perform this action at the given state
                    self.last_cmd_success = False
                    action.response = f"Agent cannot do action {action} in the current state"
                    action.success = False

                logger.debug(str(action))
            except (BrokenPipeError, KeyboardInterrupt) as err:
                logger.error("Polycraft server connection interrupted ({})".format(err))
                self.kill()
                raise err
        else:
            raise ValueError("Invalid action requested: {}".format(str(type(action))))

        # NOTE: Pulling state every action - incurs extra step cost

        return self.get_current_state(), action.get_cost(results)

    def get_current_state(self, get_viz:bool=False) -> PolycraftState:
        """
        Query polycraft instance using low level interface and return State
        """
        # Call SENSE ALL API
        try:
            sensed = self.poly_client.SENSE_ALL()

            # Update game map knowledge with the sensed knowledge (note: SENSE_ALL only returns the game map for the current room)
            # Note: this will only update cells already explored. The action ExploreDoor should be used to explore new doors
            sensed_game_map = sensed['map']
            for cell_id, cell_attr in sensed_game_map.items():
                for door_cell_id, room_game_map in self.door_to_room_cells.items():
                    if cell_id in room_game_map:
                        room_game_map[cell_id] = cell_attr

            id = sensed['step']
            facing_block = sensed['blockInFront']
            inventory = sensed['inventory']
            currently_selected = sensed['inventory']['selectedItem']
            pos = sensed['player']

            terminal = sensed['gameOver']
            if 'goal' in sensed:
                terminal = sensed['gameOver'] or sensed['goal']['goalAchieved']
            entities = sensed['entities']
            viz = None

            if get_viz:
                viz = self.poly_client.SENSE_SCREEN()

            state = PolycraftState(id, facing_block, pos, sensed_game_map, entities, inventory, currently_selected,
                                   copy.deepcopy(self.current_recipes),
                                   copy.deepcopy(self.current_trades),
                                   copy.deepcopy(self.door_to_room_cells),
                                   terminal=terminal,
                                   step_cost=self.get_level_total_step_cost(),
                                   viz=viz)


        except (BrokenPipeError, KeyboardInterrupt) as err:
            self.kill()
            logger.error("Polycraft server connection interrupted (broken pipe or keyboard interrupt")
            raise err

        return state

    def populate_current_recipes(self):
        """
        Query polycraft instance to obtain the relevant recipes
        """
        response = self.poly_client.SENSE_RECIPES()
        assert (response['command_result']['result'] == 'SUCCESS')
        self.current_recipes = response['recipes']

    def populate_door_to_room_cells(self):
        # Initialize door to cells dictionary, mapping door cell to a gamemap-like dictionary of the room that door is pointing to
        response = self.poly_client.SENSE_ALL()
        assert (response['command_result']['result'] == 'SUCCESS')
        game_map = response['map']
        self.door_to_room_cells[Polycraft.DUMMY_DOOR] = dict()
        for cell_id, cell_attr in game_map.items():
            if cell_attr["name"] == BlockType.WOODER_DOOR.value:
                self.door_to_room_cells[cell_id] = dict()
                self.door_to_room_cells[cell_id][cell_id] = cell_attr
            self.door_to_room_cells[Polycraft.DUMMY_DOOR][cell_id] = cell_attr

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
            cost = float(parts[-1])  # extract the total cost
        except ValueError as err:
            logger.error("Could not parse CHECK_COST message for total cost: {}".format(msg))
            raise err

        return cost
