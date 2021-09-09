import json
import os
import pickle
import subprocess
import time
import copy
import logging
from os import path

import settings
import worlds.polycraft_interface.client.polycraft_interface as poly
from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from utils.host import Host
from agent.planning.pddlplus_parser import PddlPlusProblem
from utils.state import State, Action, World


logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")
logger.setLevel(logging.INFO)


class PolycraftAction(Action):
    ''' Polycraft World Action '''

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        raise NotImplementedError("Subclasses of PolycraftAction should implement this")


class PolyNoAction(PolycraftAction):
    """ A no action (do nothing) """
    
    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.CHECK_COST()


class PolyTP(PolycraftAction):
    """ Teleport to a position "dist" away from the xyz coordinates"""
    def __init__(self, x: int, y: int, z: int, dist: int = 0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.dist = dist

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.TP_TO_POS(self.x, self.y, self.z, distance=self.dist)


class PolyEntityTP(PolycraftAction):
    """ Teleport to a position "dist" away from the entity facing in direction d and with pitch p"""
    def __init__(self, entity_id: str, dist: int = 0):
        super().__init__()
        self.entity_id = entity_id
        self.dist = dist

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.TP_TO_ENTITY(self.entity_id, distance=self.dist)


class PolyTurn(PolycraftAction):
    """ Turn the actor side to side in the y axis (vertical) in increments of 15 degrees """
    def __init__(self, direction: int):
        super().__init__()
        self.direction = direction

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.TURN(self.direction)


class PolyTilt(PolycraftAction):
    """ Tilt the actor's focus up/down in the x axis (horizontal) in increments of 15 degrees """
    def __init__(self, pitch: int):
        super().__init__()
        self.pitch = pitch

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.SMOOTH_TILT(self.pitch)


class PolyBreak(PolycraftAction):
    """ Break the block directly in front of the actor """

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.BREAK_BLOCK()


class PolyInteract(PolycraftAction):
    """ Similarly to SENSE_RECIPES, this command returns the list of available trades with a particular entity (must be adjacent) """
    def __init__(self, entity_id: str):
        super().__init__()
        self.entity_id = entity_id

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.INTERACT(self.entity_id)


class PolySense(PolycraftAction):
    """ Senses the actor's current inventory, all available blocks, recipes and entities that are in the same room as the actor """

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.SENSE_ALL()


class PolySelectItem(PolycraftAction):
    """ Select an item by name within the actor's inventory to be the item that the actor is currently holding (active item).  Pass no item name to deselect the current selected item. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.SELECT_ITEM(item_name=self.item_name)


class PolyUseItem(PolycraftAction):
    """ Perform the use action (use key on safe, open door) with the item that is currently selected.  Alternatively, pass the item in to use that item. """
    def __init__(self, item_name: str = ""):
        super().__init__()
        self.item_name = item_name

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.USE_ITEM(item_name=self.item_name)


class PolyPlaceItem(PolycraftAction):
    """ Place a block or item from the actor's inventory in the space adjacent to the block in front of the player.  This command may fail if there is no block available to place the item upon. """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.PLACE(self.item_name)


class PolyCollect(PolycraftAction):
    """ Collect item from block in front of actor - use for collecting rubber from a tree tap. """

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.COLLECT()


class PolyDeleteItem(PolycraftAction):
    """ Deletes the item in the player's inventory to prevent a fail state where the player is unable to pick up items due to having a full inventory """
    def __init__(self, item_name: str):
        super().__init__()
        self.item_name = item_name

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.DELETE(self.item_name)


class PolyTradeItems(PolycraftAction):
    """
    Perform a trade action with an adjacent entity. Accepts up to 5 items, and can result in up to 5 items.
    "items" is a list of tuples with format ("item_name", quantity) -> (str, int)
    """
    def __init__(self, entity_id: str, items: list):
        super().__init__()
        self.entity_id = entity_id
        self.items = items

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.TRADE(self.entity_id, self.items)


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

    def do(self, poly_client: poly.PolycraftInterface) -> dict:
        return poly_client.CRAFT(self.recipe)


class PolycraftState(State):
    """ Current State of Polycraft """
    def __init__(self, facing_block: str,  location: dict, game_map: dict,
                 entities: dict, inventory: dict, current_item: str,
                 recipes: list, trades: list, terminal: bool, step_cost: float):
        super().__init__()

        self.id = 0
        self.facing_block = facing_block    # the block the actor is currently facing
        self.location = location    # Formatted as {"pos": [x,y,z], "facing": DIR, yaw: ANGLE, pitch: ANGLE }
        self.game_map = game_map    # Formatted as {"xyz_string": {"name": "block_name", isAccessible: bool}, ...}
        self.entities = entities
        self.inventory = inventory  # Formatted as {}   # TODO: update this
        self.current_item = current_item
        self.recipes = recipes
        self.trades = trades
        self.terminal = terminal
        self.step_cost = step_cost

    def __str__(self):
        return "< Step: {} | Action Cost: {} | Inventory: {} >".format(self.id, self.step_cost, self.inventory)

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
                actions.append(PolyTP(int(x), int(y), int(z)))

        # TP to entity
        for entity in self.entities:
            actions.append(PolyEntityTP(entity))

        # Turn in a direction
        for direction in range(15, 360, 15):
            actions.append(PolyTurn(direction))

        # Tilt up/down
        for angle in range(180, 0, -45):
            actions.append(PolyTilt(angle))
            
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
        for trade in self.trades:
            actions.append(PolyTradeItems(trade['entity_id'], trade['input']))

        # Craft an item NOTE: will need to be adjacent to crafting bench for 3x3 crafts TODO: decide whether or not to enforce adjaceny in valid action?
        for recipe in self.recipes:
            actions.append(PolyCraftItem(recipe=recipe['input']))

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


class Polycraft(World):
    """
    Polycraft interface supplied by UTD
    There is one Polycraft world per session
    We will make calls through the Polycraft runtime
    """

    def __init__(self, server_config: dict, launch: bool = False, client_config: str = None):
        self.id = 2229

        self.history = []

        self.trajectory_planner = SimpleTrajectoryPlanner() # This is static to allow others to reason about it

        self.poly_server_process = None     # Subprocess running the polycraft instance
        self.poly_client = None     # polycraft client interface (see polycraft_interface.py)

        # State information
        self.current_recipes = []
        self.current_trades = []
        
        if launch:
            logger.info("Launching Polycraft instance")
            self.launch_polycraft(server_config=server_config)
            time.sleep(30) # Wait for server to start up

        # Path to polycraft client interface config file (host name/port, buffer size, etc.)
        if client_config is None:
            client_config = str(path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client', 'server_client_config.json'))

        self.create_interface(client_config)

    def kill(self):
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

    def launch_polycraft(self, server_config: dict):
        """
        Start polycraft server using parameters from config dictionary
        See https://github.com/StephenGss/PAL/tree/release_2.0 for full list of parameters
        """

        logger.info('Launching Polycraft Server')
        # Needs to be run in the "pal/PolycraftAIGym" directory

        # Config needs to specify at the least:
        # * Headless mode

        params = []

        params.append(settings.POLYCRAFT_HEADLESS)  # NOTE: Polycraft must run headless if going through 
        params.append(settings.POLYCRAFT_SERVER_CMD)

        polycraft_cmd = " ".join(params)

        cmd = '{} > polycraft_interface.log'.format(polycraft_cmd)
        logger.info('Launching Polycraft using: {}'.format(cmd))
        self.poly_server_process = subprocess.Popen(cmd,
                                                    cwd=settings.POLYCRAFT_DIR,
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                                    bufsize=1,
                                                    universal_newlines=True,
                                                    start_new_session=True)
        logger.debug('Launched Polycraft with pid: {}'.format(str(self.poly_server_process.pid)))

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
        except ConnectionRefusedError as err:
            logger.error("Failed to connect to Polycraft server - shutting down.")
            self.kill()

    def set_current_recipes(self, recipes: list):
        """ To be called some time after the initialization of each level.  Hydra Agent to explore and collect all recipes before actual operation. """
        # TODO: associate recipes with isLarge (3x3 and thus requires bench), input and output (should be [{isLarge, input, output}]) 
        
        self.current_recipes = recipes

    def set_current_trades(self, trades: list):
        """ To be called some time after the initialization of each level.  Hydra Agent to explore and collect all trades before actual operation. """
        # TODO: associate trades with entity id, input and output (should be [{entity id, input, output}, ...])
        
        self.current_trades = trades

    def init_selected_level(self, s_level: str):
        """
        Initialize a specific level (accepts a string path)
        NOTE: at every end level, the gameOver boolean in the returned dictionary turns to True - we need to handle advancing to next level on our side
        """
        self.current_level = s_level
        try:
            self.poly_client.RESET(self.current_level)
            time.sleep(5) # Wait for level to load fully (if not loaded fully, SENSE_ALL will return nothing and other undefined behavior) TODO: make consistent with RunTournament.py
        except (BrokenPipeError, KeyboardInterrupt) as err:
            self.kill()
            logger.error("Polycraft server connection interrupted (broken pipe or keyboard interrupt")
            raise err

    def act(self, action: PolycraftAction) -> tuple:
        ''' returns the state and step cost / reward '''
        # Match action with low level command in polycraft_interface.py
        results = dict()

        if isinstance(action, PolycraftAction):
            try:
                results = action.do(self.poly_client)   # Perform each polycraft action's unique do command which uses the Polycraft API
            except (BrokenPipeError, KeyboardInterrupt) as err:
                self.kill()
                logger.error("Polycraft server connection interrupted (broken pipe or keyboard interrupt")
                raise err
        else:
            raise ValueError("Invalid action requested: {}".format(str(type(action))))

        # NOTE: Pulling state every action - incurs extra step cost

        return self.get_current_state(), results['command_result']['stepCost']

    def get_current_state(self) -> PolycraftState:
        """
        Query polycraft instance using low level interface and return State
        """
        # Call API
        try:
            sensed = self.poly_client.SENSE_ALL()
        except (BrokenPipeError, KeyboardInterrupt) as err:
            self.kill()
            logger.error("Polycraft server connection interrupted (broken pipe or keyboard interrupt")
            raise err

        # Extract values from SENSE_ALL
        facing_block = sensed['blockInFront']
        inventory = sensed['inventory']
        currently_selected = sensed['inventory']['selectedItem']
        pos = sensed['player']
        entities = sensed['entities']
        game_map = sensed['map']
        terminal = sensed['gameOver']
        step_cost = sensed['command_result']['stepCost']

        return PolycraftState(facing_block, pos, game_map, entities, inventory, currently_selected, copy.copy(self.current_recipes), copy.copy(self.current_trades), terminal, step_cost)

    def get_level_total_step_cost(self) -> float:
        cost_dict = self.poly_client.CHECK_COST()
        
        msg = cost_dict['command_result']['message']

        # extract numbers from message (as of 8/25/21, message has format "Total Cost Incurred: <cost>")
        nums = [float(s) for s in msg.split(' ') if s.isdigit()]

        if len(nums) == 0:
            raise ValueError("API did not return a step cost number! (Check API)")

        return nums[0]
