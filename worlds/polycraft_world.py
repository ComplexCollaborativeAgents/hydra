import json
import os
import pickle
import subprocess
import time
import logging
from os import path

import settings
import worlds.polycraft_interface.client.polycraft_interface.PolycraftInterface as poly
from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from utils.host import Host
from agent.planning.pddlplus_parser import PddlPlusProblem
from utils.state import State, Action, World


logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")


class PolycraftState(State):
    """Current State of Science Birds"""
    def __init__(self, game_state: dict):
        super().__init__()

        self.id = 0
        self.game_state = game_state

    def summary(self):
        '''returns a summary of state'''
        raise NotImplementedError("Polycraft State summary not implmented")

    def serialize_current_state(self, level_filename: str):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename: str):
        return pickle.load(open(level_filename, 'rb'))


class PolycraftAction(Action):
    ''' Polycraft World Action'''


class PolyTP(PolycraftAction):
    """ Teleport to a position "dist" away from the xyz coordinates facing in direction d and with pitch p"""
    def __init__(self, x: int, y: int, z: int, d: int = 0, p: int = 0, dist: int = 0):
        self.x = x
        self.y = y
        self.z = z
        self.direction = d
        self.pitch = p
        self.dist = dist


class PolyEntityTP(PolycraftAction):
    """ Teleport to a position "dist" away from the entity facing in direction d and with pitch p"""
    def __init__(self, entity_id: str, dist: int = 0, d: int = 0, p: int = 0):
        self.entity_id = entity_id
        self.dist = dist


class PolyTurn(PolycraftAction):
    """ Turn the actor side to side in the y axis (vertical) in increments of 15 degrees """
    def __init__(self, direction: int):
        self.direction = direction


class PolyTilt(PolycraftAction):
    """ Tilt the actor's focus up/down in the x axis (horizontal) in increments of 15 degrees """
    def __init__(self, pitch: int):
        self.pitch = pitch


class PolyBreak(PolycraftAction):
    """ Break the block directly in front of the actor """


class Interact(PolycraftAction):
    """ Similarly to SENSE_RECIPES, this command returns the list of available trades with a particular entity (must be adjacent) """
    def __init__(self, entity_id: str):
        self.entity_id = entity_id


class Sense(PolycraftAction):
    """ Senses the actor's current inventory, all available blocks, recipes and entities that are in the same room as the actor """


class SelectItem(PolycraftAction):
    """ Select an item by name within the actor's inventory to be the item that the actor is currently holding (active item).  Pass no item name to deselect the current selected item. """
    def __init__(self, item_name: str):
        self.item_name = item_name


class UseItem(PolycraftAction):
    """ Perform the use action (use key on safe, open door) with the item that is currently selected.  Alternatively, pass the item in to use that item. """
    def __init__(self, item_name: str = ""):
        self.item_name = item_name


class PlaceItem(PolycraftAction):
    """ Place a block or item from the actor's inventory in the space adjacent to the block in front of the player.  This command may fail if there is no block available to place the item upon. """
    def __init__(self, item_name: str):
        self.item_name = item_name


class CollectSap(PolycraftAction):
    """ Collect item from block in front of actor - use for collecting rubber from a tree tap. """


class DeleteItem(PolycraftAction):
    """ Deletes the item in the player's inventory to prevent a fail state where the player is unable to pick up items due to having a full inventory """
    def __init__(self, item_name: str):
        self.item_name = item_name


class TradeItems(PolycraftAction):
    """
    Perform a trade action with an adjacent entity. Accepts up to 5 items, and can result in up to 5 items.
    "items" is a list of tuples with format ("item_name", quantity) -> (str, int)
    """
    def __init__(self, items: list):
        self.items = items


class CraftItem(PolycraftAction):
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
        self.recipe = recipe


class Polycraft(World):
    """
    Polycraft interface supplied by UTD
    There is one Polycraft world per session
    We will make calls through the Polycraft runtime
    """

    def __init__(self, launch: bool = False, config: str = 'test_config.json', host=None):
        self.id = 2229

        self.history = []

        self.trajectory_planner = SimpleTrajectoryPlanner() # This is static to allow others to reason about it

        self.Poly_server_process = None     # Subprocess running the polycraft instance
        self.poly_client = None     # polycraft client interface (see polycraft_interface.py)
        
        if launch:
            self.launch_polycraft(config)
            time.sleep(5)
        self.create_interface(host=host)

    def kill(self):
        if self.Poly_server_process:
            logger.info("Killing process groups: {}".format(self.Poly_server_process.pid))
            try:
                os.killpg(self.Poly_server_process.pid, 9)
            except:
                logger.info("Error during process termination")

    def launch_polycraft(self, config: str = 'test_config.json'):
        """
        NOTE: AS OF 8/5 test_config.json HAS NOT BEEN CREATED
        Start polycraft server using parameters from config file
        See https://github.com/StephenGss/PAL/tree/release_2.0 for full list of parameters
        """

        config_params = {}

        with open(config, 'r') as config_f:
            config_params = json.load(config_f)

        print('launching science birds')
        # Needs to be run in the "pal/PolycraftAIGym" directory

        # Config needs to specify at the least:
        # * Headless mode
        # * path to folder with level files to run (tournament)
        # * agent command
        # * agent directory (what directory to run the agent command from)

        params = []
        if config_params['headless']: # Boolean value for headless mode
            params.append(settings.POLYCRAFT_HEADLESS)
        if config_params['trial_path']: # Path to folder containing level files to run
            params.append("-g {}".format(config_params['trial_path']))
        if config_params['agent_command']: # command line for what agent to run
            params.append("-x {}".foramt(config_params['agent_command']))
        if config_params['agent_directory']: # what directory to run the agent command from
            params.append("-d {}".format(config_params['agent_directory']))

        polycraft_cmd = " ".join(params)

        cmd = 'cd {} && {} > game_playing_interface.log'.format(settings.POLYCRAFT_DIR,
                                                                polycraft_cmd)
        print('Launching Polycraft using: {}'.format(cmd))
        self.Poly_server_process = subprocess.Popen(cmd,
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                                    start_new_session=True)
        print('Launched Polycraft with pid: {}'.format(str(self.Poly_server_process.pid)))

    def load_hosts(self, server_host: Host, observer_host: Host):
        """ Holdover from ScienceBirds world - intention is to use Docker to run as if agent were being evaluated"""
        
        with open(str(path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface', 'client', 'server_client_config.json')), 'r') as config:
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

    def create_interface(self, host=None):
        """
        Create polycraft interface - connect to server (server should be started first)
        """
        settings_path = str(path.join(settings.ROOT_PATH, 'worlds', 'polycraft_interface', 'client', 'server_client_config.json'))
        
        self.poly_client = poly.PolycraftInterface(settings_path)
        
        raise NotImplementedError()

    def init_selected_level(self, s_level):
        """ 
        Initialize a specific level 
        NOTE: at every end level, the gameOver boolean in the returned dictionary turns to True - we need to handle advancing to next level on our side
        """
        self.current_level = s_level
        self.poly_client.RESET(self.current_level)

    def act(self, action: PolycraftAction) -> tuple:
        ''' returns the new current state and step cost / reward '''
        # Match action with low level command in polycraft_interface.py

        raise NotImplementedError()

        return None, None

    def get_current_state(self) -> PolycraftState:
        """
        Query polycraft instance using low level interface and return State
        """
        raise NotImplementedError()
