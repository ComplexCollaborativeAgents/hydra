import json
import logging
import socket
import enum
import os
import numpy as np
import base64


"""
In addition to command-specific response variables that vary depending on the command sent, every response JSON contains the following keys:
goal: json dict that contains:
    goalType: "BLOCK_TO_LOCATION" for the HUGA Task, "POGOSTICK" for the POGO Task
    goalAchieved: True if the goal was achieved, False otherwise. Once the goal has been achieved, for the remainder of commands sent in that instance, this will continue to report "True" (boolean)
    Distribution: "Uninformed" if this trial requires “System Detection” of Novelty. "PreNovelty" or "Novelty" if this trial is “Given Detection”. (string)
command_result: json dict that contains
    command: command sent by Agent (string)
    argument: any command arguments sent by the Agent (string)
    result: "SUCCESS" if the command was executed properly, "FAIL" if a problem arose in command execution (string)
    message: a non-null string if the result = "FAIL" containing an error message (string)
    stepCost: step cost of executing a particular command (float)
    step: Step Number, 0-indexed count of commands sent by the AI Agent (integer)
    gameOver: True if the instance is over, False otherwise. (boolean)

NOTE: agent needs to handle gameOver == True as soon as it occurs!
"""

class TiltDir(enum.Enum):
    DOWN = "DOWN"
    FORWARD = "FORWARD"
    UP = "UP"

class MoveDir(enum.Enum):
    FORWARD = "w" # Forward 1 meter
    LEFT = "a" # Left 1 meter
    RIGHT = "d" # Right 1 meter
    BACK = "x"  # Backwards 1 meter
    FL = "q" # Forward Left diagonal sqrt 2 meter
    FR = "e" # Forward Right diagonal sqrt 2 meter
    BL = "z" # Backward Left diagonal sqrt 2 meter
    BR = "c" # Backward Right diagonal sqrt 2 meter


class FacingDir(enum.Enum):
    NORTH = "NORTH" # [0,0,-1]
    SOUTH = "SOUTH" # [0,0,1]
    WEST = "WEST" # [-1,0,0]
    EAST = "EAST" # [1,0,0]

    def get_angle_to(self, target_dir):
        ''' Return the angle needed to change this FacingDir object to the target dir'''
        if self==target_dir:
            return 0
        if (self==FacingDir.NORTH and target_dir==FacingDir.EAST) or \
                (self == FacingDir.EAST and target_dir == FacingDir.SOUTH) or \
                (self==FacingDir.SOUTH and target_dir==FacingDir.WEST) or \
                (self == FacingDir.WEST and target_dir == FacingDir.NORTH):
            return 90
        if (self==FacingDir.NORTH and target_dir==FacingDir.SOUTH) or \
                (self==FacingDir.SOUTH and target_dir==FacingDir.NORTH) or \
                (self == FacingDir.EAST and target_dir == FacingDir.WEST) or \
                (self == FacingDir.WEST and target_dir == FacingDir.EAST):
            return 180
        if (self == FacingDir.NORTH and target_dir == FacingDir.WEST) or \
                (self == FacingDir.WEST and target_dir == FacingDir.SOUTH) or \
                (self == FacingDir.SOUTH and target_dir == FacingDir.EAST) or \
                (self == FacingDir.EAST and target_dir == FacingDir.NORTH):
            return 270
        raise ValueError(f"No turning found for {self.value} and {target_dir.value}")

class PolycraftInterface:
    """ Low level interface to Polycraft Tournament """
    def __init__(self, settings_path: str, **kwargs):
        self.settings = {}

        with open(settings_path, "r") as settings_file:
            self.settings = json.load(settings_file)

        # Init logging
        self._extra_args = kwargs
        if "logger" in kwargs:
            self._logger = kwargs['logger']
        else:
            self._logger = logging.getLogger('Agent Client')

        # Overwrite if environment variable is set
        if os.getenv("PAL_AGENT_PORT", default=None) is not None:
            self.settings['port'] = int(os.getenv("PAL_AGENT_PORT"))
            self._logger.info("Using port from environment variable 'PAL_AGENT_PORT': {}".format(self.settings['port']))

        # Socket connection
        self.sock = None

        self.connect_to_polycraft(self.settings['host'], self.settings['port'])

    def connect_to_polycraft(self, host, port):
        """ Setup socket and connect to polycraft world """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.settings['host'], self.settings['port']))
            self._logger.info("Client connected to Polycraft Server ({}) on port {}".format(host, port))
        except socket.error as e:
            self._logger.exception("Failed to connect to Polycraft Host\n \
                                    Host: {}\n \
                                    Port: {}\n \
                                    Error Message: {}".format(host, port, e))
            raise e

    def disconnect_from_polycraft(self):
        """ Setup socket and connect to polycraft world """
        try:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
                self._logger.info("Client disconnected from Polycraft Server")
            else:
                self._logger.info("Client attempted to disconnect, was already disconnected from Polycraft Server")
        except (socket.error, socket.timeout) as e:
            self._logger.exception("Failed to disconnect from Polycraft Host")
            raise e

    def _send_cmd(self, cmd: str):
        """Low level - send command to Polycraft Instance"""
        try:
            if "CHECK_COST" not in cmd and "SENSE_ALL" not in cmd: # This commands are so often...
                self._logger.debug("Sending command: {}".format(cmd))
            self.sock.send(str.encode(cmd + '\n'))
        except BrokenPipeError as err:
            self.disconnect_from_polycraft()
            raise err

    def _recv_response(self, cmd:str = None) -> dict:
        """ Low level - receive data from the Polycraft instance. If cmd parameter is specified, it will send CHECK COST and recieve messages until the correct message is recieved """
        # for _ in range(self.settings['message_buffer_max']):
        data = b''

        while True:
            try:
                part = self.sock.recv(self.settings['requestbufbytes'])
                data += part
                if len(part) < self.settings['requestbufbytes']:
                    # either got nothing or reached end of data
                    if cmd is not None and data is not None and data != b'':
                        data_dict = json.loads(data)
                        recv_cmd = data_dict['command_result']['command']
                        if data_dict and recv_cmd == cmd:
                            self._logger.debug("Recieved full message for {}".format(recv_cmd))
                            break
                        else:   # Did not recieve the command we wanted, try again
                            self._logger.debug("Recieved dirty response from buffer: {}".format(recv_cmd))
                            data = b''
                            self._send_cmd("CHECK_COST")    # Use no-op to pull data from server buffer
                    else:
                        break
                    # break
            except KeyboardInterrupt as err:
                self.disconnect_from_polycraft()
                raise err

        if data == b'':
            self._logger.error("Got nothing from the socket buffer!")
            raise BrokenPipeError("Got nothing from the polycraft interface socket buffer!")

        data_dict = json.loads(data)
        self._logger.debug("Recieved: {}".format(data_dict))
        
        return data_dict

    def RESET(self, level_path: str) -> dict:
        """ Reset the simulation and load a level (.json) located at level_path (NEEDS FULL PATH) """
        self._send_cmd("RESET domain {}".format(level_path))

        response = self._recv_response("RESET")
            
        if response and response['command_result']['result'] == 'FAIL':
            self._logger.error("Could not load level {} (message: {})".format(level_path, response['command_result']['message'])) 

        return response

    def START(self) -> dict:
        """ Send the command to start the simulation. Can start sending other commands afterwards """
        cmd = "START"
        self._send_cmd(cmd)
        # For some reason, START sends 2 identical responses.  Issue when sending commands too early before Polycraft instance spins up fully.
        return self._recv_response(cmd)

    def CHECK_COST(self) -> dict:
        """ Returns the step cost incurred since the last RESET command. """
        cmd = "CHECK_COST"
        self._send_cmd(cmd)
        return self._recv_response(cmd)

    def GIVE_UP(self) -> dict:
        """ Give up on the task - agent still needs to send an additonal command following the GIVE_UP to recieve the 'gameOver = True'"""
        cmd = "GIVE_UP"
        self._send_cmd(cmd)
        return self._recv_response(cmd)

    def REPORT_NOVELTY(self, level: str = "", confidence: str = "", user_msg: str = "") -> dict:
        """
        Report that the agent has detected novelty.  Additional parameters may be added to further characterize the novelty.
        level: what level of novelty it might be
        confidence: confidence interval from 0.0 to 100.0
        user_msg: A user defined message
        """
        cmd = "REPORT_NOVELTY"
        if level != "":
            cmd += " -l {}".format(level)
        if confidence != "":
            cmd += " -c {}".format(confidence)
        if user_msg != "":
            cmd += " -m {}".format(user_msg)

        self._send_cmd(cmd)
        return self._recv_response("REPORT_NOVELTY")

    def SMOOTH_TILT(self, pitch: TiltDir) -> dict:
        """ Sets the actor's pitch (vertical) to -45/0/45 (down/forward/up) to look at objects at different elevations. (Use TiltDir enums respectively) """
        cmd = "SMOOTH_TILT {}".format(pitch.value)
        self._send_cmd(cmd)
        return self._recv_response("SMOOTH_TILT")

    def TURN(self, angle: int) -> dict:
        """ Turn the actor 'angle' degrees. Change in increments of 15 degrees. """
        if angle % 15 == 0:
            cmd = "TURN {}".format(angle)
            self._send_cmd(cmd)
            return self._recv_response("TURN")
        else:
            raise ValueError("Polycraft command TURN recieved invalid parameter: {}".format(angle))

    def MOVE(self, move_dir: MoveDir) -> dict:
        """Teleport actor to x/y/z coordinate, [distance] blocks south of the position.  Also resets orientation to North and pitch to 0."""
        cmd = "MOVE {}".format(move_dir.value)
        self._send_cmd(cmd)
        return self._recv_response("MOVE")

    def TP_TO_POS(self, cell: str, distance: int) -> dict:
        """Teleport actor to x/y/z coordinate, [distance] blocks south of the position.  Also resets orientation to North and pitch to 0."""
        cmd = "TP_TO {} {}".format(cell, distance)
        self._send_cmd(cmd)
        return self._recv_response("TP_TO")
    
    def TP_TO_ENTITY(self, entityID: str) -> dict:  # NOTE: Ask about distance parameter
        """Teleport actor to the entity with ID entityID"""
        cmd = "TP_TO {}".format(entityID)
        self._send_cmd(cmd)
        return self._recv_response("TP_TO")

    def BREAK_BLOCK(self) -> dict:
        """Break the block directly in front of the actor"""
        cmd = "BREAK_BLOCK"
        self._send_cmd(cmd)
        return self._recv_response(cmd)

    def INTERACT(self, entityID: str) -> dict:
        """ Similarly to SENSE_RECIPES, this command returns the list of available trades with a particular entity """
        cmd = "INTERACT {}".format(entityID)
        self._send_cmd(cmd)
        return self._recv_response("INTERACT")

    def SENSE_ALL(self) -> dict:
        """ Senses the actor's current inventory, all available blocks, recipes and entities that are in the same room as the actor """
        cmd = "SENSE_ALL NONAV"
        self._send_cmd(cmd)
        return self._recv_response("SENSE_ALL")

    def SENSE_RECIPES(self) -> dict:
        """ Senses all recipes available within the current level. """
        cmd = "SENSE_RECIPES"
        self._send_cmd(cmd)
        return self._recv_response("SENSE_RECIPES")

    def SELECT_ITEM(self, item_name: str = '') -> dict:
        """ Select an item by name within the actor's inventory to be the item that the actor is currently holding (active item).  Pass no item name to deselect the current selected item. """
        cmd = "SELECT_ITEM {}".format(item_name)
        self._send_cmd(cmd)
        return self._recv_response("SELECT_ITEM")
    
    def USE_ITEM(self, item_name: str = '') -> dict:
        """ Perform the use action (use key on safe, open door) with the item that is currently selected.  Alternatively, pass the item in to use that item. """
        cmd = "USE {}".format(item_name)
        self._send_cmd(cmd)
        return self._recv_response("USE")

    def PLACE(self, item_name: str) -> dict:
        """ Place a block or item from the actor's inventory in the space adjacent to the block in front of the player.  This command may fail if there is no block available to place the item upon. """
        cmd = "PLACE {}".format(item_name)
        self._send_cmd(cmd)
        return self._recv_response("PLACE")

    def COLLECT(self) -> dict:
        """ Collect item from block in front of actor - use for collecting rubber from a tree tap. """
        cmd = "COLLECT"
        self._send_cmd(cmd)
        return self._recv_response(cmd)

    def PLACE_CRAFTING_TABLE(self) -> dict:
        """ Macro for PLACE - places a crafting table in front of the actor """
        cmd = "PLACE_CRAFTING_TABLE"
        self._send_cmd(cmd)
        return self._recv_response(cmd)

    def PLACE_TREE_TAP(self) -> dict:
        """ Macro for PLACE - places a tree tap in front of the actor """
        self._send_cmd("PLACE_TREE_TAP")
        return self._recv_response()

    def DELETE(self, item_name: str) -> dict:
        """ Deletes the item in the player's inventory to prevent a fail state where the player is unable to pick up items due to having a full inventory """
        cmd = "DELETE {}".format(item_name)
        self._send_cmd(cmd)
        return self._recv_response("DELETE")

    def TRADE(self, entity_id: str, items: list) -> dict:
        """
        Perform a trade action with an adjacent entity. Accepts up to 5 items, and can result in up to 5 items.
        "items" is a list of tuples with format ("item_name", quantity) -> (str, int)
        """

        if len(items) == 0 or len(items) > 5:
            raise ValueError("Polycraft command TRADE recieved an invalid trade (number of items)")

        cmd = "TRADE {} ".format(entity_id)
        cmd += " ".join(["{} {}".format(item['Item'], item['stackSize']) for item in items])
        self._send_cmd(cmd)
        return self._recv_response("TRADE")

    def CRAFT(self, recipe: list) -> dict:
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

        cmd = "CRAFT 1 "
        recipe_size = len(recipe)
        if recipe_size == 4 or recipe_size == 9:
            matrix = " ".join(recipe)
            cmd += matrix
        else:
            raise ValueError("HYDRA: CRAFT command sent with invalid recipe parameters.")

        self._send_cmd(cmd)
        return self._recv_response("CRAFT")

    def SENSE_SCREEN(self) -> np.ndarray:
        """
        Get an image of what the player character is able to see.  Returns a flattened integer array, with each integer representing the RGB value of the pixel.
        """
        cmd = "SENSE_SCREEN"

        self._send_cmd(cmd)
        raw_data = self._recv_response(cmd)

        # Initially comes across as an integer array encoded in base64
        png_data_b64 = json.loads(raw_data)['screen']['data']
        data = base64.b64decode(png_data_b64)

        # Return as numpy array for convenience
        return np.frombuffer(data, np.uint8)