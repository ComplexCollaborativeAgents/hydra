import itertools

import numpy as np

import settings
from agent.planning.nyx.syntax.state import State

from agent.planning.nyx.abstract_heuristic import AbstractHeuristic, HeuristicSum, ZeroHeuristic
import agent.planning.nyx.syntax.constants as constants
import math
from utils.comparable_interval import ComparableInterval
from agent.planning.nyx.interval_heuristic import IntervalHeuristic

active_heuristic = None  # A mechanism for setting the heuristic externally

def get_heuristic_function(heuristic=constants.CUSTOM_HEURISTIC_ID, **kwargs):
    if heuristic == 0:
        return ZeroHeuristic()
    elif heuristic == 1:
        return CartpolePlusPlusHeuristic()
    elif heuristic == 2:
        return BadSBHeuristic()
    elif heuristic == 3:
        return active_heuristic
    elif heuristic == 4:
        return CartpoltHeuristic()
    elif heuristic == 5:
        return SBOneBirdHeuristic()
    elif heuristic == 6:
        return SBBlockedPigsHeuristic()
    elif heuristic == 7:
        return IntervalHeuristic(kwargs.get('groundedPPDL'))
    elif heuristic == 11:  # 5 + 6, get it?
        return HeuristicSum([SBOneBirdHeuristic(), SBBlockedPigsHeuristic()])
    else:
        return ZeroHeuristic()  # Implements the null heuristic


class CartpolePlusPlusHeuristic(AbstractHeuristic):
    """
    CARTPOLE++ HEURISTIC
    """

    def evaluate(self, node):
        if node.state_vars["['total_failure']"]:
            node.h = 999999
        else:
            node.h = math.sqrt(math.pow(node.state_vars["['pos_x']"], 2) +
                               math.pow(node.state_vars["['theta_x']"], 2) +
                               math.pow(node.state_vars["['theta_x_dot']"], 2) +
                               math.pow(node.state_vars["['pos_x_dot']"], 2) +
                               math.pow(node.state_vars["['theta_x_ddot']"], 2) +
                               math.pow(node.state_vars["['pos_x_ddot']"], 2) +
                               math.pow(node.state_vars["['pos_y']"], 2) +
                               math.pow(node.state_vars["['theta_y']"], 2) +
                               math.pow(node.state_vars["['theta_y_dot']"], 2) +
                               math.pow(node.state_vars["['pos_y_dot']"], 2) +
                               math.pow(node.state_vars["['theta_y_ddot']"], 2) +
                               math.pow(node.state_vars["['pos_y_ddot']"], 2)) * \
                     (node.state_vars["['time_limit']"] - node.state_vars["['elapsed_time']"])
        return node.h


class BadSBHeuristic(AbstractHeuristic):
    """
    Score heuristic for science birds - only useful for GBFS, and not very good even then.
    """

    def notify_initial_state(self, node):
        return self.evaluate(node)

    def evaluate(self, node):
        node.h = 50000 - node.state_vars["['points_score']"]
        return node.h

        # return 0

class CartpoltHeuristic(AbstractHeuristic):
    # CARTPOLE HEURISTIC
    def evaluate(self, node):
        node.h = math.sqrt(math.pow(node.state_vars["['x']"], 2) + math.pow(node.state_vars["['theta']"], 2) +
                           math.pow(node.state_vars["['theta_dot']"], 2) + math.pow(node.state_vars["['x_dot']"], 2) +
                           math.pow(node.state_vars["['theta_ddot']"], 2) + math.pow(node.state_vars["['x_ddot']"],
                                                                                     2)) * \
                 (node.state_vars["['time_limit']"] - node.state_vars["['elapsed_time']"])
        return node.h


class SBOneBirdHeuristic(AbstractHeuristic):
    """
    Heuristic for guiding a single bird in science birds.
    This heuristic has two parts:
     1) angles that miss the bounding box around all pigs and blocks are strongly discouraged.
     2) for angles that are not out-of-bounds, the number of planning steps to the closest pig (at current speed).
    """

    LARGE_VALUE = 999999

    def __init__(self):
        self.targets_x_keys = {}
        self.targets_x = {}
        self.pig_x_keys = {}
        self.pigs_x = {}
        self.targets_w_keys = {}
        self.targets_w = {}
        self.targets_y_keys = {}
        self.targets_y = {}
        self.pig_y_keys = {}
        self.pigs_y = {}
        self.targets_h_keys = {}
        self.targets_h = {}
        self.pig_radii_keys = {}
        self.pig_radii = {}

    def notify_initial_state(self, node):
        self._generate_keys(node)
        return self.evaluate(node)

    def evaluate(self, node):
        # Bird powers:
        # red + black: none\no need to deal with here
        # yellow: accelerates. Just don't check falling short, because can accelerate at peak and reach blocks.
        # white: modeled as "shoots straight down when tapped". Remove "passes over everything" check.
        # Blue: adding a 20% margin to the bounding box is probably pretty good. TODO: Some experimentation can narrow
        #                                                                           that to a more accurate number.

        active_bird_string = self._get_active_bird_string(node)
        if active_bird_string is None:
            # This heuristic doesn't know what do to without a birb
            node.h = self._backup_heuristic()
            return node.h
        bird_released = node.state_vars.get("['bird_released'" + active_bird_string)

        # near end of bounding box:

        for var_key, obj_id in self.targets_x_keys.items():
            self.targets_x[obj_id] = node.state_vars[var_key]
        for var_key, obj_id in self.pig_x_keys.items():
            self.pigs_x[obj_id] = node.state_vars[var_key]
        for var_key, obj_id in self.targets_w_keys.items():
            self.targets_w[obj_id] = node.state_vars[var_key]
        for var_key, obj_id in self.pig_y_keys.items():
            self.pigs_y[obj_id] = node.state_vars[var_key]
        for var_key, obj_id in self.pig_radii_keys.items():
            self.pig_radii[obj_id] = node.state_vars[var_key]

        self.targets_x.update(self.pigs_x)
        self.targets_w.update(self.pig_radii)
        targets_min_x = [self.targets_x[o_id] - self.targets_w[o_id] for o_id in self.targets_x.keys()]
        bbox_minx = min(targets_min_x)

        if bird_released:
            # The bird is in the air
            x_0 = node.state_vars["['x_bird'" + active_bird_string]
            y_0 = node.state_vars["['y_bird'" + active_bird_string]
            v_x = node.state_vars["['vx_bird'" + active_bird_string]
            v_y = node.state_vars["['vy_bird'" + active_bird_string]
            # Step 1: discourage tapping before we reach the bounding box
            if node.state_vars["['bird_tapped'" + active_bird_string]:
                if x_0 < bbox_minx:
                    node.h = SBOneBirdHeuristic.LARGE_VALUE
                    return node.h
                else:
                    if v_x == 0.0 or v_y == 0.0:
                        # Bird has been tapped and stopped - what to do here??
                        node.h = 0
                        return node.h
            # else: heuristic value = distance to nearest pig in planning steps.
            node.h = self.time_to_pig((x_0, y_0, v_x, v_y), self.pigs_x, self.pigs_y)
            return node.h
        else:
            # Find type of bird
            bird_type = node.state_vars["['bird_type'" + active_bird_string]
            BIRD_RED = 0
            BIRD_YELLOW = 1
            BIRD_BLACK = 2
            BIRD_WHITE = 3
            BIRD_BLUE = 4
            # Find bird trajectory:
            angle = node.state_vars["['angle']"]
            v_y_0 = node.state_vars["['v_bird'" + active_bird_string] * (
                    (4 * angle * (180 - angle)) / (40500 - (angle * (180 - angle))))
            v_x = node.state_vars["['v_bird'" + active_bird_string] * (1 - ((np.power((angle * 0.0174533), 2)) / 2))
            if v_x == 0:
                # In this situation, we can't reason about ballistics.
                # I have concluded that this situation doesn't arise, but left this just in case to prevent div by 0.
                node.h = self._backup_heuristic()
                return node.h

            y_0 = node.state_vars["['y_bird'" + active_bird_string]
            x_0 = node.state_vars["['x_bird'" + active_bird_string]
            gravity = node.state_vars["['gravity']"]

            if bird_type == BIRD_RED or bird_type == BIRD_BLACK or bird_type == BIRD_YELLOW or bird_type == BIRD_BLUE:
                # far end of bounding box
                targets_max_x = [self.targets_x[o_id] + self.targets_w[o_id] for o_id in self.targets_x.keys()]
                bbox_maxx = max(targets_max_x)

                for var_key, obj_id in self.targets_y_keys.items():
                    self.targets_y[obj_id] = node.state_vars[var_key]
                for var_key, obj_id in self.targets_h_keys.items():
                    self.targets_h[obj_id] = node.state_vars[var_key]
                self.targets_y.update(self.pigs_y)
                self.targets_h.update(self.pig_radii)

                targets_max_y = [self.targets_x[o_id] + self.targets_w[o_id] for o_id in self.targets_x.keys()]
                targets_min_y = [self.targets_x[o_id] - self.targets_w[o_id] for o_id in self.targets_x.keys()]
                bbox_maxy = max(targets_max_y)
                bbox_miny = min(targets_min_y)

                # Bounding box intersections:
                t_x_min_box = (bbox_minx - x_0) / v_x
                t_x_max_box = (bbox_maxx - x_0) / v_x
                y_top = y_0 + (0.5 * np.power(v_y_0, 2)) / gravity
                y_enter = y_0 + v_y_0 - 0.5 * gravity * np.power(t_x_min_box, 2)
                y_exit = y_0 + v_y_0 - 0.5 * gravity * np.power(t_x_max_box, 2)
                max_y = max(y_top, y_enter, y_exit)
                min_y = min(y_enter, y_exit)

                # if bird_type == BIRD_BLUE:
                min_y -= min_y * 0.1
                max_y += max_y * 0.1

                if min_y > bbox_maxy or max_y < bbox_miny:
                    # I expect that the entire shot passing under the lowest object is unlikely, but it's very easy to
                    #   rule out.
                    node.h = SBOneBirdHeuristic.LARGE_VALUE  # missed everything in the level entirely
                    return node.h

            if bird_type == BIRD_RED or bird_type == BIRD_BLACK or bird_type == BIRD_WHITE or bird_type == BIRD_BLUE:
                # Hitting the ground spot
                hit_ground_time = (- v_y_0 + np.sqrt(np.power(v_y_0, 2) + 2 * gravity * y_0)) / (2 * gravity)
                hit_ground_x = x_0 + v_x * hit_ground_time

                # if bird_type == BIRD_BLUE:
                hit_ground_x += hit_ground_x * 0.1

                # This prevents shots that hit the ground before reaching any targets, though some levels might need it?
                if hit_ground_x < bbox_minx:
                    node.h = SBOneBirdHeuristic.LARGE_VALUE
                    return node.h

            else:
                # Same calculation using only x values -> shoot at closest pig first
                node.h = self.time_to_pig_x(node, (x_0, y_0, v_x, v_y_0))
                pass
        return node.h

    def _get_active_bird_string(self, node):
        """
        Find active bird ID.
        :return The key string for the active bird, or None if not found.
        """
        active_bird_string = [key for key in node.state_vars.keys()
                              if (key.startswith("['bird_id',")
                                  and node.state_vars[key] == node.state_vars["['active_bird']"])]
        if len(active_bird_string) < 1:
            # can't find active bird
            return None
        active_bird_string = active_bird_string[0][10:]
        settings.active_bird_id_string = active_bird_string[-6:-2]
        return active_bird_string

    def _generate_keys(self, node):
        """
        Generates key strings into the state_vars dictionary, and stores them in lookup tables for speed.
        """
        # Important note: Python best practices explicitly recommend using str.startswith(x) instead of
        #   str[:len(x)] == x for readability reasons, but it's faster so I'm using it anyway.
        for key in node.state_vars.keys():
            if key[:9] == "['x_block":  # x_block
                self.targets_x_keys[key] = key[9:]
            if key[:7] == "['x_pig":  # x_pig
                self.pig_x_keys[key] = key[7:]
            if key[:13] == "['block_width":  # block_width
                self.targets_w_keys[key] = key[13:]
            if key[:12] == "['pig_radius":  # pig_radius
                self.pig_radii_keys[key] = key[12:]
            if key[:7] == "['y_pig":  # y_pig
                self.pig_y_keys[key] = key[7:]
            if key[:9] == "['y_block":  # y_block
                self.targets_y_keys[key] = key[9:]
            if key[:9] == "['block_h":  # block_height
                self.targets_h_keys[key] = key[14:]

    @staticmethod
    def _backup_heuristic():
        """
        If nothing else works - but what state are we in when we get here? No pigs? No birds? something else?
        """
        return 0  # 1 / (1 + node.state_vars["['points_score']"])

    @staticmethod
    def time_to_pig(bird_coords, pigs_x, pigs_y):
        """
        Returns the distance (in planning time steps) to the closest pig.
        """
        # Plan:
        # 1. get pigs
        # 2. find closest pig
        # 3. divide distance to closets pig by?
        #   (scalar product of velocity and unit direction to pig. is that too involved?)
        # BIRD_X = 0
        # BIRD_Y = 1
        # BIRD_VX = 2
        # BIRD_VY = 3
        targets_xy = [(pigs_x[o_id], pigs_y[o_id]) for o_id in pigs_x.keys()]
        dists = [(pig[0] - bird_coords[0]) ** 2 + (pig[1] - bird_coords[1]) ** 2 for pig in targets_xy]
        closest_ind = np.argmin(dists)
        vec_in_direction = (targets_xy[closest_ind][0] - bird_coords[0]) ** 2 / dists[closest_ind], \
                           (targets_xy[closest_ind][1] - bird_coords[1]) ** 2 / dists[closest_ind]
        speed_in_direction = bird_coords[2] * vec_in_direction[0] + bird_coords[3] * vec_in_direction[1]
        if speed_in_direction == 0.0:
            return np.inf
        value = max(0, math.sqrt(dists[closest_ind]) / (speed_in_direction * constants.DELTA_T))
        return int(value)

    @staticmethod
    def time_to_pig_x(node, bird_coords):
        target_x = min([node.state_vars[obj] for obj in node.state_vars.keys() if obj[:7] == "['x_pig"])
        return int(max(0, (target_x - bird_coords[0]) / (bird_coords[2] * constants.DELTA_T)))


class SBBlockedPigsHeuristic(SBOneBirdHeuristic):
    """
    Level-wide heuristic for science birds.
     1) How many pigs are still alive
     2) blocks in "front" of live pigs
     3) Optional: blocks under live pigs
    """
    #   2a) glass blocks are "worth" 1/10 of a pig
    #   2b) wood blocks are 1/4 of a pig
    #   2c) stone blocks are 1/2 of a pig
    # These are totally arbitrary. TODO get life from domain, add some other factor?
    H_MULTIPLIER = 10000  # This constant allows composition with the single bird heuristic.
    ICE_FACTOR = 0.01
    WOOD_FACTOR = 0.025
    STONE_FACTOR = 0.04
    UNKNOWN_B_LIFE = WOOD_FACTOR  # I believe this is the running assumption

    def __init__(self, blocks_under_pig=False):
        """
        :param blocks_under_pig: should blocks directly under pigs be considered 'suspicious' (=good targets to aim at).
        """
        super().__init__()
        self.pig_dead_keys = {}
        self.sus_blocks_keys = set()
        self.blocks_under_pigs = blocks_under_pig

    def notify_initial_state(self, node):
        # find keys for all pigs and blocks
        # find blocks "blocking" pigs and save in list
        self._generate_keys(node)
        return self.evaluate(node)

    def _generate_keys(self, node):
        super()._generate_keys(node)
        for key, item in self.pig_x_keys.items():
            self.pig_dead_keys[item] = "['pig_dead" + item
        self.sus_blocks_keys = self._check_sus_blocks(node, list(self.pig_x_keys.values()), list(self.targets_x_keys.values()))

    def _check_sus_blocks(self, node, pig_keys, block_keys):
        """
        Checks the given blocks to see if they're "in front of" the given pigs.
        """
        sus_blocks = set()
        for p_key in pig_keys:
            pig_x = node.state_vars["['x_pig" + p_key]
            pig_y = node.state_vars["['y_pig" + p_key]
            # pig_r = node.state_vars["['pig_radius" + p_key]
            for b_key in block_keys:
                block_x = node.state_vars["['x_block" + b_key]
                block_y = node.state_vars["['y_block" + b_key]
                block_w = node.state_vars["['block_width" + b_key]
                block_h = node.state_vars["['block_height" + b_key]

                if block_x < pig_x < block_x + 4 * block_w and \
                        block_y - 5 * block_h < pig_y < block_y + block_h:
                    sus_blocks.add(b_key)

                if self.blocks_under_pigs \
                        and block_x + block_w < pig_x < block_x + block_w \
                        and block_y < pig_y:
                    sus_blocks.add(b_key)
        return sus_blocks

    def evaluate(self, node):
        # find which pigs are still alive
        # go through list of suspicious blocks and sum ones that are still sus
        h_value = 0
        live_pigs = []
        for pig_id, pig_dead_key in self.pig_dead_keys.items():
            if not node.state_vars[pig_dead_key]:
                live_pigs.append(pig_id)
        h_value += SBBlockedPigsHeuristic.H_MULTIPLIER * len(live_pigs)
        self.sus_blocks_keys = self._check_sus_blocks(node, live_pigs, self.sus_blocks_keys)
        for b_key in self.sus_blocks_keys:
            if b_key[:7] == "', 'ice":
                h_value += SBBlockedPigsHeuristic.H_MULTIPLIER * SBBlockedPigsHeuristic.ICE_FACTOR
            elif b_key[:8] == "', 'wood":
                h_value += SBBlockedPigsHeuristic.H_MULTIPLIER * SBBlockedPigsHeuristic.WOOD_FACTOR
            elif b_key[:9] == "', 'stone":
                h_value += SBBlockedPigsHeuristic.H_MULTIPLIER * SBBlockedPigsHeuristic.STONE_FACTOR
            else:
                h_value += SBBlockedPigsHeuristic.H_MULTIPLIER * SBBlockedPigsHeuristic.UNKNOWN_B_LIFE
        node.h = h_value
        return node.h


class SBHelpfulAngleHeuristic(SBBlockedPigsHeuristic):
    """
    Calculates trajectories to pigs\blocks in front of or under pigs as a pre-processing step, and marks states
    on those trajectories as 'preferred' so that they are tried first.
    """
    def __init__(self, blocking_blocks=False):
        """
        param: blocking_blocks: Should the heuristic consider blocks in front of/under pigs good targets?
            (otherwise, only calculate trajectories to pigs)
        """
        SBBlockedPigsHeuristic.__init__(self, blocks_under_pig=blocking_blocks)
        self.x_0, self.y_0 = 0, 0
        self.g = 9.81
        self.deviation = ComparableInterval[-5, 5]  # TODO: find reasonable values
        self.trajectories = set()  # What are they? a set of lists of states? Just a set of states?

    evaluate = SBOneBirdHeuristic.evaluate

    def notify_initial_state(self, node: State):
        SBBlockedPigsHeuristic._generate_keys(self, node)
        first_bird = self._get_active_bird_string(node)  # For now, use this as x_0, y_0 (should really be the sling)
        self.x_0, self.y_0 = node.state_vars["['y_bird'" + first_bird], node.state_vars["['x_bird'" + first_bird]
        self.g = node.state_vars["['gravity']"]
        initial_velocity = node.state_vars["['v_bird'" + first_bird]
        self._calculate_useful_trajectories(initial_velocity, node)
        return self.evaluate(node)

    def _calculate_useful_trajectories(self, initial_velocity, node):
        live_pigs = []
        for pig_id, pig_dead_key in self.pig_dead_keys.items():
            if not node.state_vars[pig_dead_key]:
                live_pigs.append(pig_id)

        for p_key in live_pigs:
            pig_x = node.state_vars["['x_pig" + p_key]
            pig_y = node.state_vars["['y_pig" + p_key]
            # get coords
            min_angle, max_angle = self._get_single_trajectory(self.g, pig_x - self.x_0, pig_y - self.y_0,
                                                               initial_velocity)
            if min_angle is not None:
                self.trajectories.add(
                    (initial_velocity * math.cos(min_angle), initial_velocity * math.sin(min_angle)))
                self.trajectories.add(
                    (initial_velocity * math.cos(max_angle), initial_velocity * math.sin(max_angle)))

        if self.blocks_under_pigs:
            self.sus_blocks_keys = self._check_sus_blocks(node, live_pigs, self.sus_blocks_keys)
            explosive_blocks = [obj_id for _, obj_id in self.targets_x_keys.items() if node.state_vars["['block_explosive" + obj_id]]
            for block_key in itertools.chain(self.sus_blocks_keys, explosive_blocks):
                block_x = node.state_vars["['x_block" + block_key]
                block_y = node.state_vars["['y_block" + block_key]
                # get coords
                min_angle, max_angle = self._get_single_trajectory(self.g, block_x - self.x_0, block_y - self.y_0,
                                                                   initial_velocity)
                if min_angle is not None:
                    self.trajectories.add(
                        (initial_velocity * math.cos(min_angle), initial_velocity * math.sin(min_angle)))
                    self.trajectories.add(
                        (initial_velocity * math.cos(max_angle), initial_velocity * math.sin(max_angle)))

    @staticmethod
    def _get_single_trajectory(gravity, delta_x, delta_y, v_0):

        solution_existence_factor = v_0 ** 4 - gravity ** 2 * delta_x ** 2 - 2 * delta_y * gravity * v_0 ** 2

        # the target point cannot be reached
        if solution_existence_factor < 0:
            return None, None

        # solve cos theta from projectile equation
        cos_theta_1 = math.sqrt(
            (delta_x ** 2 * v_0 ** 2 - delta_x ** 2 * delta_y * gravity
             + delta_x ** 2 * math.sqrt(solution_existence_factor))
            / (2 * v_0 ** 2 * (delta_x ** 2 + delta_y ** 2)))
        cos_theta_2 = math.sqrt(
            (delta_x ** 2 * v_0 ** 2 - delta_x ** 2 * delta_y * gravity
             - delta_x ** 2 * math.sqrt(solution_existence_factor))
            / (2 * v_0 ** 2 * (delta_x ** 2 + delta_y ** 2)))

        theta_1 = math.acos(cos_theta_1)
        theta_2 = math.acos(cos_theta_2)

        return theta_1, theta_2

    def is_preferred(self, node):
        def trajectory_trace(x_0: float, y_0: float, v_x_0: float, v_y_0: float, g: float, x_t: ComparableInterval):
            x_0, y_0, v_x_0, v_y_0, g = round(x_0, 10), round(y_0, 10), round(v_x_0, 10), round(v_y_0, 10), round(g, 10)
            x_t = round(x_t)
            y_t = (y_0 + x_0 * (v_y_0 / v_x_0) - 0.5 * g * (x_0 ** 2) * (v_x_0 ** -2)) \
                  + ((v_y_0 / v_x_0) + g * (v_x_0 ** -2)) * x_t - 0.5 * g * (v_x_0 ** -2) * (x_t ** 2)
            return y_t

        active_bird_string = self._get_active_bird_string(node)
        if active_bird_string is not None:
            x_t = node.state_vars["['x_bird'" + active_bird_string] + self.deviation
            y_t = node.state_vars["['y_bird'" + active_bird_string]
            v_x_t = node.state_vars["['vx_bird'" + active_bird_string]
            for v_x_0, v_y_0 in self.trajectories:
                if v_x_0 in v_x_t + self.deviation and y_t in trajectory_trace(self.x_0, self.y_0, v_x_0, v_y_0, self.g, x_t):
                    return True
        return False
