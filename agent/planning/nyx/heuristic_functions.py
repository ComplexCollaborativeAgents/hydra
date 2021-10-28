import numpy as np

from agent.planning.nyx.abstract_heuristic import AbstractHeuristic
from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants
import math

active_heuristic = None  # A mechanism for setting the heuristic externally

def get_heuristic_function(heuristic = constants.CUSTOM_HEURISTIC_ID):
    if heuristic == 1:
        return CartpolePlusPlusHeuristic()
    elif heuristic == 2:
        return BadSBHeuristic()
    elif heuristic == 3:
        return active_heuristic
    elif heuristic == 4:
        return CartpoltHeuristic()
    elif heuristic == 5:
        return SBHeuristic()
    else:
        return AbstractHeuristic() # Implements the null heuristic

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
    def evaluate(self, node):
        node.h =  1 / (1 + node.state_vars["['points_score']"])
        return node.h

        # return 0

class CartpoltHeuristic(AbstractHeuristic):
        # CARTPOLE HEURISTIC
        def evaluate(self, node):
            node.h = math.sqrt(math.pow(node.state_vars["['x']"], 2) + math.pow(node.state_vars["['theta']"], 2) +
                               math.pow(node.state_vars["['theta_dot']"], 2) + math.pow(node.state_vars["['x_dot']"], 2) +
                               math.pow(node.state_vars["['theta_ddot']"], 2) + math.pow(node.state_vars["['x_ddot']"], 2)) * \
                     (node.state_vars["['time_limit']"] - node.state_vars["['elapsed_time']"])
            return node.h

class SBHeuristic(AbstractHeuristic):
    """
    Heuristic for science birds.
    """

    LARGE_VALUE = 999999

    def evaluate(self, node):
        # Bird powers:
        # red + black: none\no need to deal with here
        # yellow: accelerates. Just don't check falling short, because can accelerate at peak and reach blocks.
        # white: modeled as "shoots straight down when tapped". Remove "passes over everything" check.
        # Blue: adding a 20% margin to the bounding box is probably pretty good. Some experimentation can narrow that to a more accurate number.


        # Find active bird ID
        active_bird_string = [key for key in node.state_vars.keys()
                              if (key.startswith("['bird_id',")
                                  and node.state_vars[key] == node.state_vars["['active_bird']"])]
        if len(active_bird_string) < 1:
            # This heuristic doesn't know what do to without a birb
            node.h = self._backup_heuristic(node)
            return node.h
        active_bird_string = active_bird_string[0][10:]
        bird_released = node.state_vars.get("['bird_released'" + active_bird_string)

        # near end of bounding box:
        targets_x = {obj[9:]: node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['x_block")}
        targets_x.update(
            {obj[7:]: node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['x_pig")})
        targets_w = {obj[13:]: node.state_vars[obj] for obj in node.state_vars.keys() if
                     obj.startswith("['block_width")}
        pig_radii = {obj[12:]: node.state_vars[obj] for obj in node.state_vars.keys() if
                     obj.startswith("['pig_radius")}
        targets_w.update(pig_radii)
        targets_min_x = [targets_x[o_id] - targets_w[o_id] for o_id in targets_x.keys()]
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
                    node.h = SBHeuristic.LARGE_VALUE
                    return node.h
                else:
                    if v_x == 0.0 or v_y == 0.0:
                        # Bird has been tapped and stopped - what to do here??
                        node.h = 0
                        return node.h
            # else: heuristic value = distance to nearest pig in planning steps.
            node.h = self.time_to_pig(node, (x_0, y_0, v_x, v_y))
            return node.h
        else:
            # Find type of bird
            bird_type = [active_bird_string.startswith(", red") or active_bird_string.startswith(", black"),
                         active_bird_string.startswith(", yellow"), active_bird_string.startswith(", white"),
                         active_bird_string.startswith(", blue")]
            BIRD_RED_BLACK = 0
            BIRD_YELLOW = 1
            BIRD_WHITE = 2
            BIRD_BLUE = 3
            # Find bird trajectory:
            angle = node.state_vars["['angle']"]
            v_y_0 = node.state_vars["['v_bird'" + active_bird_string] * (
                    (4 * angle * (180 - angle)) / (40500 - (angle * (180 - angle))))
            v_x = node.state_vars["['v_bird'" + active_bird_string] * (1 - ((np.power((angle * 0.0174533), 2)) / 2))
            if v_x == 0:
                # In this situation, we can't reason about ballistics.
                # I have concluded that this situation doesn't arise, but left this just in case to prevent div by 0.
                node.h = self._backup_heuristic(node)
                return node.h

            y_0 = node.state_vars["['y_bird'" + active_bird_string]
            x_0 = node.state_vars["['x_bird'" + active_bird_string]
            gravity = node.state_vars["['gravity']"]

            if bird_type[BIRD_RED_BLACK] or bird_type[BIRD_YELLOW] or bird_type[BIRD_BLUE]:
                # far end of bounding box
                targets_max_x = [targets_x[o_id] + targets_w[o_id] for o_id in targets_x.keys()]
                bbox_maxx = max(targets_max_x)
                targets_y = {obj[9:]:node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['y_block")}
                targets_y.update({obj[7:]:node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['y_pig")})
                targets_h = {obj[14:]:node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['block_height")}
                targets_h.update(pig_radii)
                targets_max_y = [targets_x[o_id] + targets_w[o_id] for o_id in targets_x.keys()]
                targets_min_y = [targets_x[o_id] - targets_w[o_id] for o_id in targets_x.keys()]
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

                if bird_type[BIRD_BLUE]:
                    min_y -= min_y * 0.2
                    max_y += max_y * 0.2

                if min_y > bbox_maxy or max_y < bbox_miny:
                    # I expect that the entire shot passing under the lowest object is unlikely, but it's very easy to rule out.
                    node.h = SBHeuristic.LARGE_VALUE  # missed everything in the level entirely
                    return node.h

            if bird_type[BIRD_RED_BLACK] or bird_type[BIRD_WHITE] or bird_type[BIRD_BLUE]:
                # Hitting the ground spot
                hit_ground_time = (- v_y_0 + np.sqrt(np.power(v_y_0, 2) + 2 * gravity * y_0)) / (2 * gravity)
                hit_ground_x = x_0 + v_x * hit_ground_time

                if bird_type[BIRD_BLUE]:
                    hit_ground_x += hit_ground_x * 0.2

                # This prevents shots that hit the ground before reaching any targets, though some levels might need it?
                if hit_ground_x < bbox_minx:
                    node.h = SBHeuristic.LARGE_VALUE
                    return node.h

            else:
                # Same calculation using only x values -> shoot at closest pig first
                node.h = self.time_to_pig_x(node, (x_0, y_0, v_x, v_y_0))
                pass
        return node.h

    @staticmethod
    def _backup_heuristic(node):
        """
        If nothing else works - but what state are we in when we get here? No pigs? No birds? something else?
        """
        return 0 # 1 / (1 + node.state_vars["['points_score']"])

    @staticmethod
    def time_to_pig(node, bird_coords):
        """
        Returns the distance (in planning time steps) to the closest pig.
        """
        # Plan:
        # 1. get pigs
        # 2. find closest pig
        # 3. divide distance to closets pig by? (scalar product of velocity and unit direction to pig. is that too involved?)
        # BIRD_X = 0
        # BIRD_Y = 1
        # BIRD_VX = 2
        # BIRD_VY = 3
        targets_x = {obj[7:]: node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['x_pig")}
        targets_y = {obj[7:]: node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['y_pig")}
        targets_xy = [(targets_x[o_id], targets_y[o_id]) for o_id in targets_x.keys()]
        dists = [(pig[0] - bird_coords[0]) ** 2 + (pig[1] - bird_coords[1]) ** 2 for pig in targets_xy]
        closest_ind = np.argmin(dists)
        vec_in_direction = (targets_xy[closest_ind][0] - bird_coords[0]) ** 2 / dists[closest_ind], \
                           (targets_xy[closest_ind][1] - bird_coords[1]) ** 2 / dists[closest_ind]
        speed_in_direction = bird_coords[2] * vec_in_direction[0] + bird_coords[3] * vec_in_direction[1]
        value = max(0, math.sqrt(dists[closest_ind]) / (speed_in_direction * constants.DELTA_T))
        return int(value)

    @staticmethod
    def time_to_pig_x(node, bird_coords):
        target_x = min([node.state_vars[obj] for obj in node.state_vars.keys() if obj.startswith("['x_pig")])
        return int(max(0, (target_x - bird_coords[0]) / (bird_coords[2] * constants.DELTA_T)))
