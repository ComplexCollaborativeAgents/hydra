import logging

import numpy as np

from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants
import math

def heuristic_function(state):
    # return 0

    if constants.CUSTOM_HEURISTIC_ID == 1:

        # CARTPOLE++ HEURISTIC

        if state.state_vars["['total_failure']"]:
            return 999999
        return math.sqrt(math.pow(state.state_vars["['pos_x']"], 2) + math.pow(state.state_vars["['theta_x']"], 2) + math.pow(
            state.state_vars["['theta_x_dot']"], 2) + math.pow(state.state_vars["['pos_x_dot']"], 2) + math.pow(
            state.state_vars["['theta_x_ddot']"], 2) + math.pow(state.state_vars["['pos_x_ddot']"], 2) +
            math.pow(state.state_vars["['pos_y']"], 2) + math.pow(state.state_vars["['theta_y']"], 2) + math.pow(
                state.state_vars["['theta_y_dot']"], 2) + math.pow(state.state_vars["['pos_y_dot']"], 2) + math.pow(
                state.state_vars["['theta_y_ddot']"], 2) + math.pow(state.state_vars["['pos_y_ddot']"], 2)) * (
                           state.state_vars["['time_limit']"] - state.state_vars["['elapsed_time']"])

    elif constants.CUSTOM_HEURISTIC_ID == 2:

        # SCIENCE BIRDS HEURISTIC

        return 1 / (1 + state.state_vars["['points_score']"])

        # return 0

    elif constants.CUSTOM_HEURISTIC_ID == 3:

        # POLYCRAFT HEURISTIC

        return 0

    elif constants.CUSTOM_HEURISTIC_ID == 4:

        # CARTPOLE HEURISTIC

        return math.sqrt(math.pow(state.state_vars["['x']"], 2) + math.pow(state.state_vars["['theta']"], 2) + math.pow(
            state.state_vars["['theta_dot']"], 2) + math.pow(state.state_vars["['x_dot']"], 2) + math.pow(
            state.state_vars["['theta_ddot']"], 2) + math.pow(state.state_vars["['x_ddot']"], 2)) * (
                       state.state_vars["['time_limit']"] - state.state_vars["['elapsed_time']"])

    elif constants.CUSTOM_HEURISTIC_ID == 5:
        # This heuristic only calculates ballistic birds hitting the bounding box that has targets in it. Any change
        # from ballistic motion (e.g. friction, bird powers, updrafts) will require a new heuristic.

        # Find active bird ID
        active_bird_string = [key for key in state.state_vars.keys()
                              if (key.startswith("['bird_id',")
                                  and state.state_vars[key] == state.state_vars["['active_bird']"])]
        if len(active_bird_string) < 1:
            # This heuristic doesn't know what do to without a birb
            return 1 / (1 + state.state_vars["['points_score']"])
        active_bird_string = active_bird_string[0][10:]
        bird_released = state.state_vars.get("['bird_released'" + active_bird_string)
        # Only have meaningful heuristic if bird is not yet launched. Once bird is launched, after that we just watch.
        if not bird_released:
            # Find the bounding box for all targets:
            targets_x = {obj[9:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['x_block")}
            targets_x.update({obj[7:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['x_pig")})
            targets_w = {obj[13:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['block_width")}
            pig_radii = {obj[12:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['pig_radius")}
            targets_w.update(pig_radii)
            targets_max_x = [targets_x[o_id] + targets_w[o_id] for o_id in targets_x.keys()]
            targets_min_x = [targets_x[o_id] - targets_w[o_id] for o_id in targets_x.keys()]
            bbox_minx = min(targets_min_x)
            bbox_maxx = max(targets_max_x)
            targets_y = {obj[9:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['y_block")}
            targets_y.update({obj[7:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['y_pig")})
            targets_h = {obj[14:]:state.state_vars[obj] for obj in state.state_vars.keys() if obj.startswith("['block_height")}
            targets_h.update(pig_radii)
            targets_max_y = [targets_x[o_id] + targets_w[o_id] for o_id in targets_x.keys()]
            targets_min_y = [targets_x[o_id] - targets_w[o_id] for o_id in targets_x.keys()]
            bbox_maxy = max(targets_max_y)
            bbox_miny = min(targets_min_y)

            # Find bird trajectory:
            angle = state.state_vars["['angle']"]
            # I would love to know where these calculations came from. v_x looks like an approximation of cos(angle), but that will only be correct for angles up to ~15 degrees
            v_y_0 = state.state_vars["['v_bird'" + active_bird_string] * (
                    (4 * angle * (180 - angle)) / (40500 - (angle * (180 - angle))))
            v_x = state.state_vars["['v_bird'" + active_bird_string] * (1 - ((np.power((angle * 0.0174533), 2)) / 2))
            if v_x == 0:
                # In this situation, we can't reason about ballistics.
                # I have concluded that this situation doesn't arise, but left this just in case to prevent div by 0.
                return 1 / (1 + state.state_vars["['points_score']"])

            y_0 = state.state_vars["['y_bird'" + active_bird_string]
            x_0 = state.state_vars["['x_bird'" + active_bird_string]
            gravity = state.state_vars["['gravity']"]

            # Bounding box intersections:
            t_x_min_box = (bbox_minx - x_0) / v_x
            t_x_max_box = (bbox_maxx - x_0) / v_x
            y_top = y_0 + (0.5 * np.power(v_y_0, 2)) / gravity
            y_enter = y_0 + v_y_0 - 0.5 * gravity * np.power(t_x_min_box, 2)
            y_exit = y_0 + v_y_0 - 0.5 * gravity * np.power(t_x_max_box, 2)
            max_y = max(y_top, y_enter, y_exit)
            min_y = min(y_enter, y_exit)

            if min_y > bbox_maxy or max_y < bbox_miny:
                # I expect that the entire shot passing under the lowest object is unlikely, but it's very easy to rule out.
                return 999999  # missed everything in the level entirely

            # This prevents shots that hit the ground before reaching any targets, though some levels might need it?
            hit_ground_time = (- v_y_0 + np.sqrt(np.power(v_y_0, 2) + 2 * gravity * y_0)) / (2 * gravity)
            hit_ground_x = x_0 + v_x * hit_ground_time
            if hit_ground_x < bbox_minx:
                return 999999
        return 1 / (1 + state.state_vars["['points_score']"])

    else:
        return 0