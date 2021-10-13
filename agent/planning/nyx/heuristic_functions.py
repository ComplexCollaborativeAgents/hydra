
from agent.planning.nyx.syntax.state import State
import agent.planning.nyx.syntax.constants as constants
import math

def heuristic_function(state):
    # return 0

    if constants.CUSTOM_HEURISTIC_ID == 1:

        # CARTPOLE++ HEURISTIC

        if (state.state_vars["['total_failure']"]):
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


    else:
        return 0