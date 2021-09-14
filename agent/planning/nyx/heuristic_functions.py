
from agent.planning.nyx.syntax.state import State
import math

def heuristic_function(state):
    # return 0

    if (state.state_vars["['total_failure']"]):
        return 999999

    return math.sqrt(math.pow(state.state_vars["['pos_x']"], 2) + math.pow(state.state_vars["['theta_x']"], 2) + math.pow(
        state.state_vars["['theta_x_dot']"], 2) + math.pow(state.state_vars["['pos_x_dot']"], 2) + math.pow(
        state.state_vars["['theta_x_ddot']"], 2) + math.pow(state.state_vars["['pos_x_ddot']"], 2) +
        math.pow(state.state_vars["['pos_y']"], 2) + math.pow(state.state_vars["['theta_y']"], 2) + math.pow(
            state.state_vars["['theta_y_dot']"], 2) + math.pow(state.state_vars["['pos_y_dot']"], 2) + math.pow(
            state.state_vars["['theta_y_ddot']"], 2) + math.pow(state.state_vars["['pos_y_ddot']"], 2)) * (
                       state.state_vars["['time_limit']"] - state.state_vars["['elapsed_time']"])
