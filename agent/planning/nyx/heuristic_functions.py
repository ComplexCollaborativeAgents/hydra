
from agent.planning.nyx.syntax.state import State
import math

def heuristic_function(state):
    return math.sqrt(math.pow(state.state_vars["['x']"], 2) + math.pow(state.state_vars["['theta']"], 2) + math.pow(
        state.state_vars["['theta_dot']"], 2) + math.pow(state.state_vars["['x_dot']"], 2) + math.pow(
        state.state_vars["['theta_ddot']"], 2) + math.pow(state.state_vars["['x_ddot']"], 2)) * (
                   state.state_vars["['time_limit']"] - state.state_vars["['elapsed_time']"])

    # return math.sqrt(math.pow(state.state_vars["['x']"],2) + math.pow(state.state_vars["['theta']"],2) + math.pow(state.state_vars["['theta_dot']"],2) + math.pow(state.state_vars["['x_dot']"],2))/(1.0+state.state_vars["['elapsed_time']"])
