"""
    This module provides very basic capabilities for simulating PDDL+ domain behavior,
    including actions, events, and processes.
"""
import settings
from agent.planning.meta_model import *
from agent.consistency.observation import *
from agent.planning.pddl_plus import is_op

# Constants
TI_STATE = 0  # Index in a trace_item for the state
TI_T = 1  # Index in a trace_item for the time
TI_WORLD_CHANGES = 2  # Index in a trace_item for the list of actions, processes, and events


class InconsistentPlanError(ValueError):
    """ Error thrown when the simulator tries to simulate a plan that is inconsistent
    (e.g., try to perform an action without its preconditions satisfied)"""

    def __init__(self, error_msg: str):
        super().__init__(error_msg)


class PddlPlusSimulator:
    """ A simplistic, probably not complete and sound, simulator for PDDL+ processes
     TODO: Replace this with a call to VAL. """
    def __init__(self, allow_cascading_effects=False):
        self.delta_t = None
        self.next_timed_action = None
        self.plan = None
        self.trace = None
        self.domain = None
        self.problem = None
        self.allow_cascading_effects = allow_cascading_effects
        self.ignore_failed_actions = True # If this is true, the simulation will stop properly when simulated actions cannot be applied. If false, an InconsistentPlanError will be thrown.

    def trace_fluents(self, trace: list, fluent_names: list):
        """ Return a list of (values_dict,t) pairs, where value_dict is a dictionary
            with the values of the fluents at time t, according to the given trace """
        output = list()
        for (state, t) in trace:
            values = dict()
            for fluent_name in fluent_names:
                if fluent_name in state.numeric_fluents:
                    fluent_value = state.numeric_fluents[fluent_name]
                elif fluent_name in state.boolean_fluents:
                    fluent_value = True
                else:
                    fluent_value = False
                values[fluent_name] = fluent_value
            output.append((values, t))
        return output

    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t: float,
                 max_t: float = 1000, max_iterations: float = 1000):
        """ Simulate running the given plan from the start state """
        self.problem = problem
        self.domain = domain
        self.trace = []
        self.plan = PddlPlusPlan(plan_to_simulate)  # Clone the given plan
        if len(self.plan) == 0:
            self.next_timed_action = None
        else:
            self.next_timed_action = self.plan.pop(0)
        self.delta_t = delta_t

        # Initial state and first action
        state = PddlPlusState(problem.init)
        current_state = state.clone()
        t = 0.0

        # Create the first trace_item
        trace_item = [None, None, None]
        self.trace.append(trace_item)
        trace_item[TI_STATE] = current_state.clone()  # TODO: Maybe this clone is redundant
        trace_item[TI_T] = t
        world_changes_at_t = []
        trace_item[TI_WORLD_CHANGES] = world_changes_at_t

        still_active = True
        try:
            while still_active and t<max_t and t/self.delta_t<max_iterations:
                still_active, t = self._sim_step(current_state, t)
        except InconsistentPlanError as e: # Sometimes the repair makes the executed plan be inconsistent #TODO: Discuss this
            if self.ignore_failed_actions==False:
                raise e

        return current_state, t, self.trace

    def _sim_step(self, current_state, t):
        """ Simulate a single step """
        world_changes_at_t = []

        # If we reached the time in which the next action should be applied, apply it
        if self.next_timed_action is not None and self.next_timed_action.start_at <= t:
            world_change = self.domain.get_action(self.next_timed_action.action_name)
            new_effects = self.compute_apply_action(current_state, world_change)
            world_changes_at_t.append(world_change)
            if new_effects is not None and len(new_effects) > 0:
                self.apply_effects(current_state, new_effects)

            # Next action
            if len(self.plan) > 0:
                self.next_timed_action = self.plan.pop(0)
            else:
                self.next_timed_action = None

            # Trigger events after action is performed
            fired_events = self.handle_events(current_state)
            for event in fired_events:
                world_changes_at_t.append(event)

        # Compute delta t
        # if self.next_timed_action is None or self.next_timed_action.start_at >= t + self.delta_t - 0.00001:  # Next action should not start before t+delta_t
        #     current_delta_t = self.delta_t
        # else:  # Next action should start before t+delta_t, we don't want to miss it
        #     current_delta_t = round(self.next_timed_action.start_at - t, 10)
        current_delta_t = self.delta_t

        # Advance process and apply events
        active_processes = self.handle_processes(current_state, current_delta_t)  # Advance processes
        for process in active_processes:
            world_changes_at_t.append(process)
        fired_events = self.handle_events(current_state)  # Apply events to the resulting state
        for event in fired_events:
            world_changes_at_t.append(event)

        # Add new trace item, which will be completed by the next iteration of this while
        trace_item = [None, None, None]
        trace_item[TI_STATE] = current_state.clone()
        trace_item[TI_T] = t
        trace_item[TI_WORLD_CHANGES] = world_changes_at_t  # Actions, events, and process, performed in this (state,time) pair
        self.trace.append(trace_item)
        # Stopping condition
        if len(active_processes) > 0:
            still_active = True
        elif len(fired_events) > 0:
            still_active = True  # This one is debatable TODO: Consult Wiktor and Matt
        elif self.next_timed_action is not None:
            # In this case, everything is waiting for the next timed action, so we can just "jump ahead" to get to do that action.
            still_active = True
            # t = self.next_timed_action.start_at
        else:
            still_active = False
        t = round(t + current_delta_t, 10)  # Advance time
        return still_active, t

    def simulate_observed_action(self, game_state, game_action, game_meta_model, delta_t: float = None):
        """ Simulate the trace of a given action in a given state according to the given meta model"""
        if delta_t is None:
            delta_t = game_meta_model.delta_t
        problem = game_meta_model.create_pddl_problem(game_state)
        domain = game_meta_model.create_pddl_domain(game_state)
        # domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains
        plan = PddlPlusPlan()
        plan.append(game_meta_model.create_timed_action(game_action, game_state))
        (_, _, trace) = self.simulate(plan, problem, domain, delta_t)
        return trace

    def simulate_observed_plan(self, game_states, game_actions, game_meta_model, delta_t: float):
        """ Simulate the trace of a given action in a given state according to the given meta model"""
        problem = game_meta_model.create_pddl_problem(game_states[0])
        domain = game_meta_model.create_pddl_domain(game_states[0])
        domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains

        plan = PddlPlusPlan()
        for ix in enumerate(game_actions):
            plan.append(game_meta_model.create_timed_action(game_actions[ix], ix))
        (_, _, trace) = self.simulate(plan, problem, domain, delta_t)
        return trace

    def handle_processes(self, state, delta_t):
        """  Advanced all process in the given state by the given delta t"""
        effects_list = list()
        active_processes = list()
        for process in self.domain.processes:
            if self.preconditions_hold(state, process.preconditions):
                active_processes.append(process)
        for process in active_processes:
            effects_list.extend(self.compute_advance_process(state, process, delta_t))
        # Apply all impacts
        self.apply_effects(state, effects_list)
        return active_processes

    def handle_events(self, state: PddlPlusState):
        """ Processes the events and modify the current state accordingly"""
        events_to_fire = []
        effects_list = []
        available_events = list(self.domain.events)
        fired_events = []
        while True:
            events_to_fire.clear()
            for event in available_events:
                if self.preconditions_hold(state, event.preconditions):
                    events_to_fire.append(event)
            if len(events_to_fire) == 0:
                break

            for event in events_to_fire:
                effects_list.extend(self.compute_fire_event(state, event))
                available_events.remove(event)
                fired_events.append(event)
            self.apply_effects(state, effects_list)
            if len(available_events) == 0:
                break

            if not self.allow_cascading_effects:
                break

        return fired_events

    def apply_action(self, state: PddlPlusState, action: PddlPlusWorldChange, binding: dict = None):
        """ Apply an action on the given state. If binding is not None, we first ground the action with the binding"""
        if binding is not None:
            action = action.ground(binding)

        if not self.preconditions_hold(state, action.preconditions):
            raise InconsistentPlanError(
                "Action preconditions are not satisfied")  # No effects if preconditions do not hold
        self.apply_effects(state, action.effects)

    def fire_event(self, state: PddlPlusState, event: PddlPlusWorldChange, binding: dict = None):
        """ Fire a given event at the given state.
        If binding is not None, we first ground the action with the binding"""
        if binding is not None:
            event = event.ground(binding)

        if not self.preconditions_hold(state, event.preconditions):
            raise ValueError("Event preconditions are not satisfied")  # No effects if preconditions do not hold
        self.apply_effects(state, event.effects)

    def advance_process(self, state: PddlPlusState, process: PddlPlusWorldChange, delta_t: int, binding: dict = None):
        """ Advances the process by the given delta_t time step.
         If binding is not None, we first ground the action with the binding"""
        if binding is not None:
            process = process.ground(binding)
        if not self.preconditions_hold(state, process.preconditions):
            raise ValueError("Process preconditions are not satisfied")  # No effects if preconditions do not hold

    def compute_apply_action(self, state: PddlPlusState, action: PddlPlusWorldChange, binding: dict = None):
        """ Compute the impact of applying an action on the given state.
        If binding is not None, we first ground the action with the binding"""
        if binding is not None:
            action = action.ground(binding)

        if not self.preconditions_hold(state, action.preconditions):
            raise InconsistentPlanError(
                "Action %s preconditions are not satisfied" % action.name)  # No effects if preconditions do not hold
        return self.compute_effects(state, action.effects)

    def compute_fire_event(self, state: PddlPlusState, event: PddlPlusWorldChange, binding: dict = None):
        """ Compute the impact of firing a given event at the given state. If binding is not None,
        we first ground the action with the binding"""
        if binding is not None:
            event = event.ground(binding)

        if not self.preconditions_hold(state, event.preconditions):
            raise ValueError("Event preconditions are not satisfied")  # No effects if preconditions do not hold
        return self.compute_effects(state, event.effects)

    def compute_advance_process(self, state: PddlPlusState, process: PddlPlusWorldChange, delta_t: int,
                                binding: dict = None):
        """ Compute the impact of advancing the process by the given delta_t time step. If binding is not None,
        we first ground the action with the binding"""
        if binding is not None:
            process = process.ground(binding)
        if not self.preconditions_hold(state, process.preconditions):
            raise ValueError("Process preconditions are not satisfied")  # No effects if preconditions do not hold

        return self.compute_effects(state, process.effects, delta_t)

    def preconditions_hold(self, state, preconditions):
        """ Checks if a set of preconditions hold in the given state"""
        for precondition in preconditions:
            if not self.__precondition_hold(state, precondition):
                return False
        return True

    ''' Checks if a single precondition hold in the given state'''

    def __precondition_hold(self, state: PddlPlusState, single_precondition):
        if single_precondition[0] == "not":
            return not self.__precondition_hold(state, single_precondition[1])
        if single_precondition[0] == "or":
            for precondition in single_precondition[1:]:
                if self.__precondition_hold(state, precondition) == True:
                    return True
            return False
        return self._eval(single_precondition, state)

    ''' Apply the specified effect ont he given state '''

    def apply_effects(self, state, effects, delta_t=-1):
        for effect in effects:
            effect_type = effect[0]  # increase or decrease
            if effect_type in ("increase", "decrease", "assign"):
                # Numeric fluent
                fluent_name = tuple(effect[1])
                old_value = float(state.numeric_fluents[fluent_name])
                delta = self._eval(effect[2], state, delta_t)
                if effect_type == "increase":
                    new_value = old_value + delta
                elif effect_type == "decrease":
                    new_value = old_value - delta
                elif effect_type == "assign":
                    new_value = delta
                else:
                    raise NotImplementedError("Currently not supporting %s" % effect_type)

                state.numeric_fluents[fluent_name] = new_value  # TODO: What if two effects affect the same fluent?
            else:  # Boolean effects
                if effect[0] == "not":  # remove a fact
                    fluent_name = tuple(effect[1])
                    if fluent_name in state.boolean_fluents:
                        state.boolean_fluents.remove(fluent_name)
                else:  # add a fact
                    fluent_name = tuple(effect)
                    state.boolean_fluents.add(fluent_name)

    def compute_effects(self, state, effects, delta_t=-1):
        """ Outputs a list of computed effects that should be applied. All effects are already computed. """
        effect_list = list()

        for effect in effects:
            effect_type = effect[0]  # increase or decrease
            if effect_type in ("increase", "decrease", "assign"):
                # Numeric fluent
                fluent_name = tuple(effect[1])
                delta = self._eval(effect[2], state, delta_t)
                effect_list.append([effect_type, fluent_name, delta])
            else:  # Boolean effects
                effect_list.append(effect)

        return effect_list

    def _eval(self, element, state: PddlPlusState, delta_t: float = -1):
        """ Evaluates a given expression using the fluents in the given state, and delta f, if needed"""
        if isinstance(element, list):
            if is_op(element[0]):  # If element is an operator
                assert len(element) == 3

                op_name = element[0]
                value1 = self._eval(element[1], state, delta_t)
                value2 = self._eval(element[2], state, delta_t)

                # assert(is_float(value1))
                # assert(is_float(value2))

                # Switch case
                if op_name == "+":
                    result = value1 + value2
                elif op_name == "-":
                    result = value1 - value2
                elif op_name == "*":
                    result = value1 * value2
                elif op_name == "/":
                    result = value1 / value2
                elif op_name == ">":
                    result = value1 > value2
                elif op_name == "<":
                    result = value1 < value2
                elif op_name == "<=":
                    result = value1 <= value2
                elif op_name == ">=":
                    result = value1 >= value2
                elif op_name == "=":
                    result = value1 == value2
                else:
                    result = eval("%s%s%s" % (str(value1), op_name, str(value2)))
                return result

            else:  # Else element is a fluent value
                if len(element) == 1:  # This is a fluent
                    fluent_name = (element[0],)  # Todo: understand why this hack is needed
                else:
                    fluent_name = tuple(element)
                if fluent_name in state.numeric_fluents:
                    return float(state.numeric_fluents[fluent_name])
                else:
                    return fluent_name in state.boolean_fluents
        else:
            if is_float(element):
                return float(element)  # A constant
            elif element == "#t":
                if delta_t == -1:
                    raise ValueError("Delta t not set, but needed for evaluation")
                return delta_t
            else:
                return element
                # return float(state.numeric_fluents[element])  # A fluent

    def get_expected_trace(self, observation, meta_model, delta_t=0.05):
        """ Simulate the observed action in the observed state according to the given meta model """
        problem = meta_model.create_pddl_problem(observation.get_initial_state())
        domain = meta_model.create_pddl_domain(observation.get_initial_state())
        domain = PddlPlusGrounder().ground_domain(domain, problem)  # Simulator accepts only grounded domains

        # act_start_idx = 0
        # if meta_model.constant_numeric_fluents['time_limit']/settings.CP_DELTA_T < 200:
        #     act_start_idx = int(200 - (meta_model.constant_numeric_fluents['time_limit']/delta_t))

        plan = observation.get_pddl_plan(meta_model)

        (_, _, trace,) = self.simulate(plan, problem, domain, delta_t=delta_t)
        return trace, plan


def simulate_plan_trace(plan: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t: float = 0.05,
                        simulator=PddlPlusSimulator()):
    """ Helper function: simulate the given plan, on the given problem and domain.  """
    (_, _, trace) = simulator.simulate(plan, problem, domain, delta_t)
    return trace
