'''
    This module provides very basic capabilities for simulating PDDL+ domain behavior, 
    including actions, events, and processes.
'''
from agent.planning.pddl_plus import *
from agent.planning.pddl_plus import is_float

''' A simplistic, probably not complete and sound, simulator for PDDL+ processes
 TODO: Replace this with a call to VAL. '''
class PddlPlusSimulator():

    ''' Return a list of (values_dict,t) pairs, where value_dict is a dictionary
     with the values of the fluents at time t, according to the given trace '''
    def trace_fluents(self, trace : list, fluent_names:list):
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
                values[fluent_name]=fluent_value
            output.append((values,t))
        return output

    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = float('inf')):
        state = PddlPlusState(problem.init)
        t = 0.0
        trace = [(state, t)]
        current_state = state.clone()
        plan = PddlPlusPlan(plan_to_simulate) # Clone the given plan
        next_timed_action  = plan.pop(0)

        # Ground the domain
        grounder = PddlPlusGrounder()
        domain = grounder.ground_domain(domain, problem)

        still_active = True
        while still_active and t<max_t:
            effects_list = list()

            # If we reached the time in which the next action should be applied, apply it
            if next_timed_action is not None and next_timed_action.start_at<=t:
                new_effect = self.compute_apply_action(current_state,next_timed_action.action)
                if new_effect is not None and len(new_effect)>0:
                    effects_list.extend(new_effect)

                # Next action
                if len(plan)>0:
                    next_timed_action = plan.pop(0)
                else:
                    next_timed_action = None

            # Trigger events
            for event in domain.events:
                if self.preconditions_hold(current_state, event.preconditions):
                    effects_list.extend(self.compute_fire_event(current_state, event))

            # Compute delta t
            if next_timed_action is None or next_timed_action.start_at > t+delta_t: # Next action should not start before t+delta_t
                current_delta_t = delta_t
            else: # Next action should start before t+delta_t, we don't want to miss it
                current_delta_t = next_timed_action.start_at - t

            # Advance processes
            for process in domain.processes:
                if self.preconditions_hold(current_state, process.preconditions):
                    effects_list.extend(self.compute_advance_process(current_state,process, current_delta_t))

            # Apply all impacts
            self.apply_effects(current_state, effects_list)

            # Advance time
            t = t+current_delta_t

            # Store current state to get trajectory
            trace.append((current_state.clone(), t))

            # Stopping condition
            if len(effects_list)>0:
                still_active = True
            else:
                if len(plan)==0:
                    still_active= False

        return current_state, t, trace

    ''' Apply an action on the given state. If binding is not None, we first ground the action with the binding'''
    def apply_action(self, state: PddlPlusState, action : PddlPlusWorldChange, binding: dict = None):
        if binding is not None:
            action = action.ground(binding)

        if self.preconditions_hold(state,action.preconditions)==False:
            raise ValueError("Action preconditions are not satisfied") # No effects if preconditions do not hold
        self.apply_effects(state, action.effects)

    ''' Fire a given event at the given state. If binding is not None, we first ground the action with the binding'''
    def fire_event(self, state: PddlPlusState, event : PddlPlusWorldChange, binding: dict = None):
        if binding is not None:
            event = event.ground(binding)

        if self.preconditions_hold(state,event.preconditions)==False:
            raise ValueError("Event preconditions are not satisfied") # No effects if preconditions do not hold
        self.apply_effects(state, event.effects)

    ''' Advances the process by the given delta_t time step. If binding is not None, we first ground the action with the binding'''
    def advance_process(self, state: PddlPlusState, process: PddlPlusWorldChange, delta_t: int, binding: dict = None):
        if binding is not None:
            process = process.ground(binding)
        if self.preconditions_hold(state, process.preconditions)==False:
            raise ValueError("Process preconditions are not satisfied") # No effects if preconditions do not hold

    ''' Compute the impact of applying an action on the given state. If binding is not None, we first ground the action with the binding'''
    def compute_apply_action(self, state: PddlPlusState, action: PddlPlusWorldChange, binding: dict = None):
        if binding is not None:
            action = action.ground(binding)

        if self.preconditions_hold(state, action.preconditions) == False:
            raise ValueError("Action preconditions are not satisfied")  # No effects if preconditions do not hold
        return self.compute_effect(state, action.effects)

    ''' Compute the impact of firing a given event at the given state. If binding is not None, we first ground the action with the binding'''
    def compute_fire_event(self, state: PddlPlusState, event: PddlPlusWorldChange, binding: dict = None):
        if binding is not None:
            event = event.ground(binding)

        if self.preconditions_hold(state, event.preconditions) == False:
            raise ValueError("Event preconditions are not satisfied")  # No effects if preconditions do not hold
        return self.compute_effect(state, event.effects)

    ''' Compute the impact of advancing the process by the given delta_t time step. If binding is not None, we first ground the action with the binding'''
    def compute_advance_process(self, state: PddlPlusState, process: PddlPlusWorldChange, delta_t: int,
                        binding: dict = None):
        if binding is not None:
            process = process.ground(binding)
        if self.preconditions_hold(state, process.preconditions) == False:
            raise ValueError("Process preconditions are not satisfied")  # No effects if preconditions do not hold

        return self.compute_effect(state, process.effects, delta_t)

    ''' Checks if a set of preconditions hold in the given state'''
    def preconditions_hold(self, state, preconditions):
        for precondition in preconditions:
            if self.__precondition_hold(state, precondition)==False:
                return False
        return True

    ''' Checks if a single precondition hold in the given state'''
    def __precondition_hold(self, state : PddlPlusState, single_precondition):
        if single_precondition[0]=="not":
            return not self.__precondition_hold(state, single_precondition[1])
        return self.__eval(single_precondition, state)

    ''' Apply the specified effect ont he given state '''
    def apply_effects(self, state, effects, delta_t=-1):
        for effect in effects:
            effect_type = effect[0] # increase or decrease
            if effect_type in ("increase", "decrease", "assign"):
                # Numeric fluent
                fluent_name = tuple(effect[1])
                old_value = float(state.numeric_fluents[fluent_name])
                delta = self.__eval(effect[2], state, delta_t)
                if effect_type=="increase":
                    new_value = old_value+delta
                elif effect_type=="decrease":
                    new_value = old_value-delta
                elif effect_type=="assign":
                    new_value = delta
                else:
                    raise NotImplementedError("Currently not supporting %s" % effect_type)

                state.numeric_fluents[fluent_name]=new_value # TODO: What if two effects affect the same fluent?
            else: # Boolean effects
                if effect[0]=="not": # remove a fact
                    fluent_name = tuple(effect[1])
                    state.boolean_fluents.remove(fluent_name)
                else: # add a fact
                    fluent_name = tuple(effect)
                    state.boolean_fluents.add(fluent_name)

    ''' Outputs a list of computed effects that should be applied. All effects are already computed. '''
    def compute_effect(self, state, effects, delta_t=-1):
        effect_list = list()

        for effect in effects:
            effect_type = effect[0] # increase or decrease
            if effect_type in ("increase", "decrease", "assign"):
                # Numeric fluent
                fluent_name = tuple(effect[1])
                delta = self.__eval(effect[2], state, delta_t)
                effect_list.append([effect_type, fluent_name, delta])
            else: # Boolean effects
                effect_list.append(effect)

        return effect_list

    ''' Evaluates a given expression using the fluents in the given state, and delta f, if needed'''
    def __eval(self, element, state: PddlPlusState, delta_t: float = -1):
        if isinstance(element, list):
            if self.is_op(element[0]): # If element is an operator
                assert len(element) == 3

                op_name = element[0]
                value1 = self.__eval(element[1], state, delta_t)
                value2 = self.__eval(element[2], state, delta_t)

                if op_name == "=":
                    op_name = "==" # We want comparison, not assignment
                return eval("%s%s%s" % (str(value1), op_name, str(value2)))
            else: # Else element is a fluent value
                if len(element)==1: # This is a fluent
                    fluent_name = (element[0],) # Todo: understand why this hack is needed
                else:
                    fluent_name = tuple(element)

                if fluent_name in state.numeric_fluents:
                    return float(state.numeric_fluents[fluent_name])
                else:
                    return fluent_name in state.boolean_fluents
        else:
            if is_float(element):
                return float(element) # A constant
            elif element=="#t":
                if delta_t==-1:
                    raise ValueError("Delta t not set, but needed for evaluation")
                return delta_t
            else:
                return element
                # return float(state.numeric_fluents[element])  # A fluent

    ''' Check if the given string is one of the supported mathematical operations '''
    def is_op(self, op_name:str):
        if op_name in ("+-/*=><"):
            return True
        elif op_name == "<=" or op_name == ">=":
            return True
        else:
            return False

