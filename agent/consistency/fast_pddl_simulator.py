'''
    This module provides very basic capabilities for simulating PDDL+ domain behavior,
    including actions, events, and processes.
'''
from agent.planning.pddl_meta_model import *
from agent.consistency.observation import *
from agent.consistency.pddl_plus_simulator import *

''' A simplistic, probably not complete and sound, simulator for PDDL+ processes
 TODO: Replace this with a call to VAL. '''
class CachingPddlPlusSimulator(PddlPlusSimulator):
    def __init__(self):
        self.context = dict()


    ''' Simulate running the given plan from the start state '''
    def simulate(self, plan_to_simulate: PddlPlusPlan, problem: PddlPlusProblem, domain: PddlPlusDomain, delta_t:float, max_t:float = 1000, max_iterations: float = 1000):
        self.problem = problem
        # Ground the domain
        grounder = PddlPlusGrounder(no_dummy_objects=False)
        self.domain = grounder.ground_domain(domain, problem)

        # print ("\n\nGROUNDED DOMAIN: \n")
        # print (self.domain.name)
        # print (self.domain.types)
        # print (self.domain.predicates)
        # print (self.domain.functions)
        # for pro in self.domain.processes:
        #     pro.print_info()
        # print ("")
        # for ev in self.domain.events:
        #     ev.print_info()
        # print ("")
        # for ac in self.domain.actions:
        #     ac.print_info()

        state = PddlPlusState(problem.init)
        t = 0.0
        trace = []
        current_state = state.clone()
        plan = PddlPlusPlan(plan_to_simulate) # Clone the given plan

        # print("\n\nACTIONS IN COPIED PLAN")
        # for acts in plan:
        #     print (acts.action_name, " at ", acts.start_at)

        if len(plan)==0:
            raise ValueError("Plan is empty")
        next_timed_action  = plan.pop(0)
        # print("\n\nNEXT TIMED ACTION: ", next_timed_action.action_name, " at ", next_timed_action.start_at)

        # Create the first trace_item
        trace_item = [None, None, None]
        trace.append(trace_item)
        trace_item[TI_STATE]=current_state.clone() # TODO: Maybe this close is redundant
        trace_item[TI_T] = t
        world_changes_at_t = []
        trace_item[TI_WORLD_CHANGES] = world_changes_at_t

        still_active = True
        while still_active and t<max_t and t/delta_t<max_iterations:
            self.context.clear()  # New t value may change the cached values TODO: Smarter caching

            # If we reached the time in which the next action should be applied, apply it
            # print ("TIME = ", t)
            if next_timed_action is not None and next_timed_action.start_at<=t:
                # Get the WorldChange object for the action to perform
                # print ("APPLIED TIMED ACTION: ", next_timed_action.action_name, " at time = ", next_timed_action.start_at, "/", t)
                world_change = self.domain.get_action(next_timed_action.action_name)
                new_effects = self.compute_apply_action(current_state, world_change)
                world_changes_at_t.append(world_change)
                if new_effects is not None and len(new_effects)>0:
                    self.apply_effects(current_state, new_effects)

                # Next action
                if len(plan)>0:
                    next_timed_action = plan.pop(0)
                else:
                    next_timed_action = None

            # Compute delta t
            if next_timed_action is None or next_timed_action.start_at > t+delta_t: # Next action should not start before t+delta_t
                current_delta_t = delta_t
            else: # Next action should start before t+delta_t, we don't want to miss it
                current_delta_t = next_timed_action.start_at - t

            # Trigger events after action is performed
            fired_events = self.handle_events(current_state)
            for event in fired_events:
                world_changes_at_t.append(event)

            # Advance process and apply events
            active_processes = self.handle_processes(current_state, current_delta_t) # Advance processes
            for process in active_processes:
                world_changes_at_t.append(process)

            t = t+current_delta_t # Advance time

            # Add new trace item, which will be completed by the next iteration of this while
            trace_item = [None, None, None]
            trace_item[TI_STATE]=current_state.clone()
            trace_item[TI_T] = t
            world_changes_at_t = []
            trace_item[TI_WORLD_CHANGES]=world_changes_at_t # Actions, events, and process, performed in this (state,time) pair
            trace.append(trace_item)

            # Stopping condition
            if len(active_processes)>0:
                still_active = True
            elif len(fired_events)>0:
                still_active = True # This one is debatable TODO: Consult Wiktor and Matt
            elif next_timed_action is not None:
                # In this case, everything is waiting for the next timed action, so we can just "jump ahead" to get to do that action.
                still_active = True
                t = next_timed_action.start_at
            else:
                still_active = False

        return current_state, t, trace

    ''' Apply the specified effect ont he given state '''
    def apply_effects(self, state, effects, delta_t=-1):
        self.context.clear() # Effects may change the cached values TODO: Smarter caching
        super().apply_effects(state, effects,delta_t)

    ''' Evaluates a given expression using the fluents in the given state, and delta f, if needed'''
    def _eval(self, element, state: PddlPlusState, delta_t: float = -1):
        element_str = str(element)
        if element_str in self.context:
            # print("Cache hit")
            return self.context[element_str]
        # else:
            # print("Cache miss")
        # print("Eval %s" % str(element))
        if isinstance(element, list):
            if self.is_op(element[0]): # If element is an operator
                assert len(element) == 3

                # Below: inling the __eval__ call for performance
                if is_float(element[1]):
                    value1 = float(element[1])
                else:
                    value1 = self._eval(element[1], state, delta_t)
                if is_float(element[2]):
                    value2 = float(element[2])
                else:
                    value2 = self._eval(element[2], state, delta_t)

                op_name = element[0]
                assert(is_float(value1))
                assert(is_float(value2))

                # Switch case
                if op_name=="+":
                    result = value1 + value2
                elif op_name=="-":
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

                self.context[element_str] = result
                return result
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
