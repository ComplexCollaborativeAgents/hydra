import math
import re
import time
from collections import namedtuple
from typing import List, Tuple, Iterator, Dict, Union, Optional

import logging

from agent.consistency.observation import HydraObservation
from agent.consistency.pddl_plus_simulator import PddlPlusSimulator
from agent.planning.meta_model import MetaModel
from agent.planning.pddl_plus import PddlPlusPlan, PddlPlusProblem, PddlPlusDomain, PddlPlusWorldChange, \
    WorldChangeTypes, PddlPlusState

from agent.planning.nyx import PDDL
from agent.planning.nyx.syntax import constants as nyx_constants
from agent.planning.nyx.syntax.action import Action
from agent.planning.nyx.syntax.event import Event
from agent.planning.nyx.syntax.process import Process
from agent.planning.nyx.syntax.plan import Plan as NyxPlan
from agent.planning.nyx.syntax.state import State as NyxState
from agent.planning.nyx.syntax.trace import Trace as NyxTrace

TraceItem = namedtuple('TraceItem', ['state', 'time', 'world_changes'])
Trace = List[TraceItem]
NyxHappening = Union[Action, Event, Process]
SimulationOutput = Tuple[Optional[PddlPlusState], Optional[float], Optional[Trace]]


class NyxPddlPlusSimulator(PddlPlusSimulator):

    def __init__(self, allow_cascading_effects: bool = False):
        super().__init__(allow_cascading_effects=allow_cascading_effects)

    def get_expected_trace(self,
                           observation: HydraObservation,
                           meta_model: MetaModel,
                           delta_t: float = 0.05,
                           max_t: Optional[float] = None,
                           max_iterations=1000) -> Tuple[Trace, PddlPlusPlan]:
        problem = meta_model.create_pddl_problem(observation.get_initial_state())
        domain = meta_model.create_pddl_domain(observation.get_initial_state())
        plan = observation.get_pddl_plan(meta_model)
        _, _, trace = self.simulate(plan, problem, domain, delta_t=delta_t, max_t=max_t, max_iterations=max_iterations)
        return trace, plan

    def simulate(self,
                 plan_to_simulate: PddlPlusPlan,
                 problem: PddlPlusProblem,
                 domain: PddlPlusDomain,
                 delta_t: float,
                 max_t: Optional[float] = None,
                 max_iterations: int = 1000) -> SimulationOutput:
        grounded_pddl = self.grounded_instance(domain, problem)
        return self.simulate_grounded_instance(plan_to_simulate,
                                               grounded_pddl,
                                               delta_t,
                                               max_t=max_t,
                                               max_iterations=max_iterations)

    def simulate_grounded_instance(self,
                                   plan_to_simulate: PddlPlusPlan,
                                   grounded_pddl: PDDL.GroundedPDDLInstance,
                                   delta_t: float,
                                   max_t: Optional[float] = None,
                                   max_iterations: int = 1000) -> SimulationOutput:
        nyx_constants.set_delta_t(delta_t)
        nyx_plan = self._nyx_plan(plan_to_simulate, grounded_pddl, delta_t, max_t=max_t)
        nyx_trace = nyx_plan.simulate(grounded_pddl.init_state,
                                      grounded_pddl,
                                      double_events=self.allow_cascading_effects,
                                      max_iterations=max_iterations,
                                      check_fired=True)
        hydra_trace = self._hydra_trace(nyx_trace)
        if len(hydra_trace) == 0:
            return None, None, None
        return hydra_trace[-1].state, hydra_trace[-1].time, hydra_trace

    def grounded_instance(self,
                          domain: PddlPlusDomain,
                          problem: PddlPlusProblem) -> PDDL.GroundedPDDLInstance:
        nyx_domain = self._nyx_domain(domain)
        nyx_problem = self._nyx_problem(problem)
        return PDDL.GroundedPDDLInstance(nyx_domain, nyx_problem)

    @classmethod
    def _nyx_domain(cls, domain: PddlPlusDomain) -> PDDL.PDDLDomain:
        predicates = {name: cls.__unpack_arguments(args) for name, *args in domain.predicates}
        functions = {name: cls.__unpack_arguments(args) for name, *args in domain.functions}

        nyx_domain = PDDL.PDDLDomain(name=domain.name,
                                     requirements=domain.requirements,
                                     predicates=predicates,
                                     functions=functions,
                                     types=cls.__unpack_arguments(domain.types, type_hierarchy=True),
                                     constants=cls.__unpack_arguments(domain.constants, type_hierarchy=True))

        for source, destination, happening_type in [(domain.actions, nyx_domain.actions, Action),
                                                    (domain.events, nyx_domain.events, Event),
                                                    (domain.processes, nyx_domain.processes, Process)]:
            for happening in source:
                if happening.parameters:
                    parameters = [[arg, t] for args, t in cls.__iterate_arguments(happening.parameters[0]) for arg in args]
                else:
                    parameters = []
                destination.append(happening_type(happening.name,
                                                  parameters,
                                                  happening.preconditions,
                                                  happening.effects))

        return nyx_domain

    @classmethod
    def _nyx_problem(cls, problem: PddlPlusProblem) -> PDDL.PDDLProblem:
        objects = {}
        for obj_name, obj_type in problem.objects:
            if obj_type not in objects:
                objects[obj_type] = []
            objects[obj_type].append(obj_name)

        nyx_problem = PDDL.PDDLProblem(name=problem.name,
                                       init=problem.init,
                                       objects=objects,
                                       goal=problem.goal,
                                       metric=problem.metric)
        return nyx_problem

    @classmethod
    def _nyx_plan(cls, plan: PddlPlusPlan,
                  grounded_pddl: PDDL.GroundedPDDLInstance,
                  delta_t: float,
                  expand_time_passing: bool = True,
                  max_t: Optional[float] = None) -> NyxPlan:
        nyx_plan = NyxPlan()
        action_lookup = {action.grounded_name.lower(): action for action in grounded_pddl.actions}

        for action in plan:
            nyx_action = action_lookup.get(action.action_name.lower())
            start_time = math.floor(action.start_at / delta_t) * delta_t
            nyx_plan.append_action(nyx_action, start_time, expand_time_passing=expand_time_passing)

        if max_t is not None:
            nyx_plan.pass_time(max_t)

        return nyx_plan

    @classmethod
    def _hydra_trace(cls, trace: NyxTrace, include_time_passing: bool = False) -> Trace:
        hydra_trace = []

        current_state = None
        happenings = []
        for state in trace.iter(extended=True):
            if current_state is not None and current_state.time != state.time:
                hydra_trace.append(TraceItem(cls._hydra_state(current_state),
                                             current_state.time,
                                             happenings))
                happenings = []
            current_state = state
            if state.predecessor_action:
                if include_time_passing or state.predecessor_action is not nyx_constants.TIME_PASSING_ACTION:
                    happenings.append(cls._hydra_world_change(state.predecessor_action))
        else:
            if current_state is not None:
                hydra_trace.append(TraceItem(cls._hydra_state(current_state),
                                             current_state.time,
                                             happenings))

        return hydra_trace

    @classmethod
    def _hydra_world_change(cls, happening: NyxHappening) -> PddlPlusWorldChange:
        if isinstance(happening, Action):
            change_type = WorldChangeTypes.action
        elif isinstance(happening, Event):
            change_type = WorldChangeTypes.event
        elif isinstance(happening, Process):
            change_type = WorldChangeTypes.process
        else:
            raise TypeError("Unrecognized Nyx Happening type: {}".format(str(type(happening))))

        world_change = PddlPlusWorldChange(change_type)
        world_change.name = happening.grounded_name
        world_change.preconditions = happening.preconditions
        world_change.effects = happening.effects

        return world_change

    @classmethod
    def _hydra_state(cls, state: NyxState) -> PddlPlusState:
        hydra_state = PddlPlusState()
        for key, value in state.state_vars.items():
            mapping = tuple(re.findall("'([^']*)'", key))
            if isinstance(value, bool):
                if value:
                    hydra_state.boolean_fluents.add(mapping)
            else:
                hydra_state.numeric_fluents[mapping] = value
        return hydra_state

    @classmethod
    def __iterate_arguments(cls, expr: List[str]) -> Iterator[Tuple[List[str], str]]:
        head = tail = 0
        while tail < len(expr):
            if expr[tail] == '-':
                yield expr[head:tail], expr[tail + 1]
                head = tail = tail + 2
            else:
                tail += 1
        if head != tail:
            yield expr[head:tail], 'object'

    @classmethod
    def __unpack_arguments(cls, expr, type_hierarchy=False) -> Dict[str, Union[str, List[str]]]:
        args: Dict[str, Union[str, List[str]]] = {}
        for arguments, arg_type in cls.__iterate_arguments(expr):
            if type_hierarchy:
                if arg_type not in args:
                    args[arg_type] = []
                args[arg_type].extend(arguments)
            else:
                args.update((name, arg_type) for name in arguments)
        return args
