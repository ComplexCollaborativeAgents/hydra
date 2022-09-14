from agent.planning.polycraft_planning.polycraft_pddl_objects_and_constants import PddlGameMapCellType
from worlds.polycraft_world import PolycraftState


class Task:
    """ A task that the polycraft agent can aim to do """

    def __str__(self):
        return self.__class__.__name__

    def create_relevant_actions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def create_relevant_events(self, world_state: PolycraftState, meta_model):
        """ Returns a list of events for the agent to consider when planning"""
        raise NotImplementedError()

    def get_goals(self, world_state: PolycraftState, meta_model):
        """ Returns the goal to achieve """
        raise NotImplementedError()

    def get_metric(self, world_state: PolycraftState, meta_model):
        """ Returns the metric the planner seeks to optimize """
        raise NotImplementedError()

    def get_planner_heuristic(self, world_state: PolycraftState, metamodel):
        """ Returns the heuristic to be used by the planner"""
        raise NotImplementedError()

    def get_relevant_types(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_relevant_predicates(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_relevant_functions(self, world_state: PolycraftState, meta_model):
        """ Returns a list of actions for the agent to use when planning """
        raise NotImplementedError()

    def get_type_for_cell(self, cell_attr, meta_model) -> PddlGameMapCellType:
        """ Return a PddlGameMapCellType appropriate to generate objects representing this game map cell"""
        raise NotImplementedError()

    def is_done(self, state: PolycraftState) -> bool:
        """ Checks if the task has been succesfully completed """
        raise NotImplementedError()

    def is_feasible(self, state: PolycraftState) -> bool:
        """ Checks if the task can be achived in the current state """
        return True

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __hash__(self):
        return hash(self.__class__.__name__)
