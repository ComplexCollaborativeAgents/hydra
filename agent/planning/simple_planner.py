from agent.planning.sb_planner import *
from agent.consistency.pddl_plus_simulator import *
import random

''' A simple, inefficient planner that uses the simulator to choose which action to do '''
class SimpleSBPlanner(SBPlanner):

    def __init__(self, meta_model : ScienceBirdsMetaModel = ScienceBirdsMetaModel(), default_delta_t=None):
        super().__init__(meta_model)
        self.simulator = PddlPlusSimulator()
        self.default_delta_t = default_delta_t

    ''' @Override superclass '''
    def make_plan(self, state: ProcessedSBState, prob_complexity=0, delta_t = None):
        if delta_t is None:
            if self.default_delta_t is not None:
                delta_t = self.default_delta_t
            else:
                delta_t = 1/self.meta_model.constant_numeric_fluents["angle_rate"]

        pddl_problem = self.meta_model.create_pddl_problem(state)
        pddl_domain = self.meta_model.create_pddl_domain(state)

        max_twang_time  = 100
        init_state = PddlPlusState(pddl_problem.init)
        active_bird_id = int(init_state[("active_bird",)])
        active_bird = init_state.get_bird(active_bird_id)

        grounded_domain  = PddlPlusGrounder().ground_domain(pddl_domain, pddl_problem)
        action_name = "pa-twang %s" % active_bird
        for action in grounded_domain.actions:
            action_time = 1
            while action_time <= max_twang_time:
                timed_action = TimedAction(action, action_time)
                plan = [timed_action]

                current_state, t, trace = self.simulator.simulate(plan, pddl_problem, pddl_domain, 0.05)

                if self._killed_pigs(current_state)>0:
                    return [TimedAction(action_time, action_time)]
                else:
                    action_time = action_time+delta_t

        action = grounded_domain.actions[0]

        return [[action.name, 80]]

    ''' Count the number of killed pigs '''
    def _killed_pigs(self, pddl_state : PddlPlusState):
        pigs = pddl_state.get_pigs()
        killed = 0
        for pig in pigs:
            if pddl_state[('pig_dead', pig)]==True:
                killed = killed+1
        return killed

