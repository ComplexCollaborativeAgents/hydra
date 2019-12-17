from utils.state import InvokeBasicRL
# this will likely just be calling an executable

class Planner():
    domain_file = None
    problem = None # current state of the world

    def __init__(self):
        pass

    def make_plan(self,state):
        '''
        The plan should be a list of actions that are either executable in the environment
        or invoking the RL agent
        '''
        return [InvokeBasicRL(state)]

    def execute(self,plan,policy_learner):
        '''Converts the symbolic action into an environment action'''
        if isinstance(plan[0],InvokeBasicRL):
            return policy_learner.act_and_learn(plan[0].state)
