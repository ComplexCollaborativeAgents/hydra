

class ConsistencyChecker():

    def __init__(self):
        self.novelty_likelihood = 0
        self.unknowns = []

    def is_consistent(self,state):
        '''This should take a sequence of states perhaps with actions interleaved'''
        # if there is an unknown object, then novelty_likelihood is set to 1.
        self.unknowns = []
        for obj in state.objects.values():
            if obj['type'] == 'unknown':
                self.novelty_likelihood = 1
                self.unknowns.append(obj)
        return True