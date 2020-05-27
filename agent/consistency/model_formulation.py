from agent.consistency.novelty_classification import initize_novelty_detector

class ConsistencyChecker():

    def __init__(self):
        self.novelty_likelihood = 0
        self.unknowns = []
        self.unknown_history = [0,0,0]
        self.novelty_model = initize_novelty_detector()
        self.new_level = True

    def is_consistent(self,state):
        '''This should take a sequence of states perhaps with actions interleaved'''
        # if there is an unknown object, then novelty_likelihood is set to 1.
        self.unknowns = []
        for obj in state.objects.values():
            if obj['type'] == 'unknown':
                self.unknowns.append(obj)
        if self.new_level:
            self.unknown_history.insert(0, len(self.unknowns))
            self.novelty_likelihood = self.novelty_model.predict_proba(
                [self.unknown_history[:3]])[0][1]
            self.new_level = False
        return True