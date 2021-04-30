
'''
M1: Given a set trials with novelty in them, how many does the system detect to be not-novel?
M2: Given a set of trials with novelty in them, how many does the system correctly detect to be novel?
M2.1: Given a set of trial without any novelty, how many does the system incorrectly detect to be novel?
M3: Given a set of trials (novelty and not-novelty), how many trials can the system solve? What does solving mean?

State of art AI player. Average scores for benchmark agents (before and after).

M4: If known that the trial is novel, how many trials can the system solve?


class NoveltyExperimentRunner:
    def __init__(self, environment, agent, novelty_set):
        self._env = environment
        self._agent = agent
        pass


    def run_unknown_novelty_experiment(self, novelty_set, samples): ### generate M1, M2, M3
        for i in range(0, samples):
            for novelty in novelty_set:
                for k in range (0, number_of_post_novelty_episode):
                    self._env.set_configuration(novelty)
                    self._agent.initialize() ### initialize the agent to a knowledge state every time a new novelty response is being measured.
                    is_novel, score = self.run(self._env, self._agent, is_novel=?) ### who runs the perceive, decide, act loop?
                    record_data

    def run_known_novelty_experiment(self, novelty_set, samples): ### generate M4
        for i in range(0, samples):
            for novelty in novelty_set:
                self._env.set_configuration(novelty)
                self._agent.initialize()  ### initialize the agent to a knowledge state every time a new novelty response is being measured.
                is_novel, score = self.run(self._env, self._agent, is_novel=True)  ### who runs the perceive, decide, act loop?
                record_data

    def run_no_novelty_experiment(self, samples): ### generate M2.1 and other baseline metrics
        for i in range(0, samples):
            self._env.reset_canonical()
            self._agent.initialize()
            is_novel, score = self.run(self._env, self._agent, is_novel=?)
            record_data


    def run(self):
        is_novel = False
        score = 0

        ### do until success or failure
            ### provide agent with observations
            ### ask the agent to give an action
            ### apply the action to the enivronment and advance to the next timestep and observation

        return is_novel, score

'''

