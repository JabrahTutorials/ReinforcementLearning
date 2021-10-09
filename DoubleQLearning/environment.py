import numpy as np

class Environment(object):
    def __init__(self) -> None:

        super().__init__()
        self.start_state = 'A'

        self.terminal_state = 'T'

        self.states = ['A', 'B', 'T']

        self.b_actions = list(range(1, 10))

        self.possible_actions = {
            'A': ['left', 'right'],
            'B': self.b_actions,
            'T': []
        }

        self.state_transitions = {('B', i): 'T' for i in self.b_actions}
        self.state_transitions[('A', 'left')] = 'B'
        self.state_transitions[('A', 'right')] = 'T'


    def reward(self, state, action):
        if (state == 'B'):
            mu, sigma = -0.1, 1 
            return np.random.normal(mu, sigma, 1)[0]
        return 0
    
    def step(self, state, action):
        state = state
        reward = self.reward(state, action)
        next_state = self.state_transitions[(state, action)]
        done = (next_state == self.terminal_state)
        return reward, next_state, done
