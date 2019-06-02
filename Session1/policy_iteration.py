import random
import numpy as np
random.seed(0)

class PolicyIteration(object):
    def __init__(self, cell_states):
        # 0 - up, 1 - right, 2 - down, 3 - left
        self.eps = 3
        self.actions = [0, 1, 2, 3]
        self.state_dict, self.policy_dict, self.value_dict = \
                        self.create_state_policy_dict(cell_states)

    def create_state_policy_dict(self, cell_states):
        state_dict = {}
        policy_dict = {}
        value_dict = {}
        for cell_state in cell_states:
            state_dict[cell_state.pos()] = cell_state
            policy_dict[cell_state.pos()] = random.choice(self.actions)
            if cell_state.is_terminal():
                value_dict[cell_state.pos()] = cell_state.reward()
            else:
                value_dict[cell_state.pos()] = 0
        return state_dict, policy_dict, value_dict

    def get_future_state(self, curr_pos, action):

        new_pos = curr_pos
        env_size = 5
        if action == 1:
            new_pos = (min(curr_pos[0]+1, env_size-1), curr_pos[1])

        elif action == 3:
            new_pos = (max(curr_pos[0]-1, 0), curr_pos[1])
        
        elif action == 2:
            new_pos = (curr_pos[0], min(curr_pos[1]+1, env_size-1))

        elif action == 0:
            new_pos = (curr_pos[0], max(curr_pos[1]-1, 0))
        return new_pos

    def q_value(self, state, action):
        state_ = self.get_future_state(state, action)
        q = self.state_dict[state_].reward() + \
            self.value_dict[state_]
        return q
    
    def policy_evaluation(self):
        print("evaluating policy...")
        while True:
            delta = 0
            for state in self.state_dict:
                v = self.value_dict[state]

                # if terminal state, dont update value
                if self.state_dict[state].is_terminal():
                    continue
                # next state (s') is what
                self.value_dict[state] = self.q_value(state, self.policy_dict[state])
                delta = max(delta, abs(v-self.value_dict[state]))
            if delta < self.eps:
                break

    def policy_improvement(self):
        print("improving policy...")
        policy_stable = True
        for state in self.state_dict:
            old_action = self.policy_dict[state]
            old_action_value = self.q_value(state, old_action)
            
            best_action = old_action
            best_action_value = old_action_value
            
            for action in self.actions:
                action_value = self.q_value(state, action)
                if action_value > old_action_value:
                    best_action = action
                    best_action_value = action_value
                    policy_stable = False
            self.policy_dict[state] = best_action
        return policy_stable
    
    def run(self):
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                return self.policy_dict
            





