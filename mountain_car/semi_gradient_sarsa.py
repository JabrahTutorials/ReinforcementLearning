import pickle
import random
import numpy as np
import gymnasium as gym

random.seed(18)

from tile_coding import *


class SemiGradientSarsa(object):
    def __init__(
        self,
        env,
        alpha=0.01, 
        eps=0.1,
        gamma=1,
        n_tilings = 7,
        num_eps=100) -> None:
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.num_eps = num_eps
        self.n_tilings = n_tilings
        self.env = env
        
        # define weight vector
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(n_tilings**4,))

        # hash for tile coding
        self.tile_coding = IHT(n_tilings**4)

    def q_func(self, feature_vector):
        return np.dot(self.w, feature_vector)

    def update_weight(self, reward, current_q, future_q, feature_vector, terminal):
        if terminal:
            w_update = self.alpha * (reward - current_q)
        else:
            w_update = self.alpha * (reward + self.gamma *future_q - current_q)
        self.w += np.multiply(w_update, feature_vector)


    def train(self):
        state, info = self.env.reset()
        action, q = self.select_action(state)
        episodes = 0
        steps = 0

        total_reward = 0

        while episodes < self.num_eps:
            steps += 1

            feature_vec = self.hash_feature_vector(state, action)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated:

                if episodes % 50 == 0:
                    print("episode:", episodes, 'completed', 'reward:', total_reward)
                
                self.update_weight(reward, q, None, feature_vec, True)
                state, info = self.env.reset()
                action, q = self.select_action(state)
                total_reward = 0
                steps = 0
                episodes += 1

                continue

            next_action, next_q = self.select_action(next_state)
            self.update_weight(reward, q, next_q, feature_vec, False)
            state = next_state
            action = next_action
            q = next_q
            
        self.save_params()
    
    def save_params(self):
        print(self.w)
        pickle.dump(self.w, open('weights.pkl', 'wb'))
        pickle.dump(self.tile_coding, open('tilings.pkl', 'wb'))
    
    def load_params(self):
        self.w = pickle.load(open('weights.pkl', 'rb'))
        self.tile_coding = pickle.load(open('tilings.pkl', 'rb'))

    def one_hot_encode(self, indices):
        size = len(self.w)
        one_hot_vec = np.zeros(size)
        for i in indices:
            one_hot_vec[i] = 1
        return one_hot_vec

    def hash_feature_vector(self, state, action):
        # speed you up
        feature_ind = np.array(tiles(self.tile_coding, self.n_tilings, state.tolist(), [action]))
        feature_vec = self.one_hot_encode(feature_ind)
        return feature_vec

    def select_action(self, state, eps_greedy = True):
        num_actions = self.env.action_space.n
        actions = range(num_actions)
        action_val_dict = {}
        for action in actions:
            feature_vector = self.hash_feature_vector(state, action)
            q_val = self.q_func(np.array(feature_vector))

            action_val_dict[action] = q_val
        
        greedy_action = max(action_val_dict, key=action_val_dict.get)
        
        if not eps_greedy:
            return greedy_action

        non_greedy_actions = list(set(range(num_actions)) - {greedy_action})
        
        prob_explorative_action = self.eps / num_actions
        prob_greedy_action = 1 - self.eps + prob_explorative_action

        action = np.random.choice([greedy_action] + non_greedy_actions,
                    p=[prob_greedy_action]+[prob_explorative_action for _ in range(len(non_greedy_actions))])
        return action, action_val_dict[action]

