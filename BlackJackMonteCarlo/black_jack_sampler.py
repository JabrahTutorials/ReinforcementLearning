from cards import cards
from player import Player
from environment import Environment

# We assume cards are drawn from an inifinite set with replacement

class BlackJackSampler(object):
    def __init__(self):
        pass

    def generate_episode(self, policy_func):
        dealer = Player(cards)
        player = Player(cards)
        env = Environment(player, dealer)

        state = env.state()
        episode_trace = [state]
        while True:
            action = policy_func(state)
            episode_trace.append(action)
            state, reward, done = env.step(action)
            episode_trace.append(reward)

            if done:
                break
            episode_trace.append(state)
        return episode_trace





