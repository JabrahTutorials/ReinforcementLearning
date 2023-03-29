import time
import random
import numpy as np
import gymnasium as gym

random.seed(10)

from semi_gradient_sarsa import SemiGradientSarsa

env_ = gym.make("MountainCar-v0", render_mode=None)

sarsa = SemiGradientSarsa(env_)
sarsa.load_params()

env = gym.make("MountainCar-v0", render_mode='human')
state, info = env.reset()

for i in range(200):
    action = sarsa.select_action(state, eps_greedy=False)
    next_state, reward, terminated, truncated, info = env.step(action)

    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)

    state = next_state

env.close()

