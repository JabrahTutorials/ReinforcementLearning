import gymnasium as gym
from semi_gradient_sarsa import SemiGradientSarsa

# add `.env` at the end to ignore internal truncation
env = gym.make("MountainCar-v0", render_mode=None)

sarsa = SemiGradientSarsa(env, num_eps=500, alpha=0.01)
sarsa.train()