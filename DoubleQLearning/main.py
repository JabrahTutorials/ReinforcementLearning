import matplotlib.pyplot as plt
import seaborn as sns

from environment import Environment
from qlearning import QLearningAgent
from double_qlearning import DoubleQLearningAgent

sns.set()


env = Environment()
agent = QLearningAgent(env)
agent2 = DoubleQLearningAgent(env)

left_actions_ratio_a1 = agent.update_policy()
left_actions_ratio_a2 = agent2.update_policy()


fig, ax = plt.subplots()
ax.plot(range(len(left_actions_ratio_a1)), left_actions_ratio_a1, color="red", label="Q-Learning")
ax.plot(range(len(left_actions_ratio_a2)), left_actions_ratio_a2, color="green", label="Double Q-Learning")
ax.plot(range(len(left_actions_ratio_a1)), [5]*len(left_actions_ratio_a1), '--', color='black', label='optimal')
ax.set_xlabel("Number of episodes")
ax.set_ylabel("% of left actions from A")
ax.legend(loc='best')
plt.show()
