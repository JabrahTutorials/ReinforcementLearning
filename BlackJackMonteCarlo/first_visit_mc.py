# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from black_jack_sampler import BlackJackSampler

# For solving the prediction problem
class FirstVisitMC(object):
    def __init__(self):
        self.values = {}
        self.episodes = 100
    
    def policy(self, state):
        if state[0] >= 20:
            return 1  # stick
        else:
            return 0

    def run(self):
        
        black_jack = BlackJackSampler()
        
        for i in range(100000):
            episode = black_jack.generate_episode(self.policy)
            G = 0
            for i in range(len(episode)-1, 0, -3):
                reward = episode[i]
                action = episode[i-1]
                state = episode[i-2]

                G = G+reward

                if state in episode[:i-2]:
                    # it is not our first visir to this state
                    continue
                
                if state in self.values:
                    self.values[state].append(G)
                else:
                    self.values[state] = [G]
        
        self.values = { key: sum(self.values[key])/len(self.values[key]) for key in self.values}
        return self.values
    
    def plot_value_function(self):
        x = np.arange(1, 22)
        y = np.arange(1, 10)

        x, y = np.meshgrid(x, y)

        z = []

        for (row_ind, i) in enumerate(x):
            temp = []
            for (col_ind, j) in enumerate(x[row_ind]):
                x_val = j
                y_val = y[row_ind, col_ind]
                if (x_val, y_val) in self.values:
                    temp.append(self.values[(x_val, y_val)])
                else:
                    temp.append(0)
            z.append(temp)
        z = np.array(z)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # Plot the surface.
        surf = ax.plot_surface(y, x, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        
        ax.set_xlabel("Dealer's showing card")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")

        # Customize the z axis.
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()



fv_mc = FirstVisitMC()
fv_mc.run()
fv_mc.plot_value_function()
