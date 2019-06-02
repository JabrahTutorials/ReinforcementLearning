import pygame
from agent import Agent
from config import Config
from controller import Controller
from game_widgets import CellType, CellState
from environment import Environment
from policy_iteration import PolicyIteration 

class Game:
    def __init__(self, config, controller, env, agent, policy=None):
        # initialize pygame module
        pygame.init()

        # setup config parameters for game window
        self.controller_ = controller
        self.config_ = config
        self.win_ = pygame.display.set_mode((config.screen_size,
                                            config.screen_size))
        
        # set game env
        self.env_ = env

        # set agent
        self.agent = agent

        # set policy
        self.policy = policy

        pygame.display.set_caption("Jabrah")

    def draw_env(self):
        self.win_.fill((0,0,0))
        # load images
        kfc_image = pygame.image.load("images/kfc_r.png")
        grass_image = pygame.image.load("images/grass_r.png")
        whoop_image = pygame.image.load("images/beat_r.png")

        # draw environments
        for i in range(self.env_.size):
            for j in range(self.env_.size):
                cell = self.env_.grid[i][j]

                (x, y) = cell.pos()
                width = self.config_.cell_width
                height = self.config_.cell_height

                x = x * width
                y = y * height

                if (i,j) == agent.get_pos():
                    agent_image = pygame.image.load("images/baby_r.png")
                    self.win_.blit(agent_image, (x, y))

                elif cell.cell_type() == CellType.KFC:
                    self.win_.blit(kfc_image, (x, y))

                elif cell.cell_type() == CellType.WHOOPING:
                    self.win_.blit(whoop_image, (x, y))

                else:
                    self.win_.blit(grass_image, (x, y))


    def start(self):
        run = True
        while (run):
            pygame.time.delay(self.config_.delay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
            keys = pygame.key.get_pressed()

            self.controller_.set_key_map(keys)
            curr_pos = self.agent.get_pos()
            new_pos = curr_pos

            if self.policy is None:
                # if no policy, use controller
                if self.controller_.right():
                    new_pos = (min(curr_pos[0]+1, env.size-1), curr_pos[1])

                elif self.controller_.left():
                    new_pos = (max(curr_pos[0]-1, 0), curr_pos[1])
                
                elif self.controller_.down():
                    new_pos = (curr_pos[0], min(curr_pos[1]+1, env.size-1))

                elif self.controller_.up():
                    new_pos = (curr_pos[0], max(curr_pos[1]-1, 0))
            else:
                if self.policy[curr_pos] == 1:
                    new_pos = (min(curr_pos[0]+1, env.size-1), curr_pos[1])

                elif self.policy[curr_pos] == 3:
                    new_pos = (max(curr_pos[0]-1, 0), curr_pos[1])
                
                elif self.policy[curr_pos] == 2:
                    new_pos = (curr_pos[0], min(curr_pos[1]+1, env.size-1))

                elif self.policy[curr_pos] == 0:
                    new_pos = (curr_pos[0], max(curr_pos[1]-1, 0))

            self.agent.set_pos(new_pos)
            self.draw_env()
            pygame.display.update()

        pygame.quit()

def create_game_env():
    env = Environment(size=5)
    states = []
    policy = {}

    # the elements in the matrix represent the cell type
    cell_matrix = [[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 2],
                   [2, 1, 1, 1, 1],
                   [2, 2, 1, 1, 1],
                   [1, 2, 1, 1, 3]]

    size = len(cell_matrix)
    env = Environment(size=size)

    reward = 0
    is_terminal = False
    for i in range(size):
        for j in range(size):
            cell_type = cell_matrix[j][i]

            if cell_type == CellType.WHOOPING:
                is_terminal = True
                reward = -10
            elif cell_type == CellType.KFC:
                is_terminal = True
                reward = 10
            else:
                is_terminal = False
                reward = -1
            cell = CellState((i,j), reward, cell_type, is_terminal)
            env.place_cell(i, j, cell)
            states.append(cell)
    return env, states


env, states = create_game_env()
agent = Agent("policy_eval", (0, 0))
policy_iter_algo = PolicyIteration(states)
policy = policy_iter_algo.run()
game = Game(Config, Controller, env, agent, policy)
# initiate env
game.draw_env()
pygame.display.update()
game.start()
