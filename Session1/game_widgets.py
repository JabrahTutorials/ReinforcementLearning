class CellType:
    BLANK = 1  # no reward
    WHOOPING = 2 # reward = -10
    KFC = 3 # reward = 10

class CellState:
    # initialize a cell for grid 
    def __init__(self, pos, reward, cell_type, is_terminal):
        self.__pos = pos
        self.__reward = reward
        self.__cell_type = cell_type
        self.__is_terminal = is_terminal
        self.children = []

    def append_child(self, cell):
        self.children.append(cell)

    def pos(self):
        return self.__pos

    def reward(self):
        return self.__reward

    def cell_type(self):
        return self.__cell_type

    def is_terminal(self):
        return self.__is_terminal