import numpy as np

class Environment:
    # creating a grid with 5 x 5 cells
    def __init__(self, size=5):
        self.grid = np.empty(shape=(size,size), dtype=object)
        self.size = size
    
    def place_cell(self, x, y, cell):
        self.grid[x][y] = cell

