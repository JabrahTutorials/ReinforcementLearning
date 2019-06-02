class Agent:
    def __init__(self, name, pos):
        self.__name = name
        self.__pos = pos

    def get_name(self):
        return self.__name

    def get_pos(self):
        return self.__pos
    
    def set_pos(self, new_pos):
        self.__pos = new_pos
