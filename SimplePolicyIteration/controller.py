import pygame

class Controller:
    keys = None

    def set_key_map(key_map):
        Controller.keys = key_map

    def left():
        return Controller.keys[pygame.K_LEFT]

    def right():
        return Controller.keys[pygame.K_RIGHT]

    def up():
        return Controller.keys[pygame.K_UP]

    def down():
        return Controller.keys[pygame.K_DOWN]
