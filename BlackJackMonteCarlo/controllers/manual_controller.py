import pygame

class ManualController:
    keys = None

    def set_key_map(key_map):
        ManualController.keys = key_map

    def stick():
        return ManualController.keys[pygame.K_LEFT]

    def hit():
        return ManualController.keys[pygame.K_RIGHT]

