import random
import pygame
from config import Config
from cards import cards
from player import Player
from environment import Environment
from controllers.manual_controller import ManualController

class Game(object):
    def __init__(self, config, env, controller):
        pygame.init()
        self.win_ = pygame.display.set_mode((config.screen_width,
                                            config.screen_height))
        pygame.display.set_caption("Jabrah")

        self.config_ = config
        self.env_ = env
        self.controller_ = controller

        self.dealer_wins = -2
    

    def draw_env(self):
        self.win_.fill((0, 128, 0))
        init_offset = 10
        text_font_size = 50
        root_dir = "cards_set/img/"
        img_ext = '.png'

        font1 = pygame.font.SysFont(None, text_font_size)
        font2 = pygame.font.SysFont(None, text_font_size)
        dealer_text = font1.render("Dealer's Cards", True, (255, 255, 255))
        player_text = font1.render("Player's Cards", True, (255, 255, 255))
        self.win_.blit(dealer_text, (init_offset, init_offset))
        self.win_.blit(player_text, (
            init_offset, self.config_.screen_height - text_font_size - init_offset))
        
        
        player_cards, dealer_cards = self.env_.selected_cards_state()
        
        player_cards = [c+s+img_ext for c, s in player_cards]
        dealer_cards = [c+s+img_ext for c, s in dealer_cards]
        

        if len(dealer_cards) <= 1:
            dealer_cards = ['jb_card.png'] + dealer_cards

        for (i, card_image) in enumerate(dealer_cards):
            card_obj = pygame.image.load(root_dir+card_image)
            x_coord = (i+1) * init_offset + (i * self.config_.card_width)
            y_coord = 2*init_offset + text_font_size
            self.win_.blit(card_obj, (x_coord, y_coord))

        for (i, card_image) in enumerate(player_cards):
            card_obj = pygame.image.load(root_dir+card_image)
            x_coord = (i+1) * init_offset + (i * self.config_.card_width )
            y_coord = self.config_.screen_height - self.config_.card_height - text_font_size - 3*init_offset
            self.win_.blit(card_obj, (x_coord, y_coord))
        
        # update scores
        score_text = "Score: "

        dealer_score = str(self.env_.dealer.first_card_total()) if not self.env_.dealer.stick_\
            else str(self.env_.dealer.total())
        player_score = str(self.env_.player.total())

        dealer_score_text = font2.render(score_text + dealer_score, True, (255, 255, 0))
        player_score_text = font2.render(score_text + player_score, True, (255, 255, 0))
        self.win_.blit(dealer_score_text, (self.config_.screen_width - 250, init_offset))
        self.win_.blit(
            player_score_text, (
                self.config_.screen_width - 250,
                self.config_.screen_height - text_font_size - init_offset))
        
        if self.dealer_wins == -1:
            win_text = font2.render("DRAW", True, (255, 0, 0))
            self.win_.blit(win_text, (self.config_.screen_width//2, self.config_.screen_height//2))
        elif self.dealer_wins == 0:
            win_text = font2.render("WIN", True, (0, 0, 128))
            self.win_.blit(win_text, (self.config_.screen_width//2, self.config_.screen_height//2))
        elif self.dealer_wins == 1:
            win_text = font2.render("LOST", True, (128, 0, 0))
            self.win_.blit(win_text, (self.config_.screen_width//2, self.config_.screen_height//2))

    def start(self):
        run = True
        state = None
        reward = 0
        done = False

        while (run):
            pygame.time.delay(self.config_.delay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Game Logic
            keys = pygame.key.get_pressed()
            self.controller_.set_key_map(keys)

            if self.controller_.hit():
                state, reward, done = self.env_.step(0)
            elif self.controller_.stick():
                state, reward, done = self.env_.step(1)

            if done:
                if reward == 1:
                    self.dealer_wins = 0
                elif reward == -1:
                    self.dealer_wins = 1
                else:
                    self.dealer_wins = -1
    
            self.draw_env()
            pygame.display.update()
        pygame.quit()

player = Player(cards)
dealer = Player(cards)
env = Environment(player, dealer)

game = Game(Config, env, ManualController)
game.draw_env()
pygame.display.update()
game.start()