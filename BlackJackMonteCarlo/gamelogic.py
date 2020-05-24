from player import Player
from cards import cards

# We assume cards are drawn from an inifinite set with replacement

class BlackJack(object):
    def __init__(self, dealer, player):
        self.dealer = dealer
        self.player = player
        self.dealer_fc_total = dealer.first_card_total()
        self.player_total = player.total()
        self.player_init_state = [self.player_total, self.dealer_fc_total]
        self.bust_thresh = 21
    
    def start(self):
        print("Current Player Total = ", self.player_total)
        print("Current Dealer showing total =", self.dealer_fc_total)

        #Player turn
        total = 0
        while True:
            if self.player.total() > self.bust_thresh:
                print("BUST")
                break
            turn = input("h or s: ")
            if turn == 'h':
                self.player.hit()
                print("Current Player Total = ", self.player.total())
            elif turn == 's':
                total = self.player.stick()
                break
        return total




dealer = Player(cards)
player = Player(cards)

b = BlackJack(dealer, player)
b.start()