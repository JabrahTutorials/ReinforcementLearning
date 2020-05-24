import random

class Player(object):
    def __init__(self, cards):
        self.cards = cards
        self.chosen_cards = [self._pick_card(), self._pick_card()]
        self.stick_ = False

    def _pick_card(self):
        card_suits = ['C', 'D', 'H', 'S']

        return random.choice(list(self.cards.keys())), random.choice(card_suits)
    
    def hit(self):
        # Request for additional cards
        card_choice = self._pick_card()

        card_value = self.cards[card_choice[0]]

        self.chosen_cards.append(card_choice)

    def stick(self):
        self.stick_ = True
    
    def total(self):
        # Stop requesting for additional cards
        total = 0
        for (chosen_card, _) in self.chosen_cards:
            total += self.cards[chosen_card]
        return total
    
    def first_card_total(self):
        # If the player is a dealer
        return self.cards[self.chosen_cards[1][0]]

