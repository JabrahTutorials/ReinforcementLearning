
class Environment(object):
    def __init__(self, player, dealer):
        self.player = player
        self.dealer = dealer  # The dealer is actually part of the env
        self.dealer_thresh = 17
        self.bust_thresh = 21

    def state(self):
        return self.player.total(), self.dealer.first_card_total()

    def selected_cards_state(self):
        # This is just to help with the visualization in pygame
        if not self.dealer.stick_:
            return self.player.chosen_cards, [self.dealer.chosen_cards[1]]
        return self.player.chosen_cards, self.dealer.chosen_cards
    
    def step(self, action):
        # Map 0 - hit, 1 - stick
        if action == 0:
            self.player.hit()
        elif action == 1:
            self.player.stick()

        state = self.player.total(), self.dealer.first_card_total()
        # player is done when bust or after he sticks
        player_done = (self.player.total() > self.bust_thresh) or (action == 1)
        reward = self._reward(state, player_done)
        return state, reward, player_done

    def _reward(self, state, player_done):
        if not player_done:
            return 0
        while self.dealer.total() < self.dealer_thresh:
            self.dealer.hit()
        self.dealer.stick()

        player_bust = self.player.total() > self.bust_thresh
        dealer_bust = self.dealer.total() > self.bust_thresh
        player_scored_higher = self.player.total() > self.dealer.total()
        dealer_scored_higher = self.player.total() < self.dealer.total()

        # scoring when any of the players bust
        if player_bust and dealer_bust:
            return 0
        elif player_bust and (not dealer_bust):
            return -1
        elif (not player_bust) and dealer_bust:
            return 1
        
        # scoring based on scores
        if player_scored_higher:
            return 1
        elif dealer_scored_higher:
            return -1
        else:
            return 0
        
        
    
    

    