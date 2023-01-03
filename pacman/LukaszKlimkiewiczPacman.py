import random
from typing import Dict

from .Pacman import Pacman
from .Direction import Direction


class LukaszKlimkiewiczPacman(Pacman):

    def make_move(self, game_state, invalid_move=False) -> Direction:
        return random.choice(list(Direction))  # it will make some valid move at some point

    def give_points(self, points):
        pass

    def on_win(self, result: Dict["Pacman", int]):
        pass

    def on_death(self):
        pass
