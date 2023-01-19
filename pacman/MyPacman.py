import random
import dataclasses
import numpy as np
from typing import Dict
from pathlib import Path

from .Pacman import Pacman
from .Direction import Direction
from .Helpers import can_move_in_direction, direction_to_new_position


class MyPacman(Pacman):

    WEIGHTS = np.array([])

    def __init__(self, train=False, alpha=0.001, epsilon=0.25, discount=0.6, filename='weights.txt'):
        """
        DEFAULT PARAMETERS ARE CORRECT FOR EVALUATING AGENT

        params:
        * train - whether agent should learn or not
        * alpa - determines speed of learning (only when train is true)
        * epsilon - specifies balance between exploration and exploitation (only when train is true)
        * discount - trust to environment (only when train is true)
        * filename - filename to load and save weights (only when train is true)
        """

        super().__init__()
        self.train = train
        self.alpha = alpha if train else 0
        self.epsilon = epsilon if train else 0
        self.discount = discount if train else 0
        self.filename = filename

        self.__weights = self.__load_weights()
        self.__game_states_history = []
        self.__actions_history = []

    # ------------------------ properties --------------------------

    @property
    def last_game_state(self):
        if len(self.__game_states_history) > 0:
            return self.__game_states_history[-1]
        else:
            return None

    @property
    def prev_game_state(self):
        if len(self.__game_states_history) > 1:
            return self.__game_states_history[-2]
        else:
            return None

    @property
    def last_action(self):
        if len(self.__actions_history) > 0:
            return self.__actions_history[-1]
        else:
            return None

    # ------------------------ pacman-api --------------------------

    def make_move(self, game_state, invalid_move=False) -> Direction:
        should_random = random.random() < self.epsilon

        if invalid_move:
            assert False
            action = random.choice(list(Direction))
        elif should_random:
            action = random.choice(self.__get_legal_actions(game_state))
        else:
            action = self.__get_best_action(game_state)

        self.__game_states_history.append(game_state)
        self.__actions_history.append(action)

        return action

    def give_points(self, points):
        self.__update(reward=points)

    def on_win(self, result: Dict["Pacman", int]):
        self.__on_finish()

    def on_death(self):
        self.__update(reward=-50)
        self.__on_finish()

    # ---------------- value-function-approximation-stuff ----------------

    def __on_finish(self):
        if self.train:
            self.__save_weights()

    def __load_weights(self):
        path = Path(self.filename)
        if not path.exists():
            return None
        else:
            return np.loadtxt(self.filename)

    def __save_weights(self):
        np.savetxt(self.filename, self.__weights)

    def __get_reward_for_win(self, result):
        my_score = result[self]
        op_scores = [result[player] for player in result.keys() if player != self]
        best_op_score = max(op_scores)
        return my_score - best_op_score

    def __update(self, reward):
        state = self.prev_game_state
        next_state = self.last_game_state
        action = self.last_action

        if state is None or next_state is None:
            return

        if not self.train:
            return

        features = self.__get_features(state, action)

        if self.__weights is None:
            self.__weights = np.zeros((len(features),))

        delta = (reward + self.discount * self.__get_value(next_state)) - self.__get_qvalue(state, action)
        self.__weights += self.alpha * delta * features

    def __get_best_action(self, game_state):
        legal_actions = self.__get_legal_actions(game_state)
        qvalues = [self.__get_qvalue(game_state, action) for action in legal_actions]
        best_qvalue = max(qvalues)
        best_actions = [action for action, qvalue in zip(legal_actions, qvalues) if qvalue == best_qvalue]
        best_action = random.choice(best_actions)
        return best_action

    def __get_legal_actions(self, game_state):
        return [direction for direction in Direction if self.__is_legal_action(game_state, direction)]

    def __is_legal_action(self, game_state, action):
        position = game_state.you['position']
        phasing = game_state.you['is_phasing']
        return can_move_in_direction(position, action, game_state.walls, game_state.board_size, phasing)

    def __get_value(self, game_state):
        possible_actions = self.__get_legal_actions(game_state)

        if len(possible_actions) == 0:
            return 0.0

        return max([self.__get_qvalue(game_state, action) for action in possible_actions])

    def __get_qvalue(self, game_state, action):
        features = self.__get_features(game_state, action)
        if self.__weights is None:
            self.__weights = np.random.random((len(features),))
        return self.__weights @ features

    # ------------------------- features ---------------------------

    def __get_features(self, game_state, action):
        next_game_state = self.__get_state_after_action(game_state, action)

        nearest_ghost_distance = self.__get_feature_nearest_ghost_distance(next_game_state)
        double_point_distance = self.__get_feature_double_point_distance(next_game_state)
        big_points_distance = self.__get_feature_big_points_distance(next_game_state)
        big_big_point_distance = self.__get_feature_big_big_points_distance(next_game_state)
        indestructible_distance = self.__get_feature_indestructible_distance(next_game_state)
        points = self.__get_feature_points(next_game_state)
        # connected_points = self.__get_feature_connected_points(game_state, action)

        return np.array([
            nearest_ghost_distance,
            # double_point_distance,
            # big_points_distance,
            # big_big_point_distance,
            # indestructible_distance,
            points,
            # connected_points,
        ])

    def __get_feature_nearest_ghost_distance(self, game_state):
        is_indestructible = self.__is_timer_enabled(game_state.you['is_indestructible'])
        if is_indestructible:
            return 0

        ghost_positions = [ghost_info['position'] for ghost_info in game_state.ghosts]
        distance = self.__get_distance_to_nearest(game_state.you['position'], ghost_positions)
        max_distance = 20
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_double_point_distance(self, game_state):
        is_active = self.__is_timer_enabled(game_state.you['double_points_timer'])
        if is_active:
            return 1.0

        distance = self.__get_distance_to_nearest(game_state.you['position'], game_state.double_points)
        if distance is None:
            return 0

        max_distance = 15
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_indestructible_distance(self, game_state):
        is_active = self.__is_timer_enabled(game_state.you['is_indestructible'])
        if is_active:
            return 1.0

        distance = self.__get_distance_to_nearest(game_state.you['position'], game_state.indestructible_points)
        if distance is None:
            return 0

        max_distance = 15
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_big_points_distance(self, game_state):
        distance = self.__get_distance_to_nearest(game_state.you['position'], game_state.big_points)
        if distance is None:
            return 0

        max_distance = 15
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_big_big_points_distance(self, game_state):
        distance = self.__get_distance_to_nearest(game_state.you['position'], game_state.big_big_points)
        if distance is None:
            return 0

        max_distance = 15
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_points(self, game_state):
        distance = self.__get_distance_to_nearest(game_state.you['position'], game_state.points)
        if distance is None:
            return 0

        max_distance = 4
        norm_distance = min(max_distance, distance) / max_distance
        rev_distance = 1 - norm_distance
        return rev_distance

    def __get_feature_connected_points(self, game_state):
        stuff = set()
        stuff.update(game_state.points)

    # -------------------- utility-functions -----------------------

    def __get_state_after_action(self, game_state, action):
        next_state = self.__copy_game_state(game_state)
        next_state.you['position'] = direction_to_new_position(next_state.you['position'], action)
        return next_state

    def __copy_game_state(self, game_state):
        new_you = dict(game_state.you)
        new_game_state = dataclasses.replace(game_state, you=new_you)
        return new_game_state

    def __get_distance_to_nearest(self, start_point, end_points):
        if len(end_points) == 0:
            return None
        distances = []
        for end_point in end_points:
            distance = self.__get_distance(start_point, end_point)
            distances.append(distance)
        return min(distances)

    def __get_distance(self, start, end):
        return abs(start.x - end.x) + abs(start.y - end.y)

    def __is_timer_enabled(self, timer, min_bound=4):
        return timer is not None and timer > min_bound