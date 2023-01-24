from .Helpers import can_move_in_direction, find_path, positions_to_direction, direction_to_new_position
from .Direction import Direction
from .Position import Position, clamp
from random import choice, random


def rotate_left(direction):
    return {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP}[direction]

def rotate_right(direction):
    return {Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN, Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP}[direction]

def default_strategy_eatable(my_position, my_direction, walls, pacman_positions, board_size, changed):
    if changed:
        return ~my_direction
    if can_move_in_direction(my_position, my_direction, walls, board_size) \
            and not (can_move_in_direction(my_position, rotate_left(my_direction), walls, board_size) or
                     can_move_in_direction(my_position, rotate_right(my_direction), walls, board_size)):
        return my_direction
    else:
        possible_directions = []
        for direction in Direction:
            if can_move_in_direction(my_position, direction, walls, board_size):
                possible_directions.append(direction)
        return choice(possible_directions)


def get_distance(pacman, my_position):
    return abs(pacman.x - my_position.x) + abs(pacman.y - my_position.y)


def strategy_normal_factory(relative_to_pacman: Position):
    # this strategy will look for the closest Pacman and target them
    def strategy(my_position, my_direction, walls, pacman_positions, board_size, changed):
        if can_move_in_direction(my_position, my_direction, walls, board_size) \
            and not (can_move_in_direction(my_position, rotate_left(my_direction), walls, board_size) or
                     can_move_in_direction(my_position, rotate_right(my_direction), walls, board_size)):
            return my_direction
        else:
            # find closest pacman
            closest_pacman = None
            closest_distance = None
            for pacman in pacman_positions:
                distance = get_distance(pacman, my_position)
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_pacman = pacman
            clamped_goal = clamp(closest_pacman + relative_to_pacman, Position(0, 0), Position(board_size[0] - 1, board_size[1] - 1))
            if clamped_goal == my_position:
                clamped_goal = closest_pacman
            walls_with_previous_position_blocked = walls.copy()
            walls_with_previous_position_blocked.add(direction_to_new_position(my_position, ~my_direction, board_size))
            path = find_path(my_position, clamped_goal, walls_with_previous_position_blocked, board_size)
            if not path:
                path = find_path(my_position, closest_pacman, walls_with_previous_position_blocked, board_size)
            if not path:
                path = find_path(my_position, closest_pacman, walls, board_size)
            if not path:
                while not path:
                    goal = get_any_position(board_size, my_position, walls)
                    path = find_path(my_position, goal, walls, board_size)
            return positions_to_direction(my_position, path[0], board_size)

    def get_any_position(board_size, my_position, walls):
        all_positions = set()
        for x in range(board_size[0]):
            for y in range(board_size[1]):
                all_positions.add(Position(x, y))
        goal = choice(list(all_positions - walls - {my_position}))
        return goal

    return strategy


class Ghost:
    def __init__(self, strategy_normal, strategy_eatable):
        self.strategy_normal = strategy_normal
        self.strategy_eatable = strategy_eatable
        self.was_eatable = False

    def make_move(self, my_position, my_direction, walls, pacman_positions, board_size, is_eatable):
        changed = False
        if is_eatable ^ self.was_eatable:
            changed = True
        self.was_eatable = is_eatable
        if is_eatable:
            return self.strategy_eatable(my_position, my_direction, walls, pacman_positions, board_size, changed)
        else:
            return self.strategy_normal(my_position, my_direction, walls, pacman_positions, board_size, changed)


class Ghosts:
    RED = Ghost(strategy_normal_factory(Position(0, 0)), default_strategy_eatable)
    PINK = Ghost(strategy_normal_factory(Position(2, 2)), default_strategy_eatable)
    BLUE = Ghost(strategy_normal_factory(Position(-2, 0)), default_strategy_eatable)
    ORANGE = Ghost(strategy_normal_factory(Position(0, -2)), default_strategy_eatable)
