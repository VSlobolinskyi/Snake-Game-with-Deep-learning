from game import *
from tqdm import tqdm


def generate_training_data(display, clock):
    training_data_x = []
    training_data_y = []
    training_games = 1000
    steps_per_game = 2000

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()
        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)

        for _ in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = \
                angle_with_apple(snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            button_direction, training_data_y = generate_training_data_y(snake_position, direction,
                                                                         training_data_y)

            training_data_x.append(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                 snake_direction_vector_normalized[1]])

            quit_game, snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            if quit_game:
                return training_data_x, training_data_y

            if _ == steps_per_game - 1:
                print('Exit game after 2000 steps!')

            if (is_front_blocked == 1 and direction == 0) or \
                    (is_left_blocked == 1 and direction == -1) or \
                    (is_right_blocked == 1 and direction == 1):
                print('\nExit game with collision! Direction: ', direction)
                break

    return training_data_x, training_data_y

def generate_training_data_y(snake_position, direction, training_data_y):

    button_direction = direction_vector(snake_position, direction)

    if direction == -1:
        training_data_y.append([1, 0, 0])
    if direction == 0:
        training_data_y.append([0, 1, 0])
    if direction == 1:
        training_data_y.append([0, 0, 1])

    return button_direction, training_data_y
