from game import *
from tqdm import tqdm


def generate_training_data(model, display, clock):
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

            train_data_len = len(training_data_x)
            if train_data_len > 0:
                predicted_wrong_direction = np.argmax(model.predict(np.array([training_data_x[train_data_len-1]]))) - 1
                direction = get_predicted_direction(predicted_wrong_direction)
                button_direction = direction_vector(snake_position, direction)
                print('predicted_wrong_direction: ', predicted_wrong_direction)
            else:
                direction, button_direction = generate_random_direction(snake_position)

            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            # button_direction, training_data_y = generate_training_data_y(snake_position, direction,
            #                                                              training_data_y)
            # training_data_x.append(
            #     [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
            #      snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
            #      snake_direction_vector_normalized[1]])


            quit_game, snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            collision = (is_front_blocked == 1 and direction == 0) or \
                    (is_left_blocked == 1 and direction == -1) or \
                    (is_right_blocked == 1 and direction == 1)

            x, y = get_training_data(collision, direction, snake_position, apple_position)

            for item in x:
                training_data_x.append(item)
            for item in y:
                training_data_y.append(item)

            npx = np.array(training_data_x)
            npy = np.array(training_data_y)
            print('x shape: ', np.array(x).shape)
            print('training_data_x shape: ', npx.shape)
            print('training_data_y shape: ', npy.shape)

            if quit_game:
                return training_data_x, training_data_y

            if _ == steps_per_game - 1:
                model.fit(npx,npy, batch_size = 256, epochs= 3)
                print('\nExit game after 2000 steps!')

            if collision:
                model.fit(npx,npy, batch_size = 256, epochs= 3)
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

def get_index(position):
    return int(position/cell_size)

def get_training_data(collision, direction, snake_position, apple_position):
    snake_head_mark = 0.2
    snake_body_mark = 0.5
    apple_mark = 0.7
    map = np.full((display_width, display_height), 0.0)
    for i in snake_position[1:]:
        x = get_index(i[0])
        y = get_index(i[1])
        map[x,y] = snake_body_mark
    x = get_index(snake_position[0][0])
    y = get_index(snake_position[0][1])
    map[x,y] = snake_head_mark
    x = get_index(apple_position[0])
    y = get_index(apple_position[1])
    map[x,y] = apple_mark

    reshaped = map.reshape(display_width*display_height)
    x_to_add = []
    y_to_add = []
    if collision:
        x_to_add.append(reshaped)
    else:
        x_to_add.append(reshaped)
        x_to_add.append(reshaped)

    for i in range(3):
        if collision and i-1 == direction:
            y_to_add.append(i)
        if collision==False and i-1 != direction:
            y_to_add.append(i)
    
    return x_to_add, y_to_add

def get_predicted_direction(wrong_direction):
    direction = random.randint(-1, 1)
    while direction == wrong_direction:
        direction = random.randint(-1, 1)
    return direction