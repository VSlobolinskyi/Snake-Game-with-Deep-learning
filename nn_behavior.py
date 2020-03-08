from game import *
from tqdm import tqdm
import asyncio


async def generate_training_data(model1, model2, display, clock):
    training_data_x = []
    training_data_y = []
    training_data2_x = []
    training_data2_y = []
    training_games = 1000
    steps_per_game = 2000

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()

        for _ in range(steps_per_game):
            prev_apple_position = apple_position
            train_data_len = len(training_data_x)
            train_data2_len = len(training_data2_x)
            if train_data_len > 0:
                predicted_wrong_direction = np.argmax(model1.predict(np.array([training_data_x[train_data_len-1]]))) - 1
                if train_data2_len > 0:
                    predicted_direction_to_apple = np.argmax(model2.predict(np.array([training_data2_x[train_data2_len-1]])))
                else:
                    predicted_direction_to_apple = None
                direction, button_direction = get_predicted_direction(None, predicted_direction_to_apple, snake_position)
                print('predicted_wrong_direction: ', predicted_wrong_direction)
                print('predicted_direction_to_apple: ', predicted_direction_to_apple)
                print('result direction: ', direction)
                print('result absolute direction: ', button_direction)
            else:
                direction, button_direction = generate_random_direction(snake_position)

            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            quit_game, snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            collision = (is_front_blocked == 1 and direction == 0) or \
                    (is_left_blocked == 1 and direction == -1) or \
                    (is_right_blocked == 1 and direction == 1)

            x, y = get_training_data(collision, direction, snake_position, apple_position)
            x2, y2 = get_training_data2(button_direction, snake_position, apple_position)

            if y2 != None and prev_apple_position == apple_position:
                training_data2_x.append(x2)
                training_data2_y.append(y2)

            for item in x:
                training_data_x.append(item)
            for item in y:
                training_data_y.append(item)

            if quit_game:
                return training_data_x, training_data_y

            if _ == steps_per_game - 1:
                # asyncio.run(train_model(model1,npx,npy))
                # asyncio.create_task(train_model(model2,training_data2_x,training_data2_y))
                await train_model(model2,training_data2_x,training_data2_y)
                print('\nExit game after 2000 steps!')

            if collision:
                # asyncio.run(train_model(model1,npx,npy))
                # asyncio.create_task(train_model(model2,training_data2_x,training_data2_y))
                await train_model(model2,training_data2_x,training_data2_y)
                print('\nExit game with collision! Direction: ', direction)
                break

    return training_data_x, training_data_y

async def train_model(model,training_data_x,training_data_y):
    npx = np.array(training_data_x)
    npy = np.array(training_data_y)
    model.fit(npx,npy, batch_size = 256, epochs= 3)
    print('\nModel trainded')

def get_index(position):
    return int(position/cell_size)

def get_training_data2(button_direction, snake_position, apple_position):
    curr_dist = apple_distance_from_snake(apple_position, snake_position[0])
    prev_dist = apple_distance_from_snake(apple_position, snake_position[1])
    if curr_dist >= prev_dist:
        return None, None

    sx = snake_position[0][0] / display_width
    sy = snake_position[0][1] / display_height
    ax = apple_position[0] / display_width
    ay = apple_position[1] / display_height

    x_to_add = [sx, sy, ax, ay]
    y_to_add = button_direction

    return x_to_add, y_to_add

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

def get_predicted_direction(wrong_direction, direction_to_apple, snake_position):
    if direction_to_apple == None:
        direction = random.randint(-1, 1)
    else:
        button_direction = direction_to_apple
        for i in range(3):
            tmp_direction = i-1
            tmp_button_direction = direction_vector(snake_position, tmp_direction)
            if button_direction == tmp_button_direction:
                break
        if button_direction == tmp_button_direction:
            direction = tmp_direction
        else:
            direction = random.randint(-1, 1)
    
    button_direction = direction_vector(snake_position, direction)

    if wrong_direction != None:
        wrong_button_direction = direction_vector(snake_position, wrong_direction)

        while button_direction == wrong_button_direction:
            direction = random.randint(-1, 1)
            button_direction = direction_vector(snake_position, direction)

    return direction, button_direction