from game import *
from tqdm import tqdm
import asyncio
import os.path

async def generate_training_data(model1, model2, display, clock, show_game=True):

    if os.path.isfile(os.path.dirname(__file__) + '\\' + 'model1.h5'):
        model1.load_weights(os.path.dirname(__file__) + '\\' + 'model1.h5')
    if os.path.isfile(os.path.dirname(__file__) + '\\' + 'model2.h5'):
        model2.load_weights(os.path.dirname(__file__) + '\\' + 'model2.h5')

    training_data_x = []
    training_data_y = []
    training_data2_x = []
    training_data2_y = []
    training_games = 1000
    steps_per_game = 2000
    steps_count = 0
    task1 = None
    task2 = None
    model_train2_step = 50
    model_train1_step = 200
    game_speed = 20

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()

        for _ in range(steps_per_game):

            quit_game, collision, direction, snake_position, apple_position, score = await predict_and_play( \
                training_data_x, training_data2_x, model1, model2, \
                snake_position, apple_position, snake_start, score, display, clock, show_game, game_speed, \
                training_data2_y, training_data_y, \
                steps_count)

            steps_count += 1

            if steps_count % model_train2_step == 0:
                if task2 != None:
                    await task2
                task2 = asyncio.create_task(train_model(model2, training_data2_x, training_data2_y, 'model2.h5'))
                if len(training_data2_y) > 5000:
                    training_data2_x = []
                    training_data2_y = []

            if steps_count % model_train1_step == 0:
                if task1 != None:
                    await task1
                task1 = asyncio.create_task(train_model(model1, training_data_x, training_data_y, 'model1.h5'))
                training_data_x = []
                training_data_y = []

            if quit_game:
                await train_model(model2, training_data2_x, training_data2_y, 'model2.h5')
                await train_model(model1, training_data_x, training_data_y, 'model1.h5')
                return

            if _ == steps_per_game - 1:
                print('\nExit game after 2000 steps!')

            if collision:
                print('\nExit game with collision! Direction: ', direction)
                break

    await train_model(model2,training_data2_x,training_data2_y, 'model2.h5')
    await train_model(model1, training_data_x, training_data_y, 'model1.h5')

    return

async def predict_and_play(training_data_x, training_data2_x, model1, model2, snake_position, apple_position,
                        snake_start, score, display, clock, show_game, game_speed, training_data2_y, training_data_y,
                        steps_count):
    train_data_len = len(training_data_x)
    train_data2_len = len(training_data2_x)
    if train_data_len > 0:
        predicted_wrong_direction = np.argmax(model1.predict(np.array([training_data_x[train_data_len-1]]))) - 1
    else:
        predicted_wrong_direction = None
    if train_data2_len > 0:
        predicted_direction_to_apple = np.argmax(model2.predict(np.array([training_data2_x[train_data2_len-1]])))
    else:
        predicted_direction_to_apple = None
    direction, button_direction = get_predicted_direction(predicted_wrong_direction, predicted_direction_to_apple, snake_position)
    print('predicted_wrong_direction: ', predicted_wrong_direction)
    print('predicted_direction_to_apple: ', predicted_direction_to_apple)
    print('result direction: ', direction)
    print('result absolute direction: ', button_direction)
    print('steps: ', steps_count)

    current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
        snake_position)

    prev_apple_position = [apple_position[0], apple_position[1]]
    prev_snake_position = snake_position
    quit_game, snake_position, apple_position, score = play_game(snake_start, \
                snake_position, apple_position, button_direction, score, \
                display, clock, show_game, game_speed)
    collision = (is_front_blocked == 1 and direction == 0) or \
            (is_left_blocked == 1 and direction == -1) or \
            (is_right_blocked == 1 and direction == 1)

    # x, y = get_training_data(collision, direction, prev_snake_position, prev_apple_position)
    x = []
    y = []
    x2, y2 = get_training_data2(button_direction, snake_position, prev_apple_position)

    if y2 != None and prev_apple_position == apple_position:
        training_data2_x.append(x2)
        training_data2_y.append(y2)

    for item in x:
        training_data_x.append(item)
    for item in y:
        training_data_y.append(item)
    
    return quit_game, collision, direction, snake_position, apple_position, score

async def train_model(model,training_data_x,training_data_y,save_to_file_name):
    if len(training_data_x) == 0:
        return
    npx = np.array(training_data_x)
    npy = np.array(training_data_y)
    model.fit(npx,npy, batch_size = 256, epochs= 3)
    model.save_weights(os.path.dirname(__file__) + '\\' + save_to_file_name)
    print('\nModel trainded and saved')

def get_index(position):
    return int(position/cell_size - 1)

def get_training_data2(button_direction, snake_position, apple_position):
    prev_snake = snake_position[1]
    curr_dist = apple_distance_from_snake(apple_position, snake_position[0])
    prev_dist = apple_distance_from_snake(apple_position, prev_snake)
    while curr_dist >= prev_dist:
        i = random.randint(0, 3)
        new_x, new_y = move_snake_start(i, prev_snake[0], prev_snake[1])
        curr_snake = [new_x, new_y]
        curr_dist = apple_distance_from_snake(apple_position, curr_snake)
        button_direction = i

    sx = prev_snake[0] / display_width
    sy = prev_snake[1] / display_height
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

    if direction_to_apple == None and wrong_direction == None:
        return generate_random_direction(snake_position)

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