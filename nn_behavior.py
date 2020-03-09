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
    input_training_data = []
    training_games = 1000
    steps_per_game = 2000
    steps_count = 0
    task1 = None
    task2 = None
    model_train2_step = 50
    model_train1_step = 100
    model_train2_input_limit = 5000
    model_train1_input_limit = 2000
    game_speed = 20

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()

        for _ in range(steps_per_game):

            quit_game, collision, direction, snake_position, apple_position, score, input_training_data = await predict_and_play( \
                training_data_x, training_data2_x, model1, model2, \
                snake_position, apple_position, snake_start, score, display, clock, show_game, game_speed, \
                training_data2_y, training_data_y, input_training_data, \
                steps_count)

            steps_count += 1

            if steps_count % model_train2_step == 0:
                if task2 != None:
                    await task2
                task2 = asyncio.create_task(train_model(model2, training_data2_x, training_data2_y, 'model2.h5'))
                if len(training_data2_y) >= model_train2_input_limit:
                    for j in range(model_train2_step):
                        training_data2_x.pop(0)
                        training_data2_y.pop(0)

            if steps_count % model_train1_step == 0:
                if task1 != None:
                    await task1
                task1 = asyncio.create_task(train_model(model1, training_data_x, training_data_y, 'model1.h5'))
                if len(training_data_y) >= model_train1_input_limit:
                    for j in range(model_train1_step):
                        training_data_x.pop(0)
                        training_data_y.pop(0)

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
                        input_training_data, steps_count):
    train_data_len = len(input_training_data)
    train_data2_len = len(training_data2_x)
    if train_data_len > 0:
        predicted_wrong_direction = np.argmax(model1.predict(np.array([input_training_data[0]]))) - 1
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

    x, y, input_training_data = get_training_data_ex(prev_snake_position, \
        is_front_blocked, is_left_blocked, is_right_blocked)
    # x = []
    # y = []
    x2, y2 = get_training_data2(button_direction, snake_position, prev_apple_position)

    if y2 != None and prev_apple_position == apple_position:
        training_data2_x.append(x2)
        training_data2_y.append(y2)

    for item in x:
        training_data_x.append(item)
    for item in y:
        training_data_y.append(item)
    
    return quit_game, collision, direction, snake_position, apple_position, score, input_training_data

async def train_model(model,training_data_x,training_data_y,save_to_file_name):
    if len(training_data_x) == 0:
        return
    npx = np.array(training_data_x)
    npy = np.array(training_data_y)
    model.fit(npx,npy, batch_size = 256, epochs= 3)
    model.save_weights(os.path.dirname(__file__) + '\\' + save_to_file_name)
    print('\nModel trainded and saved')

def get_index(position):
    return int(position/cell_size)

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

def get_training_data_ex(snake_position, is_front_blocked, is_left_blocked, is_right_blocked):
    size = view_window_size * 2
    map1 = []
    map2 = []
    map3 = []
    map4 = []
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    current_direction_vector = current_direction_vector/cell_size
    current_direction_vector = [int(item) for item in current_direction_vector]
    first_axis = 1
    second_axis = 0
    if current_direction_vector[0] == 0:
        first_axis = 0
        second_axis = 1
    x_shift = -1
    if current_direction_vector[second_axis] > 0:
        x_shift = 1
    if current_direction_vector[0] == 0:
        y_shift = current_direction_vector[second_axis]
    else:
        y_shift = -current_direction_vector[second_axis]
    first_shift = y_shift
    second_shift = x_shift
    if current_direction_vector[0] == 0:
        first_shift = x_shift
        second_shift = y_shift
    # first_start = snake_position[0][first_axis] + first_shift * view_window_size
    # first_end = snake_position[0][first_axis] - first_shift * view_window_size
    # second_start = snake_position[0][second_axis] + second_shift * view_window_size
    # second_end = snake_position[0][second_axis] + second_shift * view_window_size
    for j in range(size+1):
        for i in range(size+1):
            ii = snake_position[0][first_axis] + cell_size * (view_window_size * first_shift - i * first_shift)
            jj = snake_position[0][second_axis] + cell_size * (view_window_size * second_shift - j * second_shift)
            point = [0, 0]
            point[first_axis] = ii
            point[second_axis] = jj
            if point in snake_position or ii < 0 or jj < 0 or ii >= display_width or jj >= display_height :
                map1.append([item/display_width for item in point])
                map4.append([0.0, 0.0])
            else:
                map2.append([item/display_width for item in point])
                map3.append([0.0, 0.0])
    x_to_add = []
    y_to_add = []
    input_data = np.concatenate((map1, map3, map2, map4)).reshape(-1)
    
    for i in range(3):
        direction = i - 1
        if (is_front_blocked == 1 and direction == 0) or \
                    (is_left_blocked == 1 and direction == -1) or \
                    (is_right_blocked == 1 and direction == 1):
            x_to_add.append(input_data)
            y_to_add.append(i)

    return x_to_add, y_to_add, [input_data]

def get_training_data(snake_position, is_front_blocked, is_left_blocked, is_right_blocked):
    snake_head_mark = 0.2
    snake_body_mark = 0.5
    apple_mark = 0.7
    map = np.full((int(display_width/cell_size), int(display_height/cell_size)), 0.0)
    # map = np.full((int(display_width/cell_size), int(display_height/cell_size), 1), 0.0)
    for i in snake_position[1:]:
        x = get_index(i[0])
        y = get_index(i[1])
        # map[x,y,0] = snake_body_mark
        map[x,y] = snake_body_mark
    x = get_index(snake_position[0][0])
    y = get_index(snake_position[0][1])
    # map[x,y,0] = snake_head_mark
    map[x,y] = snake_head_mark

    input_data = map.reshape(-1)
    x_to_add = []
    y_to_add = []
    
    for i in range(3):
        direction = i - 1
        if (is_front_blocked == 1 and direction == 0) or \
                    (is_left_blocked == 1 and direction == -1) or \
                    (is_right_blocked == 1 and direction == 1):
            x_to_add.append(input_data)
            y_to_add.append(i)
    
    return x_to_add, y_to_add, [input_data]

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