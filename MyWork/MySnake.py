import pygame
import random
import time
import math
from tqdm import tqdm
import numpy as np
import math
from random import randrange 
from keras.models import Sequential 
from keras.layers import Dense


def display_snake(snake_position, display):
    for position in snake_position:
        if position==snake_position[0]:
            if display:
                pygame.draw.rect(display, (0, 0, 0), pygame.Rect(position[0], position[1], 10, 10))
                continue
        if display:
           pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))
    

def display_apple(apple_position, display):
    if display:
        pygame.draw.rect(display, (0, 255, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))

def starting_positions():
    field = np.zeros(2500, dtype=int)
    snake_start = [200, 100]
    snake_position = []
    x = snake_start[0]
    y = snake_start[1]
    score = randrange(3, 4)
    length = score
    for i in range(math.ceil(score/10)):
        if length-10>0:
            x_length=10
        else:
            x_length=length
        for i2 in range(x_length):
            snake_position.append([])
            if y/10%2==0:
                snake_position[i2+10*i].append(x+i2*10)
                snake_position[i2+10*i].append(y)
                fieldCell = int((x+i2*10)/10+50*(y/10))
            else:
                snake_position[i2+10*i].append(90+x-i2*10)
                snake_position[i2+10*i].append(y)
                fieldCell = int((90+x-i2*10)/10+50*(y/10))
            field[fieldCell] = 1
        length = length-10  
        y=y+10
        fieldCell = int((snake_start[0])/10+50*(snake_start[1]/10))
        field[fieldCell] = 2
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    for aplle_position in  snake_position:
        apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
            
    return snake_start, snake_position, apple_position, score, field

def apple_distance_from_snake(apple_position, snake_head):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_head))
    
def generate_snake(snake_start, snake_position, apple_position, button_direction, score, stepsBeforeGrowth, field):
    # right=1, left=0, top=2, bottom=3
    
    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10
    fieldCell = int(snake_start[0]/10-1+50*(snake_start[1]/10-1))
    field[fieldCell] = 1
    # if stepsBeforeGrowth==0:
    #     snake_position.insert(0, list(snake_start))
    #     score += 1
    #     stepsBeforeGrowth = 5
        
    # else:
    #     snake_position.insert(0, list(snake_start))
    #     fieldCell = int(snake_position[-1][0]/10+50*(snake_position[-1][1]/10))
    #     field[fieldCell] = 0
    #     snake_position.pop()
    # stepsBeforeGrowth -= 1
        
    if snake_start == apple_position:
        apple_position, score = collision_with_apple(snake_position ,apple_position, score)
        snake_position.insert(0, list(snake_start))

    else:
        snake_position.insert(0, list(snake_start))
        fieldCell = int(snake_position[-1][0]/10+50*(snake_position[-1][1]/10))
        field[fieldCell] = 0
        snake_position.pop()
    
    return snake_position, apple_position, score, stepsBeforeGrowth, field


def collision_with_apple(snake_position, apple_position, score):
    for aplle_position in  snake_position:
        apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_start):
    if snake_start[0] >= 500 or snake_start[0] < 0 or snake_start[1] >= 500 or snake_start[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_start, snake_position):
    # snake_start = snake_position[0]
    if snake_start in snake_position[1:]:
        return 1
    else:
        return 0


def blocked_directions(snake_position):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    is_front_blocked = is_direction_blocked(snake_position, current_direction_vector)
    is_left_blocked = is_direction_blocked(snake_position, left_direction_vector)
    is_right_blocked = is_direction_blocked(snake_position, right_direction_vector)

    return current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked


def is_direction_blocked(snake_position, current_direction_vector):
    next_step = snake_position[0] + current_direction_vector
    snake_start = snake_position[0]
    if collision_with_boundaries(next_step) == 1 or collision_with_self(next_step.tolist(), snake_position) == 1:
        return 1
    else:
        return 0


def generate_random_direction(snake_position, angle_with_apple):
    direction = 0
    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0
    # direction = randrange(-1, 2)
    return direction_vector(snake_position, direction)


def direction_vector(snake_position, direction):
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction = current_direction_vector
    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector

    button_direction = generate_button_direction(new_direction)
    return direction, button_direction


def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3
    return button_direction


def angle_with_apple(snake_position, apple_position):
    print(np.array(snake_position[0]))
    apple_direction_vector = np.array(apple_position) - np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[1],
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[0]) / math.pi
    return angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized


def play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock, stepsBeforeGrowth, field):
    if clock==False and display==False:
        snake_position, apple_position, score, stepsBeforeGrowth, field = generate_snake(snake_start, snake_position, apple_position,
                                                           button_direction, score, stepsBeforeGrowth, field)
        return False, snake_position, apple_position, score, stepsBeforeGrowth, field
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, snake_position, apple_position, score
    display.fill((255, 255, 255))

    display_apple(apple_position, display)
    display_snake(snake_position, display)

    snake_position, apple_position, score, stepsBeforeGrowth, field = generate_snake(snake_start, snake_position, apple_position,
                                                           button_direction, score, stepsBeforeGrowth, field)
    pygame.display.set_caption("SCORE: " + str(score))
    pygame.display.update()
    clock.tick(200)
    return False, snake_position, apple_position, score, stepsBeforeGrowth, field
    
#Test
def generate_training_data(model, display, clock, model2):
    training_data_x = []
    training_data_y = []
    training_games = 500
    steps_per_game = 500
    train_x2 = []
    train_y2 = []

    for _games in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score, field = starting_positions()

        stepsBeforeGrowth = 5
        for _steps in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position, angle)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            # if training_data_y:
            #     print('diff: ', len(training_data_y))
            
            # direction, button_direction, training_data_y = generate_training_data_y(snake_position, angle_with_apple,
            #                                                                         button_direction, direction,
            #                                                                         training_data_y, is_front_blocked,
            #                                                                         is_left_blocked, is_right_blocked, score)
            snake_head_view = []
            for i in range(3):
                int(snake_position[0][0]/10+50*(snake_position[0][1]/10))
                for i2 in range(3):
                    if snake_position[0][0]/10+i2 >= 50 or snake_position[0][1]/10+i >= 50 or \
                       snake_position[0][0]/10+i2 < 0 or snake_position[0][1]/10+i < 0 :
                        snake_head_view.append(1)
                    else:
                        if field[int(snake_position[0][0]/10+i2-1+50*(snake_position[0][1]/10+i-1))] == 1:
                            snake_head_view.append(1)
                        else:
                            snake_head_view.append(0)
            snake_head_view[4] = 2
            
            print(snake_head_view)
            direction, button_direction, training_data_y, available_dir = generate_training_data_y(snake_head_view, training_data_y, direction, button_direction, snake_position)
            print(available_dir)
            # if available_dir[0]==1:
            #     training_data_x.append(snake_head_view)
            # if available_dir[1]==1:
            #     training_data_x.append(snake_head_view)
            # if available_dir[2]==1:
            #     training_data_x.append(snake_head_view)
            
            
            
            if available_dir:
                training_data_x.append(snake_head_view)
            npx = np.array(training_data_x)
            print(npx[-1])
            npy = np.array(training_data_y)
            
            
            train_x2.append([snake_start[0]])
            train_x2[-1].append(snake_start[1])
            train_x2[-1].append(apple_position[0])
            train_x2[-1].append(apple_position[1])
            
            absolute_direction = absolute_apple_direction(apple_position, snake_position)
            relative_direction = absolute_to_relative_direction(absolute_direction, snake_position)
            train_y2.append(absolute_direction)
            
            npx2 = np.array(train_x2)
            npy2 = np.array(train_y2)
            print(train_y2, apple_direction_vector_normalized)
            model2.fit(npx2, npy2, batch_size=256, epochs=3)
            print(snake_position[0], " zero", snake_position[1], " one")
            if snake_position[0] in  snake_position[1:] or collision_with_boundaries(snake_start):
                model.fit(npx, npy, batch_size=256, epochs=3)
                print('collision with self or boundaries')
                break
            
            if len(npx) > 0:
                # direction = np.argmax(model.predict(np.array([npx[-1]])))-1
                # direction, button_direction = direction_vector(snake_position, direction)
                # print(direction, button_direction)
                direction = np.argmax(model2.predict(np.array([npx2[-1]])))-1
                direction, button_direction = direction_vector(snake_position, direction)
                
            quit_game, snake_position, apple_position, score, stepsBeforeGrowth, field = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock, stepsBeforeGrowth, field)
            # for i in field:
            #     print(i, end='')
            if quit_game:
                return
            if _steps == steps_per_game-1:
                model.fit(npx, npy, batch_size=256, epochs=3)
                print('run out of steps')
                
    return training_data_x, training_data_y
def absolute_apple_direction(apple_position, snake_position):
    south = np.array([[0, 10]])
    north = np.array([[0, -10]])
    west = np.array([[-10, 0]])
    east = np.array([[10, 0]])
    absolute_direction = 0

    if (np.array(snake_position[0])-np.array(snake_position[1])) in south:
        least_distance = apple_distance_from_snake(apple_position, snake_position[0]+south)
    else:
        least_distance = 1000
    if np.linalg.norm(np.array(apple_position) - np.array(snake_position[0])+np.array(north[0]))<least_distance and (np.array(snake_position[0])-np.array(snake_position[1])) in north:
        least_distance = apple_distance_from_snake(apple_position, snake_position[0]+north)
        absolute_direction = 1
    if np.linalg.norm(np.array(apple_position) - np.array(snake_position[0])+np.array(west[0]))<least_distance and (np.array(snake_position[0])-np.array(snake_position[1])) in west:
        least_distance = apple_distance_from_snake(apple_position, snake_position[0]+west)
        absolute_direction = 2
    if np.linalg.norm(np.array(apple_position) - np.array(snake_position[0])+np.array(east[0]))<least_distance and (np.array(snake_position[0])-np.array(snake_position[1])) in east:
        least_distance = apple_distance_from_snake(apple_position, snake_position[0]+east)
        absolute_direction = 3
    return absolute_direction
def absolute_to_relative_direction(absolute_direction, snake_position):
    south = np.array([[0, 10]])
    north = np.array([[0, -10]])
    west = np.array([[-10, 0]])
    east = np.array([[10, 0]])
    ad_array = [south[0], north[0], west[0], east[0]]
    back = np.array(snake_position[0])-np.array(snake_position[1])
    if back in south:
        forward = north
        right = east
        left = west
    elif back in north:
        forward = south
        right = west
        left = east
    elif back in west:
        forward = east
        right = north
        left = south
    else:
        forward = west
        right = south
        left = north
    if np.array(ad_array[absolute_direction]) in forward:
        return 1
    elif np.array(ad_array[absolute_direction]) in right:
        return 2
    elif np.array(ad_array[absolute_direction]) in left:
        return 0

def generate_training_data_y(snake_head_view, training_data_y, direction, button_direction, snake_position):
    direction = 0
    available_dir = []
    for i in range(len(snake_head_view)):
        # if snake_head_view[i]==0 and i==1:
        #     training_data_y.append(1)
        #     available_dir.append(1)
        # elif i==1: 
        #     available_dir.append(0)
        # if snake_head_view[i]==0 and i==3:
        #     direction = -1
        #     training_data_y.append(0)
        #     available_dir.append(1)
        # elif i==3: 
        #     available_dir.append(0)
        # if snake_head_view[i]==0 and i==5:
        #     direction = 1
        #     training_data_y.append(2)
        #     available_dir.append(1)
        # elif i==5:
        #     available_dir.append(0)
        if snake_head_view[i]==0 and i==1:
            training_data_y.append(1)
            available_dir.append(1)
            break
        elif snake_head_view[i]==0 and i==3:
            direction = -1
            training_data_y.append(0)
            available_dir.append(1)
            break
        elif snake_head_view[i]==0 and i==5:
            direction = 1
            training_data_y.append(2)
            available_dir.append(1)
        button_direction = direction_vector(snake_position, direction)
    return direction, button_direction, training_data_y, available_dir
                
            
# def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
#                              is_front_blocked, is_left_blocked, is_right_blocked, score):
#     if direction == -1:
#         if is_left_blocked == 1:
#             if is_front_blocked == 1 and is_right_blocked == 0:
#                 direction = 1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([2])#[1, 0, 0]
#             elif is_front_blocked == 0 and is_right_blocked == 1:
#                 direction = 0
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([1])#[0, 1, 0]
#             elif is_front_blocked == 0 and is_right_blocked == 0:
#                 direction = 1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([1])#[0, 0, 1]
#             else:
#                 training_data_y.append([1])
#         else:
#             training_data_y.append([0])
#     elif direction == 0:
#         if is_front_blocked == 1:
#             if is_left_blocked == 1 and is_right_blocked == 0:
#                 direction = 1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([2])
#             elif is_left_blocked == 0 and is_right_blocked == 1:
#                 direction = -1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([0])
#             elif is_left_blocked == 0 and is_right_blocked == 0:
#                 direction = 0
#                 training_data_y.append([2])
#                 button_direction = direction_vector(snake_position, direction)
#             else:
#                 training_data_y.append([1])
#         else:
#             training_data_y.append([1])
#     else:
#         if is_right_blocked == 1:
#             if is_left_blocked == 1 and is_front_blocked == 0:
#                 direction = 0
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([1])
#             elif is_left_blocked == 0 and is_front_blocked == 1:
#                 direction = -1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([0])
#             elif is_left_blocked == 0 and is_front_blocked == 0:
#                 direction = 1
#                 button_direction = direction_vector(snake_position, direction)
#                 training_data_y.append([1])
#             else:
#                 training_data_y.append([2])
#         else:
#             training_data_y.append([1])
            
#     return direction, button_direction, training_data_y

def run_game(graphicall_interface):
    model = build_model()
    model2 = build_apple_model()
    if graphicall_interface:
        display_width = 500
        display_height = 500
        pygame.init()
        display=pygame.display.set_mode((display_width,display_height))
        clock=pygame.time.Clock()
        generate_training_data(model, display, clock, model2)
    else:
        training_data_x, training_data_y = generate_training_data(model, False, False, model2)
       
def build_model():
    model = Sequential()
    model.add(Dense(250, input_dim=9, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_apple_model():
    apple_model = Sequential()
    apple_model.add(Dense(250, input_dim=4, activation='relu'))
    apple_model.add(Dense(50, activation='relu'))
    apple_model.add(Dense(3, activation='softmax'))
    apple_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return apple_model

run_game(True)