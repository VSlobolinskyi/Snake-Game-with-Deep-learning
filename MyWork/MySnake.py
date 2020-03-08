import pygame
import random
import time
import math
from tqdm import tqdm
import numpy as np
import math
from random import randrange 
import tensorflow as tf

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
    snake_start = [200, 200]
    snake_position = []
    x = snake_start[0]
    y = snake_start[1]
    score = 4   
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
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    for aplle_position in  snake_position:
        apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
            
    return snake_start, snake_position, apple_position, score, field

def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))
    
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
    fieldCell = int(snake_start[0]/10+50*(snake_start[1]/10))
    field[fieldCell] = 1
    if stepsBeforeGrowth==0:
        snake_position.insert(0, list(snake_start))
        score += 1
        stepsBeforeGrowth = 5
        
    else:
        snake_position.insert(0, list(snake_start))
        fieldCell = int(snake_position[-1][0]/10+50*(snake_position[-1][1]/10))
        field[fieldCell] = 0
        snake_position.pop()
    # stepsBeforeGrowth -= 1
        
    # if snake_start == apple_position:
    #     apple_position, score = collision_with_apple(snake_position ,apple_position, score)
    #     snake_position.insert(0, list(snake_start))

    # else:
    #     snake_position.insert(0, list(snake_start))
    #     snake_position.pop()
    
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
    direction = randrange(-1, 2)
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
    clock.tick(10)
    return False, snake_position, apple_position, score, stepsBeforeGrowth, field
    
#Test
def generate_training_data(display=False, clock=False):
    training_data_x = []
    training_data_y = []
    training_games = 1
    steps_per_game = 10

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score, field = starting_positions()

        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)
        stepsBeforeGrowth = 5
        for _ in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position, angle)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            
            direction, button_direction, training_data_y = generate_training_data_y(snake_position, angle_with_apple,
                                                                                    button_direction, direction,
                                                                                    training_data_y, is_front_blocked,
                                                                                    is_left_blocked, is_right_blocked, score)
            
            if snake_position[0] in  snake_position[1:] or collision_with_boundaries(snake_start):
                print(snake_position)
                print('collision with self or boundaries')
                break
            
            training_data_x.append(field)
            quit_game, snake_position, apple_position, score, stepsBeforeGrowth, field = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock, stepsBeforeGrowth, field)
            for i in field:
                print(i, end='')
            if quit_game:
                return
            if _ == steps_per_game-1:
                print('run out of steps')
    print(len(training_data_y))
    return training_data_x, training_data_y

def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
                             is_front_blocked, is_left_blocked, is_right_blocked, score):
    if direction == -1:
        if is_left_blocked == 1:
            if is_front_blocked == 1 and is_right_blocked == 0:
                direction = 1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([0, 0, 1])
            elif is_front_blocked == 0 and is_right_blocked == 1:
                direction = 0
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([0, 1, 0])
            elif is_front_blocked == 0 and is_right_blocked == 0:
                direction = 1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([0, 0, 1])
        else:
            training_data_y.append([1, 0, 0])
    elif direction == 0:
        if is_front_blocked == 1:
            if is_left_blocked == 1 and is_right_blocked == 0:
                direction = 1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([0, 0, 1])
            elif is_left_blocked == 0 and is_right_blocked == 1:
                direction = -1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked == 0 and is_right_blocked == 0:
                direction = 0
                training_data_y.append([0, 0, 1])
                button_direction = direction_vector(snake_position, direction)
        else:
            training_data_y.append([0, 1, 0])
    else:
        if is_right_blocked == 1:
            if is_left_blocked == 1 and is_front_blocked == 0:
                direction = 0
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([0, 1, 0])
            elif is_left_blocked == 0 and is_front_blocked == 1:
                direction = -1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked == 0 and is_front_blocked == 0:
                direction = 1
                button_direction = direction_vector(snake_position, direction)
                training_data_y.append([1, 0, 0])
        else:
            training_data_y.append([0, 0, 1])

    return direction, button_direction, training_data_y

def run_game(graphicall_nterface):
    if graphicall_nterface:
        display_width = 500
        display_height = 500
        pygame.init()
        display=pygame.display.set_mode((display_width,display_height))
        clock=pygame.time.Clock()
        generate_training_data(display, clock)
    else:
        clock=pygame.time.Clock()
        generate_training_data()
        

run_game(False)