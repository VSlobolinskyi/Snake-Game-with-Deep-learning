import pygame
import random
import time
import math
from tqdm import tqdm
import numpy as np
import math


def display_snake(snake_position, display):
    for position in snake_position:
        pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))


def display_apple(apple_position, display):
    pygame.draw.rect(display, (0, 255, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))

# def move(cellCoordinates, x_move, y_move):
#     if cellCoordinates[1] == 0 and x_move == 10 :
#         return 10, 0
    
#     if cellCoordinates[1] == 100 and x_move == 10 :
#         return -10, 0
    
#     if cellCoordinates[1] == 0 and x_move == 0:
#         return 0, 10
    
#     if cellCoordinates[1] == 100 and x_move == 0:
#         return 0, 10
    
#     return x_move, y_move

# def calculate_values_for_starting_positon():
#     snake_start = [100, 100]
#     snake_position = []
#     score = 15
#     length = score
#     cellCoordinates = snake_start
#     y_move = -10
#     x_move = 10
#     while (length>0):
#         snake_position.append(cellCoordinates)
        
#         y_move, x_move = move(cellCoordinates, x_move, y_move)
            
#         cellCoordinates = [cellCoordinates[0] + x_move, cellCoordinates[1] + y_move]
#         length = length-1
#     return snake_start, snake_position, score

def starting_positions():
    snake_start = [200, 200]
    snake_position = []
    x = snake_start[0]
    y = snake_start[1]
    score = 100
    for i in range(math.ceil(score/10)):
        for i2 in range(10):
            # if(y==snake_start[1] and score!=10):
            #     i2=10-score%10
            snake_position.append([])
            if(y%2==0):
                snake_position[i2+10*i].append(x+i2*10)
                snake_position[i2+10*i].append(y)
            else:
                snake_position[i2+10*i].append(x-i2*10)
                snake_position[i2+10*i].append(y)
        y=y+10
    print(snake_position)
    
    

    
    # while (length>0):
    #     cellCoordinates = snake_start
    #     while((cellCoordinates[1]>0 and cellCoordinates[1]%2==0) or (cellCoordinates[1]<100)):
    #         while(cellCoordinates[0]>0):
    #             snake_position.append(cellCoordinates) [[100, 100], [90, 100], [80, 100]]
    #             if cellCoordinates[1]%2==0:
    #                 cellCoordinates[0] = cellCoordinates[0]-10
    #             else:
    #                 cellCoordinates[0] = cellCoordinates[0]+10
    #         cellCoordinates = cellCoordinates[1]-10
    #     length = length-1
    
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    return snake_start, snake_position, apple_position, score

def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))


def generate_snake(snake_start, snake_position, apple_position, button_direction, score):
    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10

    if snake_start == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_start))

    else:
        snake_position.insert(0, list(snake_start))
        snake_position.pop()

    return snake_position, apple_position, score


def collision_with_apple(apple_position, score):
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


def play_game(snake_start, snake_position, apple_position, button_direction, score, display, clock):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, snake_position, apple_position, score
    display.fill((255, 255, 255))

    display_apple(apple_position, display)
    display_snake(snake_position, display)

    snake_position, apple_position, score = generate_snake(snake_start, snake_position, apple_position,
                                                           button_direction, score)
    pygame.display.set_caption("SCORE: " + str(score))
    pygame.display.update()
    clock.tick(10)
    return False, snake_position, apple_position, score
    
#Test
def generate_training_data(display, clock):
    training_data_x = []
    training_data_y = []
    training_games = 1000
    steps_per_game = 10000

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()
        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)

        for _ in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position, angle)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            direction, button_direction, training_data_y = generate_training_data_y(snake_position, angle_with_apple,
                                                                                    button_direction, direction,
                                                                                    training_data_y, is_front_blocked,
                                                                                    is_left_blocked, is_right_blocked)

            if is_front_blocked == 1 and is_left_blocked == 1 and is_right_blocked == 1:
                print('collision with self')
                break

            training_data_x.append(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0], \
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1], \
                 snake_direction_vector_normalized[1]])

            quit_game, snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)
            if quit_game:
                return
        if steps_per_game == 5000:
            print('run out of steps')

    return training_data_x, training_data_y


def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
                             is_front_blocked, is_left_blocked, is_right_blocked):
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

display_width = 500
display_height = 500
pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()
generate_training_data(display, clock)