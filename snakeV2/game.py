import random
import math
import numpy as np

'''
complexity:
apple and snake head ignoring all collisions -> = 0
0 + collisions with borders ->  1
1 + snake fixed body -> direction = 2
2 + snake growing body -> direction = 3
3 + snake body collisions -> direction = 4
all possible -> 100
'''   

'''
abs_direction:
LEFT -> direction = 0
RIGHT -> direction = 1
DOWN -> direction = 2
UP -> direction = 3
'''     

class Env:
  def __init__(self, verbose = 0, complexity=100, min_iterations=500, max_iterations=1000):
    self.verbose = verbose
    if self.verbose == 1:
      print('Env init')
    self.field_width = 50
    self.field_height = 50
    self.done_reward = 10.0
    self.snake_position = []
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    # self.reset()

  # Resets env
  def reset(self): 
    if self.verbose == 1:
      print('Env reset')
    self.__make_starting_positions()
    self.info = []
    self.negative_reward = 0.0
    self.positive_reward = 0.0
    self.__fill_observation()
    self.iteration = 0

    return self.observation

  # Env step
  def step(self, action): 
    if self.verbose == 1:
      print('Env step')

    self.reward = 0.0
    self.done = False

    self.__move_snake(action)
    self.__fill_observation()
    self.iteration += 1
    
    return self.observation, self.reward, self.done, self.info

  def seed(self, seed):
    random.seed(seed)

  def get_observation_space(self):
    return self.field_width, self.field_height, 1

  def get_action_space_count(self):
    return 4

  def get_action_space_sample(self):
    return random.randrange(0, self.get_action_space_count())

  def __make_starting_positions(self):
    self.snake_position.clear()
    self.score = random.randrange(5, 25)
    snake_start_x = random.randrange(self.score, self.field_width)
    snake_start_y = random.randrange(0, self.field_height)
    self.snake_start = [snake_start_x, snake_start_y]
    for i in range(self.score):
        self.snake_position.append([snake_start_x-(i), snake_start_y])
    self.__make_new_apple_position()
    if self.verbose == 1:
      print('Starting posiotion snake head x:{}, y:{}, snake second x: {}, y: {}'.format(self.snake_position[0][0], self.snake_position[0][1], \
        self.snake_position[1][0], self.snake_position[1][1]))

  def __make_new_apple_position(self):
    self.apple_position = [self.snake_start[0], self.snake_start[1]]
    while self.apple_position in self.snake_position:
        self.apple_position = [random.randrange(0, self.field_width), random.randrange(0, self.field_height)]

  def __move_snake(self, abs_direction):
    new_x, new_y = self.__move_snake_start(abs_direction, self.snake_start[0], self.snake_start[1])
    snake_start = [new_x, new_y]
    if self.verbose == 1:
      print('Snake head old x:{}, y:{}'.format(self.snake_start[0], self.snake_start[1]))

    if self.__collision_with_boundaries(new_x, new_y) or self.__collision_with_self(new_x, new_y):
      self.reward = -1.0
      self.negative_reward += 1.0
      self.__make_starting_positions()
    else:
      snake_start[0] = self.__fix_out_of_field_coordinate(new_x, self.field_width)
      snake_start[1] = self.__fix_out_of_field_coordinate(new_y, self.field_height)

      if snake_start == self.apple_position:
        self.reward = 1.0
        self.score += 1
        self.positive_reward += 1.0
        self.snake_position.insert(0, list(snake_start))
      else:
        self.snake_position.insert(0, list(snake_start))
        self.snake_position.pop()

      if snake_start == self.apple_position:
        self.__make_new_apple_position()

      self.snake_start = snake_start

    if self.iteration > self.max_iterations or (self.iteration > self.min_iterations and self.reward != 0.0) :
      self.done = True

    if self.verbose == 1:
      print('Snake head new x:{}, y:{}'.format(self.snake_start[0], self.snake_start[1]))
      print('Snake head from position new x:{}, y:{}'.format(self.snake_position[0][0], self.snake_position[0][1]))

  def __fix_out_of_field_coordinate(self, coord, size):
    if coord < 0:
      return size - 1
    if coord >= size:
      return 0
    return coord

  def __move_snake_start(self, abs_direction, x, y):
    if abs_direction == 1:
        x += 1
    elif abs_direction == 0:
        x -= 1
    elif abs_direction == 2:
        y += 1
    else:
        y -= 1
    return x, y
    
  def __collision_with_self(self, x, y):
    if [x, y] in self.snake_position[1:-1]:
      return True
    else:
      return False

  def __collision_with_boundaries(self, x, y):
    if x < 0 or y < 0 or x >= self.field_width or y >= self.field_height:
      return True
    return False

  def __fill_observation(self):
    self.observation = [[0.0] * self.field_width for i in range(self.field_height)]
    # self.observation = [[0.0 for col in range(self.field_height)] for row in range(self.field_width)]
    self.observation[self.snake_start[0]][self.snake_start[1]] = 0.1
    for position in self.snake_position[1:]:
      self.observation[position[0]][position[1]] = 0.2
    self.observation[self.apple_position[0]][self.apple_position[1]] = 1.0
    # if self.verbose == 1:
    #   print('apple_position:', self.apple_position)
    #   print('observation apple_position:', self.observation[self.apple_position[0]][self.apple_position[1]])