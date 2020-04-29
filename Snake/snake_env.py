from random import randrange
class snake_env:
    def __init__(self):
        self.verbose = 1
        self.field_width = 50
        self.field_height = 50
        self.snake_positions = []
        self.apple_position = []
        
    def reset(self):
        self.__allocate_static_elements()
        self.__allocate_snake()
        self.__allocate_apple()
        self.info = []
        self.positive_reward = 0.0
        self.negative_reward = 0.0
        self.iteration = 0
        self.steps_to_go = 0
        return self.observation
    
    def step(self, action):
        # self.steps_to_go -= 1
        # # Snake will not be realocated when colliding with itself but will receive significant negative reward, snake will be realocated when colliding with walls
        # if steps_to_go == 0 or self.observation[snake_head[1]][snake_head[0]] == 9:
        #     self.__alocate_static_elements()
        #     self.__allocate_snake
        #     self.__allocate_apple
        # if steps_to_go == 0 or self.observation[snake_head[1]][snake_head[0]] != 1:
        #     observation[snake_head[1]][snake_head[0]] = 1
        # if steps_to_go == 0 or self.observation[snake_head[1]][snake_head[0]] != 1:
        #     observation[snake_head[1]][snake_head[0]] = 1
        return self.done, self.reward, self.observation, self.info
        
    def __allocate_static_elements(self):
        self.observation = [[0] * self.field_width for i in range(self.field_height)]
        for i in range(self.field_width):
            self.observation[0][i] = 9
            self.observation[self.field_height-1][i] = 9
        for i in range(self.field_height):
            self.observation[i][0] = 9
            self.observation[i][self.field_width-1] = 9   
    
    def __allocate_snake(self):
        head_x = randrange(9, 30)
        head_y = randrange(9, 30)
        snake_size = randrange(4, 20)
        self.snake_positions.append([head_x, head_y])
        self.observation[head_y][head_x] = 1
        for i in range(snake_size):
            self.snake_positions.append([head_x+i+1, head_y])
            self.observation[head_y][head_x+i+1] = 8
        
    def __allocate_apple(self):
        while True:
            apple_x = randrange(9, 30)
            apple_y = randrange(9, 30)
            if self.observation[apple_y][apple_x] == 0:
                self.apple_position = [apple_x, apple_y]
                self.observation[apple_y][apple_x] = 5
                break

snake_env = snake_env()
for i in range(10):
    obs = snake_env.reset()
    print()
    print("HERE IS OBSERVATION MATRIX",i)
    for i in obs:
        print(i)
        