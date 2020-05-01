from random import randrange
class snake_env:
    # regime=0: environment returns negative reward on collision
    # regime=1: environment also sets done=True on collision
    def __init__(self, field_width=50, field_height=50, steps=10, apples_on_field=10, regime=0, growth=False):
        self.verbose = 1
        self.field_width = field_width
        self.field_height = field_height
        self.steps_in_game = steps
        self.steps_to_go = steps
        self.apples_to_allocate = apples_on_field
        self.apples_allocated = 0
        self.growth = growth
        self.regime = regime
        self.snake_positions = []
        self.tail = []
        self.apple_position = []
        self.info = []
        self.reward = 0.0
        self.done = False
        self.empty_cell = 0.0
        self.head = 0.1
        self.apple = 0.5
        self.body = 0.8
        self.wall = 0.9
        self.__allocate_static_elements()
        self.__allocate_snake()
        self.__allocate_apple()
    
    def step(self, action):
        self.reward = 0.0
        self.__move_snake(action)
        self.__get_reward()
        if self.steps_to_go == 0:
            self.done = True
        return self.done, self.reward, self.observation, self.info
        
    def __allocate_static_elements(self):
        self.observation = [[self.empty_cell] * self.field_width for i in range(self.field_height)]
        for i in range(self.field_width):
            self.observation[0][i] = self.wall
            self.observation[self.field_height-1][i] = self.wall
        for i in range(self.field_height):
            self.observation[i][0] = self.wall
            self.observation[i][self.field_width-1] = self.wall 
    
    def __allocate_snake(self):
        while True:
            snake_size = randrange(4, 20)
            if snake_size+2 <= self.field_width:
                break
        head_x = abs(randrange(1, self.field_width-1)-snake_size)+1
        head_y = abs(randrange(1, self.field_height-1)-snake_size)+1
        self.snake_positions.append([head_x, head_y])
        self.observation[head_y][head_x] = self.head
        for i in range(snake_size):
            self.snake_positions.append([head_x+i+1, head_y])
            self.observation[head_y][head_x+i+1] = self.body
        
    def __allocate_apple(self):
        while self.apples_allocated != self.apples_to_allocate:
            apple_x = randrange(0, 50)
            apple_y = randrange(0, 50)
            if self.observation[apple_y][apple_x] == self.empty_cell:
                self.apple_position = [apple_x, apple_y]
                self.observation[apple_y][apple_x] = self.apple
                self.apples_allocated += 1
          
    # 0 - top, 1 -right, 2 - bottom, 3 - left
    def __move_snake(self, action):
        if action == 0:
            move_x = 0
            move_y = -1
        if action == 1:
            move_x = 1
            move_y = 0
        if action == 2:
            move_x = 0
            move_y = 1
        if action == 3:
            move_x = -1
            move_y = 0
        head = self.snake_positions[0]
        self.snake_positions.insert(1, [head[0], head[1]])
        head[0] += move_x
        head[1] += move_y
        self.tail = [self.snake_positions[-1][0], self.snake_positions[-1][1]]
        if not (self.growth and self.observation[head[1]][head[0]] == self.apple):
            self.snake_positions.pop()

        if self.tail[0] == 49 or self.tail[0] == 0 or self.tail[1] == 49 or self.tail[1] == 0:
            self.observation[self.tail[1]][self.tail[0]] = self.wall
        else:
            self.observation[self.tail[1]][self.tail[0]] = self.empty_cell
        skip_step = True
        for i in self.snake_positions:
            if skip_step:
                skip_step = False
                continue
            self.observation[i[1]][i[0]] = self.body
            
        self.observation[self.snake_positions[1][1]][self.snake_positions[1][0]] = self.body
        
        self.steps_to_go -= 1
        
    def __get_reward(self):
        snake_head = self.snake_positions[0]
        if self.observation[snake_head[1]][snake_head[0]] == self.wall or self.observation[snake_head[1]][snake_head[0]] == self.body:
            self.reward = -0.3
            if self.regime == 1:
                self.done = True
        elif self.observation[snake_head[1]][snake_head[0]] == self.apple:
            self.reward = 0.1
            self.apples_allocated -= 1
            self.__allocate_apple()
        self.observation[snake_head[1]][snake_head[0]] = self.head
            
    def reset(self):
        self.reward = 0.0
        self.__allocate_static_elements()
        self.__allocate_snake()
        self.__allocate_apple()
        self.steps_to_go = self.steps_in_game
        self.done = False
        return self.observation

    


    
        