from random import randrange
class snake_env:
    def __init__(self, steps=10, complexity=0):
        self.verbose = 1
        self.field_width = 50
        self.field_height = 50
        self.snake_positions = []
        self.apple_position = []
        self.steps_in_game = 10
        self.steps_to_go = 10
        self.iteration = 0
        self.info = []
        self.reward = 0.0
        self.done = True
    
    def step(self, action):
        self.reward = 0.0
        if self.done:
            self.__reset()
        else:
            self.__move_snake(action)
            self.__get_reward()
            if self.steps_to_go == 0:
                self.done = True
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
            apple_x = randrange(0, 50)
            apple_y = randrange(0, 50)
            if self.observation[apple_y][apple_x] == 0:
                self.apple_position = [apple_x, apple_y]
                self.observation[apple_y][apple_x] = 5
                break
            
    def __move_snake(self, action):
        if action == 0:
            move_x = 0
            move_y = -1
        elif action == 1:
            move_x = 1
            move_y = 0
        elif action == 2:
            move_x = 0
            move_y = 1
        elif action == 3:
            move_x = -1
            move_y = 0  
        elem_switch_x = None
        elem_switch_y = None
        for i in self.snake_positions:
            if elem_switch_x == None or elem_switch_y == None:
                elem_switch_x = [i[0], None]
                elem_switch_y = [i[1], None]
                i[0] = (i[0]+move_x)%50
                i[1] = (i[1]+move_y)%50
            else:
                elem_switch_x[1] = i[0]
                elem_switch_y[1] = i[1]
                i[0] = elem_switch_x[0]
                i[1] = elem_switch_y[0]
                elem_switch_x[0] = elem_switch_x[1]
                elem_switch_y[0] = elem_switch_y[1]
        head_x = self.snake_positions[0][0]
        head_y = self.snake_positions[0][1]
        tail_x = self.snake_positions[-1][0]
        tail_y = self.snake_positions[-1][1]
        self.observation[head_y][head_x] = 1
        self.observation[head_y-move_y][head_x-move_x] = 8
        if elem_switch_x[0] == 49 or elem_switch_x[0] == 0 or elem_switch_y[0] == 49 or elem_switch_y[0] == 0:
            self.observation[elem_switch_y[0]][elem_switch_x[0]] = 9
        else:
            self.observation[elem_switch_y[0]][elem_switch_x[0]] = 0
        self.steps_to_go -= 1
        
        
    def __get_reward(self):
        snake_head = self.snake_positions[0]
        if self.observation[snake_head[1]][snake_head[0]] == 9 or self.observation[snake_head[1]][snake_head[0]] == 8:
            self.negative_reward = 5
        elif self.observation[snake_head[1]][snake_head[0]] == -3:
            self.positive_reward = 1
        if self.steps_to_go == 0 or self.observation[snake_head[1]][snake_head[0]] != 1:
            self.observation[snake_head[1]][snake_head[0]] = 1
            
    def __reset(self):
        self.__allocate_static_elements()
        self.__allocate_snake()
        self.__allocate_apple()
        self.iteration += 1
        self.steps_to_go = self.steps_in_game
        self.done = False

snake_env = snake_env()
for i in range(11):
    done, reward, obs, info = snake_env.step(randrange(0,4))
    print()
    print("HERE IS OBSERVATION MATRIX",i)
    for i in obs:
        print(i)

    
        