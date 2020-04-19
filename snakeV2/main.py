from game import Env
from memory import Memory
from display import Render
import time
import pygame

env = Env(0)
mem = Memory(0)
# rend = Render(1)

def print_observation(env, obs):
  for i in range(env.field_height):
    for j in range(env.field_width):
      if obs[j][i] > 0.0:
        print('x:',j,'y:',i,'value:',obs[j][i])

env.seed(1)
obs = env.reset()

start_time = time.time()
for k in range(6000):
  # print('===================')
  observation, reward, done, info = env.step(env.get_action_space_sample())
  # print_observation(env, observation)
  # time.sleep(1)
  # rend.draw_step(observation, env)

end_time = time.time()
print('time {}'.format(round(end_time - start_time, 2)))