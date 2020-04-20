from game import Env
from memory import Memory
from display import Render
import time

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
  if reward != 0.0:
    print(reward)
  if done:
    print('done in', k, 'episodes')
    break

end_time = time.time()
print('time {}'.format(round(end_time - start_time, 2)))

def make_step():
  observation, reward, done, info = env.step(env.get_action_space_sample())
  return observation

obs = env.reset()
render = Render(env, make_step)
