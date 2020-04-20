from game import Env
from display import Render
from trainer import SnakeExecutor
import numpy as np
import time

env = Env(0)
input_size = env.get_observation_space()
output_size = env.get_action_space_count()
executor = SnakeExecutor(input_size, output_size)

def env_init():
  env.seed(1)
  observation = env.reset()
  observation = np.reshape(observation, (50, 50, 1))
  return observation

def env_step(action):
  observation, reward, done, info = env.step(action)
  observation = np.reshape(observation, (50, 50, 1))
  # if reward != 0.0:
  #   print('reward: {}, negative_reward: {}, done: {}'.format(reward, env.negative_reward, done))

  return observation, reward, done, info

start_time = time.time()
executor.run_training(10000, init_function=env_init, step_function=env_step)
end_time = time.time()
print('time {}'.format(round(end_time - start_time, 2)))

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
render = Render(env, make_step, speed=100)
