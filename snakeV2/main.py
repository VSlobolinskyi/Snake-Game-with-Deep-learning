from game import Env
from display import Render
from trainer import SnakeExecutor
import numpy as np
import time

env = Env(0)
input_size = env.get_observation_space()
output_size = env.get_action_space_count()
executor = SnakeExecutor(input_size, output_size, str(env.complexity))

def env_init():
  env.seed(1)
  observation = env.reset()
  observation = np.reshape(observation, (50, 50, 1))
  return observation

def env_step(action):
  observation, reward, done, info = env.step(action)
  observation = np.reshape(observation, (50, 50, 1))

  return observation, reward, done, info

start_time = time.time()
executor.run_training(50, init_function=env_init, step_function=env_step)
end_time = time.time()
print('training time {}'.format(round(end_time - start_time, 2)))

new_model = executor.clone_model()

def make_step(prev_observation):
  if prev_observation == None:
    prev_observation = env.reset()
  observation = np.reshape(prev_observation, (50, 50, 1))
  observation = np.expand_dims(observation, axis=0)
  action = np.argmax(new_model.predict(observation))
  observation, reward, done, info = env.step(action)
  return observation

render = Render(env, make_step, speed=50, steps=500)
