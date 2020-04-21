from agent import PgAgent
from memory import Memory
import numpy as np
import time
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from keras.models import load_model

tf.keras.backend.set_floatx('float64')

Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

class SnakeExecutor:
  def __init__(self, input_shape, output_size, verbose = 0):
    self.verbose = verbose
    if self.verbose == 1:
      print('Executor init')
    self.input_shape = input_shape
    self.output_size = output_size
    self.agent = PgAgent(verbose)
    self.memory = Memory(verbose)
    self.model = self.__create_model()
    self.agent.init_model(self.model, self.output_size)
    self.weights_folder = 'examples\output\{}_dqn_{}\weights'.format('v1', 'snake')
    learning_rate=1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  def run_training(self, steps, saving_steps=10 ,init_function = None, step_function = None):
    if init_function == None or step_function == None:
      return

    start_time = time.time()

    for i_episode in range(steps):
      observation = init_function()

      while True:
        action = self.agent.choose_action(observation)
        new_observation, reward, done, info = step_function(action)

        self.memory.add_to_memory(observation, action, reward)
        
        if done:
          total_reward = sum(self.memory.rewards)
          acts = self.memory.actions
          disc = self.agent.discount_rewards(self.memory.rewards)
          self.agent.train_step( 
            optimizer=self.optimizer, 
            observations=self.memory.observations, 
            actions=acts,
            discounted_rewards=disc)

          if i_episode % saving_steps == 0:
            end_time = time.time()
            print('episode {}/{} score {} time {} records {}'.format(i_episode, steps, total_reward, \
              round(end_time - start_time, 2), len(self.memory.observations)))
            start_time = time.time()

          self.memory.clear()
          break

        observation = new_observation

  def clone_model(self):
    model = self.__create_model()
    model.compile(self.optimizer, loss="mse", metrics = ['mse'])
    model_weights = self.model.get_weights()
    model.set_weights(model_weights)

    return model

  def __create_model(self):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=self.input_shape, filters=32, kernel_size=7, \
      strides=(4, 4), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=5, \
      strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, \
      strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(self.output_size, activation='linear'))
    return model