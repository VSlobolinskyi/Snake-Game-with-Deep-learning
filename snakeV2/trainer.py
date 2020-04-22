from agent import PgAgent
from memory import Memory
import numpy as np
import time
import tensorflow as tf
import os

tf.compat.v1.enable_eager_execution()

from keras.models import load_model

tf.keras.backend.set_floatx('float64')

Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

class SnakeExecutor:
  def __init__(self, input_shape, output_size, weights_folder_suffix, weights_folder = None, verbose = 0):
    self.verbose = verbose
    if self.verbose == 1:
      print('Executor init')
    self.input_shape = input_shape
    self.output_size = output_size
    self.weights_folder = 'examples\output\{}_pg_{}_{}\weights'.format('v4', 'snake', weights_folder_suffix)
    if weights_folder != None:
      self.weights_folder = '{}_{}\weights'.format(weights_folder, weights_folder_suffix)
    self.agent = PgAgent(verbose)
    self.memory = Memory(verbose)
    self.model = self.__create_model()
    self.start_episode = 0
    self.__load_weights()
    self.agent.init_model(self.model, self.output_size)
    learning_rate=1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  def run_training(self, steps, saving_steps=10, init_function = None, step_function = None):
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

          if i_episode % saving_steps == saving_steps - 1:
            end_time = time.time()
            print('episode {}/{} score {} time {} records {}'.format(self.start_episode+i_episode+1, self.start_episode+steps, total_reward, \
              round(end_time - start_time, 2), len(self.memory.observations)))
            self.__save_waights()
            self.__save_eposode(self.start_episode+i_episode+1)
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

  def __create_model_v2(self):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=self.input_shape, filters=32, kernel_size=7, \
      strides=(4, 4), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, \
      strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(self.output_size, activation='linear'))
    return model

  def __create_model_v3(self):
    model.add(Conv2D(input_shape=self.input_shape, filters=16, kernel_size=7, \
      strides=(4, 4), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=5, \
      strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=3, \
      strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(self.output_size, activation='linear'))
    return model

  def __create_model_v4(self):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(input_shape=self.input_shape, filters=50, kernel_size=7, \
      strides=(2, 2), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(self.output_size, activation='linear'))
    return model
    
  def __create_model(self):
    return self.__create_model_v4()

  def __save_waights(self):
    if self.verbose == 1:
      print("Saving trained model weights")
    self.model.save_weights(self.weights_folder, overwrite=True)
    if self.verbose == 1:
      print("Model weights saved")
      
  def __load_weights(self):
    if os.path.isdir(os.path.dirname(self.weights_folder)):
      if self.verbose == 1:
        print("Loading trained model weights")
      self.model.load_weights(self.weights_folder)
      self.__load_episode()
      if self.verbose == 1:
        print("Model weights loaded")

  def __save_eposode(self, episode):
    with open('{}\episode.f'.format(os.path.dirname(self.weights_folder)), 'w+') as file:
      file.write(str(episode))

  def __load_episode(self):
    file_path = '{}\episode.f'.format(os.path.dirname(self.weights_folder))
    if os.path.isfile(file_path):
      with open(file_path, 'r') as file: 
        str_episode = file.read()
        self.start_episode = int(str_episode)