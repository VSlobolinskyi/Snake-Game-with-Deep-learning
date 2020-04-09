import tensorflow as tf

import numpy as np
import base64, io, time, gym
import time
import os

tf.compat.v1.enable_eager_execution()

from keras.models import load_model

tf.keras.backend.set_floatx('float64')

runCartPole = False
runPong = True

Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

### Define the agent's action function ###

# Function that takes observations as input, executes a forward pass through model, 
#   and outputs a sampled action.
# Arguments:
#   model: the network that defines our agent
#   observation: observation which is fed as input to the model
# Returns:
#   action: choice of agent action
def choose_action(model, observation):
  # add batch dimension to the observation
  observation = np.expand_dims(observation, axis=0)

  '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''
  # print('observation', observation)
  logits = model.predict(observation) # TODO
  # logits = model.predict('''TODO''')
  # print('logits', logits)
  
  # pass the log probabilities through a softmax to compute true probabilities
  prob_weights = tf.nn.softmax(logits).numpy()
  # print('prob_weights', prob_weights)
  
  '''TODO: randomly sample from the prob_weights to pick an action.
  Hint: carefully consider the dimensionality of the input probabilities (vector) and the output action (scalar)'''
  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0] # TODO
  # action = np.random.choice('''TODO''', size=1, p=''''TODO''')['''TODO''']

  return action

### Agent Memory ###

class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward): 
      self.observations.append(new_observation)
      '''TODO: update the list of actions with new action'''
      self.actions.append(new_action) # TODO
      # ['''TODO''']
      '''TODO: update the list of rewards with new reward'''
      self.rewards.append(new_reward) # TODO
      # ['''TODO''']

### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
  # print('normalize x', x)
  x -= np.mean(x)
  std = np.std(x)
  if std == 0.0:
    std = 1.0
  x /= std
  return x.astype(np.float32)

# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.95): 
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      # NEW: Reset the sum if the reward is not 0 (the game has ended!)
      if rewards[t] != 0:
        R = 0
      # update the total discounted reward
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
      
  return normalize(discounted_rewards)

### Loss function ###

# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards): 
  '''TODO: complete the function call to compute the negative log probabilities'''
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions) # TODO
  # neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits='''TODO''', labels='''TODO''')
  
  '''TODO: scale the negative log probability by the rewards'''
  loss = tf.reduce_mean( neg_logprob * rewards ) # TODO
  # loss = tf.reduce_mean('''TODO''')
  return loss

### Training step (forward and backpropagation) ###

def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      # print('observations', observations.shape)
      logits = model(observations)
      # print('logits', logits)

      '''TODO: call the compute_loss function to compute the loss'''
      loss = compute_loss(logits, actions, discounted_rewards) # TODO
      # loss = compute_loss('''TODO''', '''TODO''', '''TODO''')

  '''TODO: run backpropagation to minimize the loss using the tape.gradient method'''
  grads = tape.gradient(loss, model.trainable_variables) # TODO
  # grads = tape.gradient('''TODO''', model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
def save_weights(new_model, trained_model, folder):
  trained_model.save_weights(folder, overwrite=True)
  model_weights = trained_model.get_weights()
  new_model.set_weights(model_weights)
  return new_model
  
def load_weights(folder, model):
  if os.path.isdir(os.path.dirname(folder)):
      print("Loading trained model weights")
      model.load_weights(folder)

def test_model(test_env, transform_observation, new_model, sleep):
  
  for e in range(20):
    current_frame = test_env.reset()
    previous_frame = current_frame

    done = False
    i = 0
    summ_reward = 0.0
    while True:
        test_env.render()
        observation = transform_observation(current_frame, previous_frame)
        observation = np.expand_dims(observation, axis=0)
        action = np.argmax(new_model.predict(observation))

        previous_frame = current_frame
        current_frame, reward, done, _ = test_env.step(action)
        summ_reward += reward
        i += 1
        if done:
            print("episode: {}/{}, score: {}".format(e, 20, summ_reward))
            break
        time.sleep(sleep)

  return new_model

"""# Cart Pole"""

### Define the Cartpole agent ###

# Defines a feed-forward neural network
def create_cartpole_model(input_size, output_size):
  model = tf.keras.models.Sequential()
  model.add(Dense(128, input_dim=input_size, activation='relu'))
  model.add(Dense(52, activation='relu'))
  model.add(Dense(output_size, activation='linear'))

  return model

def prapare_cart_pole_observation(current_frame, previous_frame):
    # observation = np.expand_dims(current_frame, axis=0)
    return current_frame

if runCartPole:
  ENV_NAME = "CartPole-v1"
  env = gym.make(ENV_NAME)
  env.seed(1)

  n_observations = env.observation_space
  print("Environment has observation space =", n_observations)

  n_actions = env.action_space.n
  print("Number of possible actions that the agent can choose from =", n_actions)

  ### Cartpole training! ###

  # Learning rate and optimizer
  learning_rate = 1e-3
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  memory = Memory()

  # instantiate cartpole agent
  cartpole_model = create_cartpole_model(env.observation_space.shape[0], n_actions)

  MAX_ITERS = 500

  for i_episode in range(MAX_ITERS):

    # Restart the environment
    current_frame = env.reset()
    previous_frame = current_frame
    memory.clear()

    while True:
        # using our observation, choose an action and take it in the environment
        observation = prapare_cart_pole_observation(current_frame, previous_frame)
        action = choose_action(cartpole_model, observation)
        previous_frame = current_frame
        current_frame, reward, done, _ = env.step(action)
        # add to memory
        memory.add_to_memory(observation, action, reward)
        
        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            print('episode:', i_episode, 'reward:', total_reward)
            
            # initiate training - remember we don't know anything about how the 
            #   agent is doing until it has crashed!
            train_step(cartpole_model, optimizer, 
                      observations=np.vstack(memory.observations),
                      actions=np.array(memory.actions),
                      discounted_rewards = discount_rewards(memory.rewards))
            
            # reset the memory
            memory.clear()
            break

  # Test
  # cartpole_model.summary()
  new_model = create_cartpole_model(env.observation_space.shape[0], n_actions)
  
  # print('model input shape:', trained_model.layers[0].input_shape)
  new_model.compile(optimizer, loss="mse", metrics = ['mse'])
  new_model = save_weights(new_model, cartpole_model, ENV_NAME)
  # new_model.summary()

  test_model(env, prapare_cart_pole_observation, new_model)

"""# PONG"""

### Define the Pong agent ###

# Defines a CNN for the Pong agent
def create_pong_model(input_shape, output_size):
  print('input_shape', input_shape)
  model = tf.keras.models.Sequential()
  model.add(Conv2D(input_shape=input_shape, filters=16, kernel_size=7, \
    strides=(4, 4), activation='relu'))
  model.add(Conv2D(filters=32, kernel_size=5, \
    strides=(2, 2), activation='relu'))
  model.add(Conv2D(filters=48, kernel_size=3, \
    strides=(2, 2), activation='relu'))
  model.add(Flatten())
  model.add(Dense(output_size, activation='linear'))

  return model

def preprocess_pong(image):
    I = image[35:195] # Crop
    I = I[::2, ::2, 0] # Downsample width and height by a factor of 2
    I[I == 144] = 0 # Remove background type 1
    I[I == 109] = 0 # Remove background type 2
    I[I != 0] = 1 # Set remaining elements (paddles, ball, etc.) to 1
    return I.astype(np.float).reshape(80, 80, 1)

def prapare_pong_observation(current_frame, previous_frame):
    cur = preprocess_pong(current_frame)
    prev = preprocess_pong(previous_frame)
    obs_change = cur - prev
    return obs_change

if runPong:
  ENV_NAME = "Pong-v0"
  env = gym.make(ENV_NAME, frameskip=5)
  env.seed(1); # for reproducibility

  print("Environment has observation space =", env.observation_space)

  n_actions = env.action_space.n
  print("Number of possible actions that the agent can choose from =", n_actions)

  ### Training Pong ###

  # Hyperparameters
  MAX_ITERS = 100000 # increase the maximum number of episodes, since Pong is more complex!

  pong_shape = (80, 80, 1)

  # Model and optimizer
  pong_model = create_pong_model(pong_shape, n_actions)

  pong_weights_folder = 'examples\output\{}_dqn_{}\weights'.format('v8', ENV_NAME)
  load_weights(pong_weights_folder, pong_model)

  learning_rate=1e-4
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  memory = Memory()

  for i_episode in range(MAX_ITERS):

    # Restart the environment
    observation = env.reset()
    previous_frame = preprocess_pong(observation)
    start_time = time.time()

    while True:
        # Pre-process image 
        current_frame = preprocess_pong(observation)
        
        '''TODO: determine the observation change
        Hint: this is the difference between the past two frames'''
        obs_change = current_frame - previous_frame # TODO
        # obs_change = # TODO
        
        '''TODO: choose an action for the pong model, using the frame difference, and evaluate'''
        action = choose_action(pong_model, obs_change) # TODO 
        # action = # TODO
        # Take the chosen action
        next_observation, reward, done, info = env.step(action)

        '''TODO: save the observed frame difference, the action that was taken, and the resulting reward!'''
        memory.add_to_memory(obs_change, action, reward) # TODO
        
        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            end_time = time.time()
            # print('memory.observations', np.stack(memory.observations, 0).shape)
            # print('observation', np.stack(memory.observations, 0)[0])

            # begin training
            train_step(pong_model, 
                      optimizer, 
                      observations = np.stack(memory.observations, 0), 
                      actions = np.array(memory.actions),
                      discounted_rewards = discount_rewards(memory.rewards))
            
            print('episode {}/{} score {} time {} records {}'.format(i_episode, MAX_ITERS, total_reward, round(end_time - start_time, 2), len(memory.observations)))
            if i_episode % 5 == 0:
              print("Saving trained model weights")
              pong_model.save_weights(pong_weights_folder, overwrite=True)
              print("Model weights saved")

            memory.clear()
            break

        observation = next_observation
        previous_frame = current_frame

  # Test
  new_model = create_pong_model(pong_shape, n_actions)
  new_model.compile(optimizer, loss="mse", metrics = ['mse'])
  new_model = save_weights(new_model, pong_model, pong_weights_folder)

  test_model(env, prapare_pong_observation, new_model, 0.02)
