import tensorflow as tf
import numpy as np
import gym
tf.compat.v1.enable_eager_execution()


def choose_action(model, observation):
  # add batch dimension to the observation
  observation = np.expand_dims(observation, axis=0)

  '''TODO: feed the observations through the model to predict the log probabilities of each possible action.'''
  logits = model.predict(observation) # TODO
  # logits = model.predict('''TODO''')
  
  # pass the log probabilities through a softmax to compute true probabilities
  prob_weights = tf.nn.softmax(logits).numpy()
  
  '''TODO: randomly sample from the prob_weights to pick an action.
  Hint: carefully consider the dimensionality of the input probabilities (vector) and the output action (scalar)'''
  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0] # TODO
  # action = np.random.choice('''TODO''', size=1, p=''''TODO''')['''TODO''']

  return action


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
        
memory = Memory()


def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
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
      # update the total discounted reward
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
      
  return normalize(discounted_rewards)


def compute_loss(logits, actions, rewards): 
  '''TODO: complete the function call to compute the negative log probabilities'''
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions) # TODO
  # neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits='''TODO''', labels='''TODO''')
  
  '''TODO: scale the negative log probability by the rewards'''
  loss = tf.reduce_mean( neg_logprob * rewards ) # TODO
  # loss = tf.reduce_mean('''TODO''')
  return loss


def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      logits = model(observations)

      '''TODO: call the compute_loss function to compute the loss'''
      loss = compute_loss(logits, actions, discounted_rewards) # TODO
      # loss = compute_loss('''TODO''', '''TODO''', '''TODO''')

  '''TODO: run backpropagation to minimize the loss using the tape.gradient method'''
  grads = tape.gradient(loss, model.trainable_variables) # TODO
  # grads = tape.gradient('''TODO''', model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  
class LossHistory:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss

  
env = gym.make("CartPole-v1")
env.seed(1)
n_observations = env.observation_space
print("Environment has observation space =", n_observations)
n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)

def create_cartpole_model():
  model = tf.keras.models.Sequential([
      # First Dense layer
      tf.keras.layers.Dense(units=32, activation='relu'),

      # TODO: Define the last Dense layer, which will provide the network's output.
      # Think about the space the agent needs to act in!
      tf.keras.layers.Dense(units=n_actions, activation=None) # TODO
      # [TODO Dense layer to output action probabilities]
  ])
  return model
cartpole_model = create_cartpole_model()


# Learning rate and optimizer
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# instantiate cartpole agent
cartpole_model = create_cartpole_model()

# to track our progress
smoothed_reward = LossHistory(smoothing_factor=0.9)

for i_episode in range(500):
  # Restart the environment
  observation = env.reset()
  memory.clear()

  while True:
      # using our observation, choose an action and take it in the environment
      action = choose_action(cartpole_model, observation)
      next_observation, reward, done, info = env.step(action)
      # add to memory
      memory.add_to_memory(observation, action, reward)
      
      # is the episode over? did you crash or do so well that you're done?
      if done:
          # determine total reward and keep a record of this
          total_reward = sum(memory.rewards)
          print(total_reward)
          smoothed_reward.append(total_reward)
          
          # initiate training - remember we don't know anything about how the 
          #   agent is doing until it has crashed!
          train_step(cartpole_model, optimizer, 
                      observations=np.vstack(memory.observations),
                      actions=np.array(memory.actions),
                      discounted_rewards = discount_rewards(memory.rewards))
          
          # reset the memory
          memory.clear()
          break
      # update our observatons
      observation = next_observation
      

for i_episode in range(10):
  observation = env.reset()
  memory.clear()
  while True:
      env.render()
      observation = np.expand_dims(observation, axis=0)
      action = np.argmax(cartpole_model.predict(observation))
      observation, _, done, _ = env.step(action)
      if done:
          break


sess = tf.Session()
print(sess)
