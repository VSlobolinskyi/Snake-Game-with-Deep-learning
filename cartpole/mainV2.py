import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

# from gym.wrappers.monitoring.video_recorder import VideoRecorder # Because we want to record a video

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
print('States', states)
actions = env.action_space.n
print('Actions', actions)

episodes = 10

# rec = VideoRecorder(env)      # Create the video recorder
# rec.capture_frame()           # Capture the starting position

for episode in range(1,episodes+1):
    # At each begining reset the game 
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:
        # visualize each step
        env.render()
        # choose a random action
        action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
        # rec.capture_frame() # and record
    print('episode {} score {}'.format(episode, score))

def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
  
model = agent(env.observation_space.shape[0], env.action_space.n)

policy = EpsGreedyQPolicy()
sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)
sarsa.compile('adam', metrics = ['mse'])
sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
scores = sarsa.test(env, nb_episodes = 100, visualize= False)
print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))

sarsa.save_weights('cartpole\sarsa_weights.h5f', overwrite=True)
sarsa.load_weights('cartpole\sarsa_weights.h5f')

_ = sarsa.test(env, nb_episodes = 2, visualize= True)

# rec.close()  # Close the recording
env.close()  # Close the environment
