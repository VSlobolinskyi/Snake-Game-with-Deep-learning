class Memory:
  def __init__(self, verbose = 0):
    self.verbose = verbose
    if self.verbose == 1:
      print('Memory init')
    self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
    if self.verbose == 1:
      print('Memory clear')
    self.observations = []
    self.actions = []
    self.rewards = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward): 
    if self.verbose == 1:
      print('Memory add_to_memory')
    self.observations.append(new_observation)
    self.actions.append(new_action)
    self.rewards.append(new_reward)