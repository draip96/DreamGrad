import embodied
import numpy as np


class BSuite(embodied.Env):

  def __init__(self, task, memory_length=None, seed=0):
    np.int = int  # Patch deprecated Numpy alias used inside BSuite.
    from . import from_dm
    if task == 'memory':
      if memory_length is None or int(memory_length) < 1:
        raise ValueError('BSuite MemoryChain requires memory_length >= 1.')
      from bsuite.experiments.memory_len import memory_len
      env = memory_len.load(memory_length=int(memory_length), seed=int(seed))
      print(
          'Created official BSuite MemoryChain with literal memory_length '
          f'{int(memory_length)} and seed {int(seed)}.')
    else:
      if '/' not in task:
        task = f'{task}/0'
      import bsuite
      env = bsuite.load_from_id(task)
    self.num_episodes = 0
    self.max_episodes = env.bsuite_num_episodes
    env = from_dm.FromDM(env)
    env = embodied.wrappers.FlattenTwoDimObs(env)
    self.env = env

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_last']:
      self.num_episodes += 1
    return obs
