"""One-bit cue/query memory environment with literal dependency distance."""

import elements
import embodied
import numpy as np


class ToyMemory(embodied.Env):

  def __init__(self, task, distance=257, seed=0):
    if task != 'onebit':
      raise ValueError(f'Unknown ToyMemory task: {task}')
    if int(distance) < 1:
      raise ValueError('ToyMemory distance must be positive.')
    self.distance = int(distance)
    self.rng = np.random.default_rng(int(seed))
    self.cue = None
    self.time = None
    self.done = True

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, 2),
        'reset': elements.Space(bool),
    }

  @property
  def obs_space(self):
    return {
        'observation': elements.Space(np.float32, (4,), -1.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/correct': elements.Space(np.float32),
        'log/query': elements.Space(np.float32),
    }

  def step(self, action):
    if self.done or bool(action['reset']):
      self.cue = int(self.rng.integers(0, 2))
      self.time = 0
      self.done = False
      return self._obs(is_first=True)

    if self.time == self.distance:
      correct = int(action['action']) == self.cue
      self.done = True
      return self._obs(
          reward=1.0 if correct else -1.0,
          is_last=True,
          is_terminal=True,
          correct=float(correct))

    self.time += 1
    return self._obs()

  def _obs(
      self, reward=0.0, is_first=False, is_last=False, is_terminal=False,
      correct=0.0):
    cue = (2.0 * self.cue - 1.0) if self.time == 0 else 0.0
    query = float(self.time == self.distance and not is_last)
    progress = 2.0 * self.time / self.distance - 1.0
    observation = np.asarray([progress, cue, query, 1.0], np.float32)
    return {
        'observation': observation,
        'reward': np.float32(reward),
        'is_first': np.asarray(is_first, bool),
        'is_last': np.asarray(is_last, bool),
        'is_terminal': np.asarray(is_terminal, bool),
        'log/correct': np.float32(correct),
        'log/query': np.float32(query),
    }
