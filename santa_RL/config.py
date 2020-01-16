from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict


def _get_default_config():
  config = edict()
  config.N_ACTIONS = 6
  config.MAX_CAPACITY = 280.  # max number of people per day
  config.DAYS_OF_MAX_CAPACITY = 25
  config.ADDITIONAL_REWARD = 0
  config.REWARD_SCALE = 1.
  config.n_neurons = 4096
  config.episdodes_monte = 200
  config.batch_size = 128
  config.gamma = 0.99
  return config
