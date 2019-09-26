from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict


# def _get_default_config():
#   checkpoint_dict = dict()
#   checkpoint_dict[1] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint/' + \
#                        'resnet18_general_site_1_117_val_accuracy=0.3217415.pth'
#   checkpoint_dict[2] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_densenet121/' + \
#                        'densenet121_general_site_1_77_val_accuracy=0.4011501.pth'
#   checkpoint_dict[3] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_densenet121_focal/' + \
#                        'densenet121_general_site_1_66_val_accuracy=0.4329135.pth'
#
#   model_dict = dict()
#   model_dict[1] = 'resnet18'
#   model_dict[2] = 'densenet121'
#
#   exp_dict = dict()
#   exp_dict[1] = 'HEPG2'
#   exp_dict[2] = 'HUVEC'
#   exp_dict[3] = 'RPE'
#   exp_dict[4] = 'U2OS'
#
#   config = edict()
#   config.checkpoint_folder = 'densenet121_focal'
#
#   config.model = model_dict[2]
#   config.site = 1  # or 2
#   config.all = False  # or False
#   config.experiment = exp_dict[4]
#   config.checkpoint_name = checkpoint_dict[3]
#   config.warm_start = True
#   config.random_seed = 42
#   config.batch_size = 12
#   return config


def _get_default_config():
  checkpoint_dict = dict()
  checkpoint_dict[1] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint/' + \
                       'resnet18_general_site_1_117_val_accuracy=0.3217415.pth'
  checkpoint_dict[2] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_densenet121/' + \
                       'densenet121_general_site_2_74_val_accuracy=0.4164841.pth'
  checkpoint_dict[3] = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_densenet121_focal/' + \
                       'densenet121_general_site_2_67_val_accuracy=0.4394852.pth'

  model_dict = dict()
  model_dict[1] = 'resnet18'
  model_dict[2] = 'densenet121'

  exp_dict = dict()
  exp_dict[1] = 'HEPG2'
  exp_dict[2] = 'HUVEC'
  exp_dict[3] = 'RPE'
  exp_dict[4] = 'U2OS'

  config = edict()
  config.checkpoint_folder = 'densenet121_focal'
  config.model = model_dict[2]
  config.site = 2  # or 2k
  config.all = False  # or False
  config.experiment = exp_dict[4]
  config.checkpoint_name = checkpoint_dict[3]
  config.warm_start = True
  config.random_seed = 24
  config.batch_size = 12
  return config
