import torch
from data_loader import tloader
from tqdm import tqdm
import pandas as pd
import numpy as np
from model_k import model_resnet_18
import os
from config import _get_default_config

config = _get_default_config()
MODEL_NAME = config.model

# models_name = \
#     [['resnet18_HEPG2_site_1_49_val_accuracy=0.1797468.pth', 'resnet18_HEPG2_site_2_74_val_accuracy=0.1959545.pth'],
#      ['resnet18_HUVEC_site_1_86_val_accuracy=0.575.pth', 'resnet18_HUVEC_site_2_91_val_accuracy=0.5804118.pth'],
#      ['resnet18_RPE_site_1_150_val_accuracy=0.2769424.pth', 'resnet18_RPE_site_2_47_val_accuracy=0.2866579.pth'],
#      ['resnet18_U2OS_site_1_64_val_accuracy=0.07565789.pth', 'resnet18_U2OS_site_2_50_val_accuracy=0.09120521.pth']]

# models_name = \
#     [['densenet121_HEPG2_site_1_103_val_accuracy=0.3316456.pth', 'densenet121_HEPG2_site_2_27_val_accuracy=0.3198483.pth'],
#      ['densenet121_HUVEC_site_1_72_val_accuracy=0.6909091.pth', 'densenet121_HUVEC_site_2_120_val_accuracy=0.7067334.pth'],
#      ['densenet121_RPE_site_1_66_val_accuracy=0.4235589.pth', 'densenet121_RPE_site_2_92_val_accuracy=0.4504624.pth'],
#      ['densenet121_U2OS_site_1_75_val_accuracy=0.1151316.pth', 'densenet121_U2OS_site_2_75_val_accuracy=0.1433225.pth']]


models_name = \
    [['densenet121_HEPG2_site_1_22_val_accuracy=0.3101266.pth', 'densenet121_HEPG2_site_2_93_val_accuracy=0.335019.pth'],
     ['densenet121_HUVEC_site_1_49_val_accuracy=0.6846591.pth', 'densenet121_HUVEC_site_2_87_val_accuracy=0.6956038.pth'],
     ['densenet121_RPE_site_1_47_val_accuracy=0.3934837.pth', 'densenet121_RPE_site_2_85_val_accuracy=0.4319683.pth'],
     ['densenet121_U2OS_site_1_54_val_accuracy=0.1085526.pth', 'densenet121_U2OS_site_2_55_val_accuracy=0.1465798.pth']]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_{}'.format(config.checkpoint_folder)
path_data = '/mnt/ssd1/datasets/Recursion_class/'
device = 'cuda'
model_1 = model_resnet_18
model_2 = model_resnet_18

preds_list = []
for checkpoint_name in models_name:
    checkpoint_1 = torch.load(path + '/' + checkpoint_name[0])
    checkpoint_2 = torch.load(path + '/' + checkpoint_name[1])
    model_1.load_state_dict(checkpoint_1)
    model_1.to(device)
    model_2.load_state_dict(checkpoint_2)
    model_2.to(device)
    with torch.no_grad():
        preds = np.empty(0)
        for x, _ in tqdm(tloader):
            x = x.to(device)
            output_1 = model_1(x)
            output_2 = model_2(x)
            output = output_1 + output_2
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)

    preds_list.append(preds)


submission = pd.read_csv(path_data + '/test.csv')
final_results = []

for n, experiment in enumerate(submission.experiment.values):
    if 'HEPG2' in experiment:
        final_results.append(preds_list[0][n])
    elif 'HUVEC' in experiment:
        final_results.append(preds_list[1][n])
    elif 'RPE' in experiment:
        final_results.append(preds_list[2][n])
    elif 'U2OS' in experiment:
        final_results.append(preds_list[3][n])


submission['sirna'] = np.array(final_results).astype(int)
submission.to_csv('submission_{}.csv'.format('multi_8_{}'.format(config.model)), index=False, columns=['id_code', 'sirna'])


