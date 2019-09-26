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

checkpoint_name_1 = 'densenet121_general_site_1_77_val_accuracy=0.4011501.pth'
checkpoint_name_2 = 'densenet121_general_site_2_74_val_accuracy=0.4164841.pth'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint_{}'.format(MODEL_NAME)
path_data = '/mnt/ssd1/datasets/Recursion_class/'
device = 'cuda'
model_1 = model_resnet_18
model_2 = model_resnet_18


checkpoint = torch.load(path + '/' + checkpoint_name_1)
model_1.load_state_dict(checkpoint)
model_1.to(device)

checkpoint = torch.load(path + '/' + checkpoint_name_2)
model_2.load_state_dict(checkpoint)
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


submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission_{}.csv'.format('dense_mix_2_2_sites'), index=False, columns=['id_code', 'sirna'])


