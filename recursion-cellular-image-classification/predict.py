import torch
from data_loader import tloader
from tqdm import tqdm
import pandas as pd
import numpy as np
from model_k import model_resnet_18
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
path = '/mnt/ssd1/datasets/Recursion_class/recurs_proj/checkpoint'
path_data = '/mnt/ssd1/datasets/Recursion_class/'
device = 'cuda'
model = model_resnet_18
checkpoint_name = 'resnet18_general_site_1_107_val_accuracy=0.237779.pth'

checkpoint = torch.load(path + '/' + checkpoint_name)
model.load_state_dict(checkpoint)
model.to(device)
with torch.no_grad():
    preds = np.empty(0)
    for x, _ in tqdm(tloader):
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        preds = np.append(preds, idx, axis=0)


submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission_{}.csv'.format(checkpoint_name), index=False, columns=['id_code', 'sirna'])


