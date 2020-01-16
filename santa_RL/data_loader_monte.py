import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

path_data = ''
BATCH_SIZE = 1


class FamilyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.df = pd.read_csv(self.root + 'family_data_standard_scaled.csv',)
        self.len = self.df.shape[0]

    def __getitem__(self, index):
        return torch.Tensor(self.df.iloc[index].values[1:]), self.df.iloc[index].values[0]#.unsqueeze(0)

    def __len__(self):
        return self.len


train_data = FamilyDataset(path_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
imgs = next(iter(train_loader))

