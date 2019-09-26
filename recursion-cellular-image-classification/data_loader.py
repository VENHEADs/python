import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms as T
import torch.utils.data as D
import torch
from config import _get_default_config
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, RandomCrop, RandomSizedCrop, CenterCrop
)

path_data = '/mnt/ssd1/datasets/Recursion_class'
config = _get_default_config()
BATCH_SIZE = config.batch_size


def strong_aug(p=.3):
    return Compose([
        # RandomRotate90(),
        Flip(),  # good
        # Transpose(),
        # OneOf([
        #    IAAAdditiveGaussianNoise(),
        #    GaussNoise(),
        # ], p=0.2),
        # OneOf([
        #     RandomCrop(384, 384),
        # ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=90, p=0.2),  # good
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomBrightnessContrast(),
        # ], p=0.3),
    ], p=p)


aug = strong_aug()

transform = T.Compose([
                       T.Normalize((5.845, 15.567, 10.105, 9.964, 5.576, 9.067),
                                   (6.905, 12.556, 5.584, 7.445, 4.668, 4.910))])


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', site=config.site, channels=[1, 2, 3, 4, 5, 6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        if self.mode == 'train':
            return aug(image=img.cpu().detach().numpy())['image'], int(self.records[index].sirna)
            # return img, int(self.records[index].sirna)

        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len


df = pd.read_csv(path_data+'/train.csv', engine='python')


df_train, df_val = train_test_split(df, test_size=0.1, stratify=df.sirna, random_state=config.random_seed)

if not config.all:

    col_name = df.columns.tolist()
    df_train = pd.DataFrame([x for x in df_train.values if config.experiment in x[0]])
    df_val = pd.DataFrame([x for x in df_val.values if config.experiment in x[0]])
    df_train.columns = col_name
    df_val.columns = col_name

df_test = pd.read_csv(path_data+'/test.csv', engine='python')

ds = ImagesDS(df_train, path_data, mode='train')
ds_val = ImagesDS(df_val, path_data, mode='train')
ds_test = ImagesDS(df_test, path_data, mode='test')

loader = D.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = D.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
tloader = D.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)