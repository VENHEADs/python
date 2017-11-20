
# coding: utf-8

# In[1]:

import pandas as pd

#################
#weg = /media/n01z3/storage3/dataset/carvana/linknet_18_new/fold4/test/ # путь к картинкам
#weg_csv = #путь к csv
########пу
import numpy as np
from multiprocessing import Pool


import cv2
from tqdm import tqdm
load_img_5 = lambda im: cv2.imread(join(weg, '{}.png'.format(im)))

threshold = 0.5



df_test = pd.read_csv(weg_csv+'sample_submission.csv') 
ids_test = df_test['img'].map(lambda s: s.split('.')[0])


orig_width = 1918
orig_height = 1280

def _mask_to_rle_string(mask):
    """Convert boolean/`binary uint` mask to RLE string."""
    # Mask to RLE
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    # RLE to string
    return ' '.join(str(x) for x in runs)

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


rles = []

def get_rle(id_1):
            
    
    z0 = (load_img_5(id_1)/255.)[:,:,0:1]

    
    
    


    loaded = z0
    
    
    #prob = cv2.resize(loaded, (orig_width, orig_height))
    mask = loaded > threshold
    rle = _mask_to_rle_string(mask)
    return rle


names_2 = list(ids_test)

chunk = 50
rles = []
for i in tqdm(range(0, len(names_2), chunk), total=len(names_2) // chunk):
    p = Pool(processes=25)
    rles += p.map(get_rle, names_2[i:i + chunk])
    p.terminate()
    #print('vasya')

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('nizh.csv.gz', index=False, compression='gzip')
print('finished_fold_')





