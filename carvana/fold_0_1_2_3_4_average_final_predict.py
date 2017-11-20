
# coding: utf-8

# In[1]:

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
weg = '/media/n01z3/storage3/dataset/carvana/'


# In[2]:

from renorm import BatchRenormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import ELU

import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool


# In[3]:

white = False
image_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    zca_whitening=white)

mask_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    zca_whitening=white)


# In[4]:

load_img = lambda im: cv2.imread(join(weg+'train', '{}.jpg'.format(im)))
load_mask = lambda im: imread(join(weg+'train_masks', '{}_mask.gif'.format(im)))
load_mask_gif = lambda im: imread(join(weg+'train_masks', '{}_mask.gif'.format(im)))[:,:,0]
load_img_2 = lambda fold,im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/9970/{}/test/'.format(fold), '{}.png'.format(im)))
load_img_3 = lambda fold,im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/linknet18_mxnet_test/{}/'.format(fold), '{}.png'.format(im)))
load_img_4 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/best_submission_png/', '{}.png'.format(im)))
load_img_5 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/linknet_18_new/fold4/test/', '{}.png'.format(im)))
load_img_6 = lambda fold,im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/test/{}/'.format(fold), '{}.png'.format(im)))
load_img_7 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/dinknet/fold4/test/', '{}.png'.format(im)))




resize = lambda im: downscale_local_mean(im, (4,4) if im.ndim==2 else (4,4,1))
mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))
from os.path import join


# In[5]:

def batch_gen(size=1,height=160,width=240,aug=True,inter=True,fit=False):
    counter = 0 #счетчик запускам
    global fold_number
    select = df_folds.img.values[df_folds.fold!=fold_number]
    number_of_batches = np.ceil(len(select)/size) #количестов партий - зависит от размера партии
    while 1:



        cc = 0
        x = np.empty((size, height, width, 3), dtype=np.float32) # пустой массив для загрузки картинок
        y = np.empty((size, height, width, 1), dtype=np.float32)
        for img_id in select[size*counter:size*(counter+1)]: #16

            imgs_id = [cv2.resize(load_img(img_id),(width,height))/255.]
            # Input is image + mean image per channel + std image per channel
           
            mask_16 = [cv2.resize(load_mask(img_id), ( width,height))/ 255.]
            try:
                if mask_16[0].shape[2]>1:

                    mask_16 = [cv2.resize(load_mask_gif(img_id), ( width,height))/ 255.]
            except:
                pass
                
            #X[i] = np.concatenate([imgs_id[idx-1], np.mean(imgs_id, axis=0), np.std(imgs_id, axis=0)], axis=2)
            
            x[cc:(1+cc)] = np.array(imgs_id)
            y[cc:(1+cc)] = np.expand_dims(np.array(mask_16),3)
            
            cc+=1    
        del imgs_id
        counter+=1
        SEED = int(np.random.randint(0,999999,1))
        if inter == False: #если не чередуем данные
            if aug == True:  # делаем ли аугментацию?
                if fit==True:
                    image_generator.fit(x)
                    mask_generator.fit(y)


                for img in image_generator.flow((x),batch_size=size,seed=SEED):
                    break

                for msk in mask_generator.flow((y),batch_size=size,seed=SEED):
                    break 
                yield(img,msk)

            if aug == False: # или базовую картинку?
                yield(x,y)

        if inter == True: # если чередуем то по очереди загружаем то одно то другое
            if counter % 2 ==0:
                if fit==True:
                    image_generator.fit(x)
                    mask_generator.fit(y)
                for img in image_generator.flow((x),batch_size=size,seed=SEED):
                    break

                for msk in mask_generator.flow((y),batch_size=size,seed=SEED):
                    break 
                yield(img,msk)

            else:
                yield(x,y)
                
        if counter == number_of_batches:
            counter = 0
            np.random.shuffle(select)


# In[6]:

def val_gen(size=1,height=160,width=240,aug=True,inter = True,fit=False):
    global fold_number
    select = df_folds.img.values[df_folds.fold==fold_number]
    number_of_batches = np.ceil(len(select)/size) #количестов партий - зависит от размера партии
    counter = 0
    while 1:



        cc = 0
        x = np.empty((size, height, width, 3), dtype=np.float32) # пустой массив для загрузки картинок
        y = np.empty((size, height, width, 1), dtype=np.float32)
        for img_id in select[size*counter:size*(counter+1)]: #16

            imgs_id = [cv2.resize(load_img(img_id),(width,height))/255.]
            # Input is image + mean image per channel + std image per channel
            mask_16 = [cv2.resize(load_mask(img_id), ( width,height))/ 255.]
            try:
                if mask_16[0].shape[2]>1:

                    mask_16 = [cv2.resize(load_mask_gif(img_id), ( width,height))/ 255.]
            except:
                pass
            
            x[cc:(1+cc)] = np.array(imgs_id)
            y[cc:(1+cc)] = np.expand_dims(np.array(mask_16),3)
            
            cc+=1    
        del imgs_id
        counter+=1
        
        SEED = int(np.random.randint(0,999999,1))
        if inter == False: #если не чередуем данные
            if aug == True:  # делаем ли аугментацию?
                if fit==True:
                    image_generator.fit(x)
                    mask_generator.fit(y)


                for img in image_generator.flow((x),batch_size=size,seed=SEED):
                    break
                for msk in mask_generator.flow((y),batch_size=size,seed=SEED):
                    break 
                yield(img,msk)
                #del img,msk
                #del x,y
            if aug == False: # или базовую картинку?
                yield(x,y)
                #del x,y
        if inter == True: # если чередуем то по очереди загружаем то одно то другое
            if counter % 2 ==0:
                if fit==True:
                    image_generator.fit(x)
                    mask_generator.fit(y)
                for img in image_generator.flow((x),batch_size=size,seed=SEED):
                    break
                
                for msk in mask_generator.flow((y),batch_size=size,seed=SEED):
                    break 
                yield(img,msk)
                #del img,msk
                #del x,y
            else:
                yield(x,y)
                #del x,y
        
        if counter == number_of_batches:
            counter = 0
            np.random.shuffle(select)


# In[7]:

import pandas as pd
import numpy as np

import cv2


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

df_folds = pd.read_csv(weg+'folds_ready.csv')
df_train = pd.read_csv(weg+'train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

input_size_1 = 1920
input_size_2 = 1280

epochs = 100
batch_size = 1

#ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1)#, random_state=42)

#print('Training on {} samples'.format(len(ids_train_split)))
#print('Validating on {} samples'.format(len(ids_valid_split)))


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread(weg+'train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size_1, input_size_2))
                mask = imread(weg+'train_masks/{}_mask.gif'.format(id), cv2.IMREAD_GRAYSCALE)
                try:
                    if mask.shape[2]>1:

                        mask = mask[:,:,0]
                except:
                    pass
                mask = cv2.resize(mask, (input_size_1, input_size_2))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread(weg+'train/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size_1, input_size_2))
                mask = imread(weg+'train_masks/{}_mask.gif'.format(id), cv2.IMREAD_GRAYSCALE)
                try:
                    if mask.shape[2]>1:

                        mask = mask[:,:,0]
                except:
                    pass
                mask = cv2.resize(mask, (input_size_1, input_size_2))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch



# In[8]:

import h5py
from tqdm import tqdm
size_1 = 1280
size_2 = 1280

    import pandas as pd
    import numpy as np

    import cv2
    from tqdm import tqdm
    threshold = 0.5


    
    df_test = pd.read_csv(weg+'sample_submission.csv')
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
                

        #x0 = np.load(weg+str(0)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x1 = np.load(weg+str(1)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x2 = np.load(weg+str(2)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x3 = np.load(weg+str(3)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x4 = np.load(weg+str(4)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.2
        #img = cv2.imread(weg+'test/{}.jpg'.format(id_1))
        #img = cv2.resize(img, (x0.shape[0], x0.shape[1]))
        #img = img/255.
        #img = np.expand_dims(img,0)
       # img = model.predict(img)
        #img = img[0] #*0.4       


        
        #loaded_1 = x0+x1+x2+x3+x4+img
        #y0 = (load_img_2(0,id_1)/255.)[:,:,0:1]*0.2
        #y1 = (load_img_2(1,id_1)/255.)[:,:,0:1]*0.2
        #y2 = (load_img_2(2,id_1)/255.)[:,:,0:1]*0.2
        #y3 = (load_img_2(3,id_1)/255.)[:,:,0:1]*0.2
        #y4 = (load_img_2(4,id_1)/255.)[:,:,0:1]*0.2
        
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
    df.to_csv('submit/nizh.csv.gz', index=False, compression='gzip')
    print('finished_fold_')
    
    
    
    
    import pandas as pd
    import numpy as np

    import cv2
    from tqdm import tqdm
    threshold = 0.5


    
    df_test = pd.read_csv(weg+'sample_submission.csv')
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
                

        #x0 = np.load(weg+str(0)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x1 = np.load(weg+str(1)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x2 = np.load(weg+str(2)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x3 = np.load(weg+str(3)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x4 = np.load(weg+str(4)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.2
        #img = cv2.imread(weg+'test/{}.jpg'.format(id_1))
        #img = cv2.resize(img, (x0.shape[0], x0.shape[1]))
        #img = img/255.
        #img = np.expand_dims(img,0)
       # img = model.predict(img)
        #img = img[0] #*0.4       


        
        #loaded_1 = x0+x1+x2+x3+x4+img
        y0 = (load_img_6(0,id_1)/255.)[:,:,0:1]*0.2
        y1 = (load_img_6(1,id_1)/255.)[:,:,0:1]*0.2
        y2 = (load_img_6(2,id_1)/255.)[:,:,0:1]*0.2
        y3 = (load_img_6(3,id_1)/255.)[:,:,0:1]*0.2
        y4 = (load_img_6(4,id_1)/255.)[:,:,0:1]*0.2
        
        z0 = (load_img_3(0,id_1)/255.)[:,:,0:1]*0.2
        z1 = (load_img_3(1,id_1)/255.)[:,:,0:1]*0.1
        z2 = (load_img_3(2,id_1)/255.)[:,:,0:1]*0.2
        z3 = (load_img_3(3,id_1)/255.)[:,:,0:1]*0.1
        z4 = (load_img_3(4,id_1)/255.)[:,:,0:1]*0.4
        
        img = (load_img_4(id_1)/255.)[:,:,0:1]
        
        n0 = (load_img_5(id_1)/255.)[:,:,0:1]
        
        
        y_total = y3+y4+y0+y1+y2
        z_total = z3+z4+z0+z1+z2
        loaded = (img*0.87+z_total*0.13)*0.8 + 0.1*y_total + 0.1*n0
        
        
        #prob = cv2.resize(loaded, (orig_width, orig_height))
        mask = loaded > threshold
        rle = _mask_to_rle_string(mask)
        return rle
    

    names_2 = list(ids_test)

    chunk = 50
    rles = []
    for i in tqdm(range(0, len(names_2), chunk), total=len(names_2) // chunk):
        p = Pool(processes=50)
        rles += p.map(get_rle, names_2[i:i + chunk])
        p.terminate()
        #print('vasya')в

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/best_noiz_3.csv.gz', index=False, compression='gzip')
    print('finished_fold_')
    
    
    
    
    import pandas as pd
    import numpy as np

    import cv2
    from tqdm import tqdm
    threshold = 0.5


    
    df_test = pd.read_csv(weg+'sample_submission.csv')
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
                

        #x0 = np.load(weg+str(0)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x1 = np.load(weg+str(1)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x2 = np.load(weg+str(2)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x3 = np.load(weg+str(3)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x4 = np.load(weg+str(4)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.2
        #img = cv2.imread(weg+'test/{}.jpg'.format(id_1))
        #img = cv2.resize(img, (x0.shape[0], x0.shape[1]))
        #img = img/255.
        #img = np.expand_dims(img,0)
       # img = model.predict(img)
        #img = img[0] #*0.4       


        
        #loaded_1 = x0+x1+x2+x3+x4+img

        
        z0 = (load_img_3(0,id_1)/255.)[:,:,0:1]*0.2
        z1 = (load_img_3(1,id_1)/255.)[:,:,0:1]*0.1
        z2 = (load_img_3(2,id_1)/255.)[:,:,0:1]*0.2
        z3 = (load_img_3(3,id_1)/255.)[:,:,0:1]*0.1
        z4 = (load_img_3(4,id_1)/255.)[:,:,0:1]*0.4
        
        img = (load_img_4(id_1)/255.)[:,:,0:1]
        
        d4 = (load_img_7(id_1)/255.)[:,:,0:1]
        
        z_total = z3+z4+z0+z1+z2
        loaded = (img*0.87+z_total*0.13)*0.6 + 0.4*d4
        
        
        #prob = cv2.resize(loaded, (orig_width, orig_height))
        mask = loaded > threshold
        rle = _mask_to_rle_string(mask)
        return rle
    

    names_2 = list(ids_test)

    chunk = 50
    rles = []
    for i in tqdm(range(0, len(names_2), chunk), total=len(names_2) // chunk):
        p = Pool(processes=50)
        rles += p.map(get_rle, names_2[i:i + chunk])
        p.terminate()
        #print('vasya')в

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/best_noiz_3_nizh_dinknet.csv.gz', index=False, compression='gzip')
    print('finished_fold_')
    
    
    
    
    import pandas as pd
    import numpy as np

    import cv2
    from tqdm import tqdm
    threshold = 0.5


    
    df_test = pd.read_csv(weg+'sample_submission.csv')
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
                

        #x0 = np.load(weg+str(0)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x1 = np.load(weg+str(1)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x2 = np.load(weg+str(2)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x3 = np.load(weg+str(3)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
        #x4 = np.load(weg+str(4)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.2
        #img = cv2.imread(weg+'test/{}.jpg'.format(id_1))
        #img = cv2.resize(img, (x0.shape[0], x0.shape[1]))
        #img = img/255.
        #img = np.expand_dims(img,0)
       # img = model.predict(img)
        #img = img[0] #*0.4       


        
        #loaded_1 = x0+x1+x2+x3+x4+img
        y0 = (load_img_6(0,id_1)/255.)[:,:,0:1]*0.2
        y1 = (load_img_6(1,id_1)/255.)[:,:,0:1]*0.2
        y2 = (load_img_6(2,id_1)/255.)[:,:,0:1]*0.2
        y3 = (load_img_6(3,id_1)/255.)[:,:,0:1]*0.2
        y4 = (load_img_6(4,id_1)/255.)[:,:,0:1]*0.2
        
        z0 = (load_img_3(0,id_1)/255.)[:,:,0:1]*0.2
        z1 = (load_img_3(1,id_1)/255.)[:,:,0:1]*0.1
        z2 = (load_img_3(2,id_1)/255.)[:,:,0:1]*0.2
        z3 = (load_img_3(3,id_1)/255.)[:,:,0:1]*0.1
        z4 = (load_img_3(4,id_1)/255.)[:,:,0:1]*0.4
        
        img = (load_img_4(id_1)/255.)[:,:,0:1]
        
        n0 = (load_img_5(id_1)/255.)[:,:,0:1]
        
        
        #y_total = y3+y4+y0+y1+y2
        #z_total = z3+z4+z0+z1+z2
        d4 = (load_img_7(id_1)/255.)[:,:,0:1]
        
        loaded = np.power(y0*y1*y2*y3*y4*z0*z1*z2*z3*z4*img*n0*d4,1./13)
        
        
        #prob = cv2.resize(loaded, (orig_width, orig_height))
        mask = loaded > threshold
        rle = _mask_to_rle_string(mask)
        return rle
    

    names_2 = list(ids_test)

    chunk = 50
    rles = []
    for i in tqdm(range(0, len(names_2), chunk), total=len(names_2) // chunk):
        p = Pool(processes=50)
        rles += p.map(get_rle, names_2[i:i + chunk])
        p.terminate()
        #print('vasya')в

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/all_gmean.csv.gz', index=False, compression='gzip')
    print('finished_fold_')
    
    
    
    

# In[ ]:

load_img_7 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/linknet18_mxnet/test_gmean/', '{}.png'.format(im)))
load_img_8 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/linknet18_new/test_gmean/', '{}.png'.format(im)))
load_img_9 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/linknet34_mxnet/test_gmean/', '{}.png'.format(im)))
load_img_10 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/venheads_folder/test_gmean/', '{}.png'.format(im)))
load_img_11 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/ternaus/test_gmean/', '{}.png'.format(im)))
load_img_12 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/dinknet/', '{}.png'.format(im)))
load_img_13 = lambda im: cv2.imread(join('/media/n01z3/storage3/dataset/carvana/geom_mean_total/pspnet34/test_gmean/', '{}.png'.format(im)))





# In[ ]:

import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
threshold = 0.5



df_test = pd.read_csv(weg+'sample_submission.csv')
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
            

    #x0 = np.load(weg+str(0)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
    #x1 = np.load(weg+str(1)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
    #x2 = np.load(weg+str(2)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
    #x3 = np.load(weg+str(3)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.1
    #x4 = np.load(weg+str(4)+'_fold_test_predicted/' + str(id_1)+'.npz')['arr_0']#*0.2
    #img = cv2.imread(weg+'test/{}.jpg'.format(id_1))
    #img = cv2.resize(img, (x0.shape[0], x0.shape[1]))
    #img = img/255.
    #img = np.expand_dims(img,0)
   # img = model.predict(img)
    #img = img[0] #*0.4       


    
    #loaded_1 = x0+x1+x2+x3+x4+img

    
    x0 = (load_img_7(id_1)/255.)[:,:,0:1]
    x1 = (load_img_8(id_1)/255.)[:,:,0:1]
    x2 = (load_img_9(id_1)/255.)[:,:,0:1]
    x3 = (load_img_10(id_1)/255.)[:,:,0:1]
    x4 = (load_img_11(id_1)/255.)[:,:,0:1]
    x5 = (load_img_12(id_1)/255.)[:,:,0:1]
    x6 = (load_img_13(id_1)/255.)[:,:,0:1]
    #img = (load_img_4(id_1)/255.)[:,:,0:1]
    
    
    
    #y_total = y3+y4+y0+y1+y2
    #z_total = z3+z4+z0+z1+z2
    #d4 = (load_img_7(id_1)/255.)[:,:,0:1]
    
    loaded = np.power(x0*x1*x2*x3*x4*x5*x6,1./7)
    
    
    #prob = cv2.resize(loaded, (orig_width, orig_height))
    mask = loaded > threshold
    rle = _mask_to_rle_string(mask)
    return rle


names_2 = list(ids_test)

chunk = 50
rles = []
for i in tqdm(range(0, len(names_2), chunk), total=len(names_2) // chunk):
    p = Pool(processes=50)
    rles += p.map(get_rle, names_2[i:i + chunk])
    p.terminate()
    #print('vasya')в

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/all_gmean_7.csv.gz', index=False, compression='gzip')
print('finished_fold_')





