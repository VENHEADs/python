
# coding: utf-8

# In[43]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
import os
import tqdm
import pandas as pd
import pickle
from scipy.spatial.distance import cosine


# In[2]:


model = ResNet50(weights='imagenet')


# In[47]:


def preproc_image(img_path):
    img = image.load_img('mirflickr/{}'.format(img_path), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
test_img = preproc_image('im3.jpg')


# In[4]:


model.layers.pop()
model2 = Model(model.input, model.layers[-1].output)
if np.sum(model2.get_weights()[0] - model.get_weights()[0]) == 0:
    print('model is ok')


# In[5]:


preds = model.predict(test_img)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])


# In[13]:


list_names = os.listdir('mirflickr')
list_names.pop(0)
list_names = list_names[:-1]


# In[ ]:


vector_representation = []
for name in tqdm.tqdm(list_names,miniters=10000):
    img = preproc_image(name)
    vector_representation.append(model2.predict(img))


# In[45]:


d = dict((key, value) for (key, value) in zip(list_names,vector_representation))


# In[23]:


with open('dict_representation.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[78]:


img = preproc_image('im50.jpg')
test_vector = model2.predict(img)

def find_distance(vector):
    return cosine(test_vector,vector)

distance = map(find_distance,d.values())

distances = pd.DataFrame(distance)
distances['img'] = pd.DataFrame(d.keys())
distances.rename(columns = {0:'distance'},inplace=True)

print(distances.sort_values(by='distance').img.values[0])


# In[121]:


test = distances.sort_values(by='distance').img.values[0:5]


# In[83]:


model2.save('model_to_predict_vector.h5')

