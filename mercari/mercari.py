# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys

from multiprocessing import Pool

from functools import reduce
from nltk.corpus import stopwords
stopWords = []
for i in """!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'""":
    stopWords.append(i)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()



#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])



start_time = time.time()
from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# if 1 == 1:
train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

####
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test
#test = simulate_test(test)
print('new shape ', test.shape)
####
#train = pd.read_table('../input/./input/mercari-price-suggestion-challenge/train.tsv', engine='c')
#test = pd.read_table('../input/./input/mercari-price-suggestion-challenge/test.tsv', engine='c')

print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
nrow_test = train.shape[0]  # -dftt.shape[0]
dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)
del dftt['price']
nrow_train = train.shape[0]
# print(nrow_train, nrow_test)
y = np.log1p(train["price"])
merge = pd.concat([train, dftt, test])
#submission: pd.DataFrame = test[['test_id']]

del train
del test
gc.collect()
import re
def clen_text(text):
    text = text.replace(' blk ', ' black ')
    text = text.replace(' sz ', ' size ')
    text = text.replace(' l ', ' large ')
    text = text.replace(' + ', ' plus ')
    text = text.replace('+', ' plus ')
    text = text.replace('14kt', '14 carat gold')
    text = text.replace('14k', '14 carat gold')
    text = text.replace('carat', 'carat gold')
    text = text.replace('tiffany & co', 'tiffany')
    text = text.replace(' vs ', ' victoria secret ')
    text = text.replace('42ct', '42 carat gold')
    text = text.replace('42kt', '42 carat gold')
    text = text.replace('10kt', '10 carat gold')
    text = text.replace('10ct', '10 carat gold')
    text = text.replace('18k', '10 carat gold')
    
    text = re.sub(r'\doz ', '\1 oz ', text)
    
    
    return text

    
    

merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))


handle_missing_inplace(merge)
merge.fillna(method='ffill',inplace=True)
merge.fillna(method='bfill',inplace=True)

print('[{}] Handle missing completed.'.format(time.time() - start_time))

p = Pool(processes=8)
merge['item_description'] = p.map(clen_text, merge.item_description.values)
p.terminate()
    

cutting(merge)
print('[{}] Cut completed.'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Convert categorical completed'.format(time.time() - start_time))

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True
X_name = wb.fit_transform(merge['name'])
del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

wb = CountVectorizer()
X_category1 = wb.fit_transform(merge['general_cat'])
X_category2 = wb.fit_transform(merge['subcat_1'])
X_category3 = wb.fit_transform(merge['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

# wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                              "idf": None})
                         , procs=8)
wb.dictionary_freeze= True

# p = Pool(processes=8)
# merge['item_description'] = p.map(transform, merge.item_description.values)
# p.terminate()

X_description = wb.fit_transform(merge['item_description'])
del(wb)
gc.collect()
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
      X_name.shape)
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

print('[{}] Create sparse merge completed'.format(time.time() - start_time))

#    pd.to_pickle((sparse_merge, y), "xy.pkl")
# else:
#    nrow_train, nrow_test= 1481661, 1482535
#    sparse_merge, y = pd.read_pickle("xy.pkl")

# Remove features with document frequency <=1
print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)
d_shape = sparse_merge.shape[1]
gc.collect()
train_X, train_y = X, y
if develop:
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

model = FTRL(alpha=0.1680, beta=9.7895, L1=0.0011, L2=8.9635, D=d_shape, iters=int(11), inv_link="identity", threads=4)
###

del lb
del mask
del X_name
del X_category1
del X_category2
del X_category3
del X
del y
del merge
del X_dummies
del X_brand
del dftt
del X_description

del sparse_merge
gc.collect()

####
# model.fit(train_X, train_y)
# print('[{}] Train FTRL completed'.format(time.time() - start_time))
# if develop:
#     preds = model.predict(X=valid_X)
#     print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

# predsF = model.predict(X_test)
print('[{}] Predict FTRL completed'.format(time.time() - start_time))

model = FM_FTRL(alpha=0.1410, beta=0.1896, L1=4.9447, L2=9.8198, D=d_shape, alpha_fm=0.0498, L2_fm=0.0027, 
    init_fm=0.0040, D_fm=int(99), e_noise=0.0172, iters=int(3), inv_link="identity", 
    threads=4, seed=2017)
gc.collect() 
model.fit(train_X, train_y)
print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
del train_X
del train_y
gc.collect()

if develop:
    preds = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

predsFM = model.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))



preds = predsFM

preds_FTML = np.expm1(preds)

#submission.to_csv("submission_wordbatch_ftrl_fm_lgb.csv", index=False)
del X_test
#del predsF
del predsFM
del preds
gc.collect()

import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
start_time = time.time()

train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')
train = train[train.price!=0]
test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv', sep='\t')

train['target'] = np.log1p(train['price'])
# In[ ]:


print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test
#test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


# In[ ]:

#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
train.fillna(method='ffill',inplace=True)
train.fillna(method='bfill',inplace=True)
test.fillna(method='ffill',inplace=True)
test.fillna(method='bfill',inplace=True)
print(train.shape)
print(test.shape)

print('[{}] Finished handling missing data...'.format(time.time() - start_time))


# In[ ]:


#PROCESS CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
train.head(3)


# In[ ]:

from functools import reduce
from nltk.corpus import stopwords
stopWords = []
for i in """!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'""":
    stopWords.append(i)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
lemma_dict = {}
def lemmatizer(word):
    try:
        word = word.lower()
        if word in lemma_dict:
            return lemma_dict[word]
        else:
            normal_form = ps.stem(word)
            lemma_dict[word] = normal_form
            return normal_form
    except:
        return 'nonascii'
def join_2(x,*args):
    if len(args) == 0:
        return x
    else:
        return x+" " +args[0]
def transform(x):
    try:
        for symbol in stopWords:
            if symbol in x:
                x = x.replace(symbol,"")
        x = x.split()
        x = map(lemmatizer,x)
        #x = map(asc,x)
        x =  reduce(join_2,x)
        return x
    except:
        return "problem"
    


p = Pool(processes=8)
train.item_description = p.map(clen_text, train.item_description.values)
p.terminate()

p = Pool(processes=8)
test.item_description = p.map(clen_text, test.item_description.values)
p.terminate()
        
p = Pool(processes=8)
train.item_description = p.map(transform, train.item_description.values)
p.terminate()

p = Pool(processes=8)
test.item_description = p.map(transform, test.item_description.values)
p.terminate()
      

print ("Lemma finished")


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train.category_name.str.lower(), 
                      train.item_description.str.lower(), 
                      train.name.str.lower()])

tok_raw = Tokenizer(num_words=40000)
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)
del raw_text
del tok_raw
del lemma_dict
gc.collect()
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


# In[ ]:


#EXTRACT DEVELOPTMENT TEST
from sklearn.model_selection import train_test_split
dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)

# In[ ]:


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 20 #17
MAX_ITEM_DESC_SEQ = 60 #269
MAX_CATEGORY_NAME_SEQ = 20 #8
MAX_TEXT = np.max([np.max(train.seq_name.max())
                   , np.max(test.seq_name.max())
                   , np.max(train.seq_category_name.max())
                   , np.max(test.seq_category_name.max())
                   , np.max(train.seq_item_description.max())
                   , np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()])+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), 
                        test.item_condition_id.max()])+1

print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))


# In[ ]:


#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'category_name': pad_sequences(dataset.seq_category_name
                                        , maxlen=MAX_CATEGORY_NAME_SEQ)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)


print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))


# In[ ]:


#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization,MaxPooling1D, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers

def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

dr = 0.25

def get_model():
    #params
    dr_r = dr
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]], 
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 42
    MAX_TEXT = 40000
    emb_name = Embedding(MAX_TEXT+2, emb_size)(name)
    emb_item_desc = Embedding(MAX_TEXT+2, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT+2, emb_size//3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    # emb_item_desc = MaxPooling1D(2,2)(emb_item_desc)
    # emb_category_name = MaxPooling1D(2,2)(emb_category_name)
    # emb_name = MaxPooling1D(2,2)(emb_name)
    
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_category_name)
    rnn_layer3 = GRU(12) (emb_name)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    main_l = Dropout(0.1)(Dense(256,activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand
                   , category, category_name
                   , item_condition, num_vars], output)
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mae", 
                  optimizer=optimizer)
    return model

def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)
    
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle
#fin_lr=init_lr * (1/(1+decay))**(steps-1)
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))


# In[ ]:

gc.collect()
#FITTING THE MODEL
epochs = 3
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init, lr_fin = 0.010, 0.00001
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)
results = []
history = model.fit(X_train, dtrain.target
                    , epochs=1
                    , batch_size=512 * 3
               
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
v_rmsle_2 = eval_model(model)
if v_rmsle_2 < 0.43:
    preds_2 = model.predict(X_test, batch_size=512 * 10)
    preds_2 = np.expm1(preds_2)
    results.append(preds_2)  
    
    
lr_init, lr_fin = 0.010, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)



history = model.fit(X_train, dtrain.target
                    , epochs=1
                    , batch_size=512 * 10
             
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
v_rmsle_2 = eval_model(model)
if v_rmsle_2 < 0.43:
    preds_2 = model.predict(X_test, batch_size=512 * 10)
    preds_2 = np.expm1(preds_2)
    results.append(preds_2)  

lr_init, lr_fin = 0.010, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)



history = model.fit(X_train, dtrain.target
                    , epochs=1
                    , batch_size=512 * 10
             
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
v_rmsle_2 = eval_model(model)
if v_rmsle_2 < 0.43:
    preds_2 = model.predict(X_test, batch_size=512 * 10)
    preds_2 = np.expm1(preds_2)
    results.append(preds_2) 
    
lr_init, lr_fin = 0.010, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)



history = model.fit(X_train, dtrain.target
                    , epochs=1
                    , batch_size=512 * 10
             
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
v_rmsle_2 = eval_model(model)
if v_rmsle_2 < 0.43:
    preds_2 = model.predict(X_test, batch_size=512 * 10)
    preds_2 = np.expm1(preds_2)
    results.append(preds_2) 
    
lr_init, lr_fin = 0.010, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)



history = model.fit(X_train, dtrain.target
                    , epochs=1
                    , batch_size=512 * 10
             
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
v_rmsle_2 = eval_model(model)
if v_rmsle_2 < 0.43:
    preds_2 = model.predict(X_test, batch_size=512 * 10)
    preds_2 = np.expm1(preds_2)
    results.append(preds_2) 
    

    
     


if len(results) ==1:
    preds = results[0]
if len(results) == 0:
    preds = model.predict(X_test, batch_size=512 * 10)
    preds = np.expm1(preds)
if len(results)>1:
    preds = results[0]
    for i in results[1:]:
        preds*=i
    preds = preds **(1./len(results))
    
    
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
#EVLUEATE THE MODEL ON DEV TEST
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))

# In[ ]:
del X_train 
del X_valid 
gc.collect()

#CREATE PREDICTIONS
#preds = model.predict(X_test, batch_size=BATCH_SIZE)
#preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]]#[:test_len]
submission["price"] = preds#[:test_len]
#submission.to_csv("./myNN"+log_subdir+"_{:.6}.csv".format(v_rmsle), index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))

del test
del model
del preds
del X_test 
gc.collect()






print('final prediction')
#final = preds_FTML* 0.25 + preds_nn*0.65 + pred_ridge * 0.1
print('final prediction_1')
submission["price"]  = submission["price"]*0.477 + preds_FTML*0.523
print('final prediction_2')

#submission['price']  = preds_FTML
print('final prediction_3')


submission.to_csv("sumbmission_final.csv", index = False)