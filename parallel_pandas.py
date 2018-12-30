
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from multiprocessing import Pool
from tsfresh.feature_extraction import extract_features
import multiprocessing
from tqdm import tqdm
import multiprocessing.pool
from functools import reduce
from numba import jit


# In[3]:


train = pd.read_csv('training_set.csv.zip')


# In[2]:


train = pd.read_csv('training_set.csv.zip')
test = pd.read_csv('test_set.csv.zip')
train_metada = pd.read_csv('training_set_metadata.csv')
test_metada = pd.read_csv('test_set_metadata.csv.zip')


# In[15]:


fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},                            {'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}
def make_features(df):
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    # train[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]

#     agg_df_ts_ex_f = extract_features(df, column_id='object_id', n_jobs=1,disable_progressbar = True)

    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'detected': ['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'flux_ratio_sq':['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'flux_by_flux_ratio_sq':['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'detected': ['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'mjd' : ['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum'],
        'passband' :['min', 'max', 'mean', 'median', 'std','skew','prod','count','sum']
    }
    
    
    agg_df = df.groupby('object_id').agg(aggs)
    
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_df.columns = new_columns
    agg_df['flux_diff'] = agg_df['flux_max'] - agg_df['flux_min']
    agg_df['flux_dif2'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_mean']
    agg_df['flux_w_mean'] = agg_df['flux_by_flux_ratio_sq_sum'] / agg_df['flux_ratio_sq_sum']
    agg_df['flux_dif3'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_w_mean']
    # Add more features with 
    agg_df_ts = extract_features(df, column_id='object_id',disable_progressbar = True,             column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=1)
    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()

    agg_df_mjd = extract_features(df_det, column_id='object_id',disable_progressbar = True,                         column_value = 'mjd', default_fc_parameters = {'maximum':None, 'minimum':None}, n_jobs=1)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'] - agg_df_mjd['mjd__minimum']
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on = 'id')
    # tsfresh returns a dataframe with an index name='id'
    agg_df_ts.index.rename('object_id',inplace=True)
    agg_df = pd.merge(agg_df, agg_df_ts, on='object_id')
#     agg_df = pd.merge(agg_df, agg_df_ts_ex_f, on='object_id')
    agg_df['object_id'] = agg_df.index
    
    return agg_df.transpose().drop_duplicates().transpose()


# In[7]:


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# In[8]:


def break_in_to_chunks(df,n_step = 900):
    list_of_ids = df.object_id.values
    step = len(list_of_ids) // n_step
    indexes = []
    for index in tqdm(range(step,len(list_of_ids) - step,step)):
        if list_of_ids[index] != list_of_ids[index+1]:
            indexes.append(index+1)
        else: 
            while list_of_ids[index] == list_of_ids[index+1]:
                index+=1
            indexes.append(index+1)

    splitted = []
    splitted.append(df[:indexes[0]])
    for i in tqdm(range(len(indexes)-1)):
        splitted.append(df[indexes[i]:indexes[i+1]])
    splitted.append((df[indexes[i+1]:]))
    assert reduce(lambda x,y: x + y,map(len,splitted)) == len(df)
    return splitted


# In[9]:


train_splitted = break_in_to_chunks(train,30)
test_splitted = break_in_to_chunks(test,600)


# In[15]:


c = 0
for i in range(len(train_splitted)):
    c+=train_splitted[i]['object_id'].nunique()
assert c == train_metada.shape[0]
  
c = 0
for i in range(len(test_splitted)):
    c+=test_splitted[i]['object_id'].nunique()
assert c == test_metada.shape[0]
    


# In[ ]:


print('train is splitted')
pool = MyPool(30)
train_agg = pd.concat(tqdm(pool.imap(make_features, train_splitted),total=30))
pool.terminate()


# In[ ]:



print('train is splitted')
pool = MyPool(30)
train_agg = pd.concat(tqdm(pool.imap(make_features, train_splitted),total=30))
pool.terminate()

print('test is splitted')
pool = MyPool(30)
test_agg = pd.concat(tqdm(pool.imap(make_features, test_splitted),total=600))
pool.terminate()


# In[ ]:


assert train_agg.shape[0] == train_metada.shape[0]
assert test_agg.shape[0] == test_metada.shape[0]


# In[ ]:


new_train = train_metada.merge(train_agg)
new_test = test_metada.merge(test_agg)


# In[ ]:


new_train.to_csv('new_train_features_agregated.csv.gz',compression='gzip')
new_test.to_csv('new_test_features_agregated.csv.gz',compression='gzip')

