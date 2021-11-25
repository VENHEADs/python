#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple, List
import numpy as np
import multiprocessing as mp
import os
from tqdm import tqdm_notebook
from sklearn.preprocessing import scale


# In[2]:


N_bootstraps: int = 10000
mp.set_start_method('fork', force=True)
def poisson_bootstrap_tp_fp_fn_tn(
    bundle: Tuple[float, List[Tuple[float, float, float, int]]],
                                 ) ->List[np.ndarray]:
    threshold, data = bundle
    TP = np.zeros((N_bootstraps))
    FP = np.zeros((N_bootstraps))
    FN = np.zeros((N_bootstraps))
    TN = np.zeros((N_bootstraps))
    for current_label, current_predict, weight, index in data:
        np.random.seed(index)
        current_predict += np.random.normal(0,0.0125,1) # this can be replaced with precalc noise
        current_predict = int(np.clip(current_predict,0,1) >= threshold)
        p_sample = np.random.poisson(1, N_bootstraps) * weight # this can be replaced with precalc poisson
        
        if current_label == 1 and current_predict == 1:
            TP += p_sample
        if current_label == 1 and current_predict == 0:
            FN += p_sample
        if current_label == 0 and current_predict == 1:
            FP += p_sample
        if current_label == 0 and current_predict == 0:
            TN += p_sample
            
    return [TP, FP, FN, TN]
            


# In[3]:


N = 10**6
labels = np.random.randint(0,2,N)
predicts = np.clip(np.random.normal(0.5,1,N),0,1)
weights = np.array([1 for _ in range(N)])

print(labels[:10])
print(predicts[:10])
print(weights[:10])


# In[5]:


chunk_size = 1000
threshold = 0.81
generator = (
    (
        threshold,
        [
            (labels[x + y],
             predicts[x + y],
             weights[x + y],
             x + y,
            )
    
        for x in range(chunk_size)
        if x+y < N
        ],
        
    )
        for y in range(0,N,chunk_size)

)


# In[6]:


cpu_to_use = np.max([os.cpu_count() - 3,1])
print(cpu_to_use)

with mp.Pool(processes=cpu_to_use) as pool:
    stat_list = list(tqdm_notebook(pool.imap(poisson_bootstrap_tp_fp_fn_tn,generator),
                    total = N//chunk_size))
    
TP, FP, FN, TN = np.sum(stat_list,0)
print(TP[:10])

