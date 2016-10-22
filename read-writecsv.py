import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from math import exp, log
import os
from os import listdir
import csv
x = pd.read_csv('x111.csv', header=0) # base8_nr900_p10¬_cleaned
y = pd.read_csv('x222.csv', header=0) #submission_5fold-average-xgb_1146.10852_2016-10-13-02-40
z = pd.read_csv('x333.csv', header=0) #genetic gpsubmission

x1 = np.asarray(x)
y1 = np.asarray(y)
z1 = np.asarray(z)
res = x1[:,1]*0.46+0.46*y1[:,1]+0.08*z1[:,1]

x1[:,1]=res
x.iloc[:][[1]]=res
x.to_csv('av0.46_0.46_0.08.csv', index=None)
v = pd.DataFrame(x1)
