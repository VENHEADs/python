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
x = pd.read_csv('111.csv', header=0)
y = pd.read_csv('222.csv', header=0)
x1 = np.asarray(x)
y1 = np.asarray(y)
res = x1[:,1]*0.55+0.45*y1[:,1]
x1[:,1]=res
x.iloc[:][[1]]=res
x.to_csv('xgstacker_starter.sub.csv', index=None)
