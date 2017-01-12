
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt

def perf_measure(y_actual,y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if np.all(y_actual[i]==1 and y_hat[i]==1):
            TP += 1
    for i in range(len(y_hat)): 
        if np.all(y_hat[i]==1 and y_actual[i]==0):
            FP += 1
    for i in range(len(y_hat)): 
        if np.all(y_actual[i]==0 and y_hat[i]==0):
            TN += 1
    for i in range(len(y_hat)): 
        if np.all(y_hat[i]==0 and y_actual[i]==1):
            FN += 1

    return(TP, FP, TN, FN)
def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')

from sklearn.metrics import accuracy_score, confusion_matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap="cool"):
    
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    target_names ="01"
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

