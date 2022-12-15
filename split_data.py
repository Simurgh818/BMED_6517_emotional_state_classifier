import torch
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from DNN import DNN_concatenated
from sklearn.model_selection import train_test_split

# load in data
x = np.load('feats_win8_4.npy')
x = np.abs(x)
labels = np.load('labels_win8_4.npy')
labels[np.where(labels<5)] = 0
labels[np.where(labels>=5)] = 1
y = labels[:,0]*2 + labels[:,1]

# stratify split shuffling subjects together
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42,stratify=y)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle=True,random_state=42,stratify=y_train)

np.save('x_strat_test.npy',x_test)
np.save('y_strat_test.npy',y_test)

np.save('x_strat_train.npy',x_train)
np.save('y_strat_train.npy',y_train)

np.save('x_strat_val.npy',x_val)
np.save('y_strat_val.npy',y_val)

# subject wise split
# 520 samples per subject... 7 gives 79/21 split 
x_test = x[:520*7]
x_train = x[520*7:]
y_test = y[:520*7]
y_train = y[520*7:]

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,shuffle=True,random_state=42,stratify=y_train)

np.save('x_subj_test.npy', x_test)
np.save('y_subj_test.npy', y_test)

np.save('x_subj_train.npy',x_train)
np.save('y_subj_train.npy',y_train)

np.save('x_subj_val.npy',x_val)
np.save('y_subj_val.npy',y_val)
