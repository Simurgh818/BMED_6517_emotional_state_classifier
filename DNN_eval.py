import torch
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from DNN import DNN_concatenated
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import shap

dnn_batch_sz = 16

# load in data
x = np.load('x_subj_test.npy')
y = np.load('y_subj_test.npy')

x_background = np.load('x_subj_train.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x_test = torch.from_numpy(x).float()
y_test = torch.from_numpy(y).long()
x_background = torch.from_numpy(x_background).float()

# load model
model = DNN_concatenated(device,'logs\CNN_22_25\CNN.pt','logs\DNN_22_34\SAE.pt','logs\DNN_22_40\DNN.pt')
model.eval()
# run inference
# test accuracy...
num_batches = x_test.shape[0]//dnn_batch_sz
y_test_pred = []
for i in range(num_batches):
    x_batch = x_test[i*dnn_batch_sz:(i+1)*dnn_batch_sz].to(device)
    y_test_pred.append(model(x_batch.to(device)))
y_test_pred = torch.vstack(y_test_pred)

y_test_pred = torch.nn.functional.softmax(y_test_pred,dim=1)
y_pred = np.argmax(y_test_pred.detach().cpu().numpy(),axis=1)

# evaluate accuracy (and other metrics if needed)
acc = accuracy_score(y_test[:3632],y_pred)
print('Test Accuracy: ', acc)


# SHAP importance extraction
e = shap.DeepExplainer(model,x_background[:100].to(device))
shap_values = e.shap_values(x_test[:10].to(device))

print('debug')