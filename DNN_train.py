import torch
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from DNN import DNN,Trainer

from sklearn.model_selection import train_test_split

np.random.seed(42)

dnn_iterations = 100
dnn_batch_sz = 128
dnn_lr = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

now = datetime.now()
time_start = now.strftime('%H_%M')

log_dir = Path(f'logs/DNN_{time_start}')
log_dir.mkdir(parents=True,exist_ok=True)

y = np.load('y_subj_train.npy')
y_train = torch.from_numpy(y).long()

y_val = np.load('y_subj_val.npy')
y_val = torch.from_numpy(y_val).long()

y_test = np.load('y_subj_test.npy')
y_test = torch.from_numpy(y_test).long()

x_train = torch.load('x_subj_train_sae.pt')

x_val = torch.load('x_subj_val_sae.pt')


#================================Split Data for Training======================================
#x_test = x_sae[:4*200]
#y_test = y_torch[:4*200]
#x_train_val = x_sae[4*200:]
#y_train_val = y_torch[4*200:]

#x_train, x_val, y_train, y_val = train_test_split(x_train_val,y_train_val,test_size=0.2,shuffle=True,random_state=42)

#======================================Training DNN===========================================
dnn_model = DNN(x_train.shape[1],4)
dnn_model.to(device)
dnn_model.train()
dnn_optimizer = torch.optim.Adam(dnn_model.parameters(),lr=dnn_lr,betas=(0.9,0.999),eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    dnn_optimizer, factor=0.7, patience=5
)

trainer = Trainer(dnn_model,x_train,y_train,dnn_batch_sz,device,dnn_optimizer,scheduler,x_val,y_val,max_iter=dnn_iterations)

start = time.time()
train_loss,val_loss = trainer.run_trainer()
end = time.time()
print('DNN training time: ',end-start)
torch.save(dnn_model.state_dict(),log_dir / 'DNN.pt')

fig, axs = plt.subplots(2, 1)
axs[0].plot(train_loss)
axs[0].set_title('DNN Training Loss')
axs[1].plot(val_loss)
axs[1].set_title('Validation Loss')
plt.savefig(log_dir / 'dnn-loss.png')
plt.clf()
plt.close(fig)


# calculate accuracy on test and validation set
dnn_model.eval()
num_batches = x_train.shape[0]//dnn_batch_sz
y_train_pred = []
for i in range(num_batches):
    x_batch = x_train[i*dnn_batch_sz:(i+1)*dnn_batch_sz].to(device)
    y_train_pred.append(dnn_model(x_batch.to(device)))
y_train_pred = torch.vstack(y_train_pred)


dnn_model.eval()
num_batches = x_val.shape[0]//dnn_batch_sz
y_val_pred = []
for i in range(num_batches):
    x_batch = x_val[i*dnn_batch_sz:(i+1)*dnn_batch_sz].to(device)
    y_val_pred.append(dnn_model(x_batch.to(device)))
y_val_pred = torch.vstack(y_val_pred)

y_val_pred = y_val_pred.detach().cpu().numpy()
y_train_pred = y_train_pred.detach().cpu().numpy()

from sklearn.metrics import balanced_accuracy_score,accuracy_score

y_pred = np.argmax(y_train_pred,axis=1)
acc = balanced_accuracy_score(y_train[:10368],y_pred)
print('Train Accuracy: ', acc)

y_pred = np.argmax(y_val_pred,axis=1)
acc = balanced_accuracy_score(y_val[:2560],y_pred)
print('Validation Accuracy: ', acc)


