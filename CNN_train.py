import torch
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from DNN import Conv,Trainer

from sklearn.model_selection import train_test_split


np.random.seed(42)
cnn_iterations = 40
cnn_batch_sz = 128
cnn_lr = 0.01


x_train = np.load('x_subj_train.npy')
x_val = np.load('x_subj_val.npy')

y_train = np.load('y_subj_train.npy')
y_val = np.load('y_subj_val.npy')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

now = datetime.now()
time_start = now.strftime('%H_%M')

log_dir = Path(f'logs/CNN_{time_start}')
log_dir.mkdir(parents=True,exist_ok=True)

#======================================Training CNN===========================================
cnn_model = Conv(True,False)
cnn_model.to(device)
cnn_model.train()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(),lr=cnn_lr,betas=(0.9,0.999),eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    cnn_optimizer, factor=0.8, patience=100
)
x_torch = torch.from_numpy(x_train).float()
y_torch = torch.from_numpy(y_train).long()

x_val_torch = torch.from_numpy(x_val).float()
y_val_torch = torch.from_numpy(y_val).long()

trainer = Trainer(cnn_model,x_torch,y_torch,cnn_batch_sz,device,cnn_optimizer,scheduler,x_val=x_val_torch,y_val=y_val_torch,max_iter=cnn_iterations)

#cnn_model.load_state_dict(torch.load('logs/CNN_17_27/CNN.pt'))

start = time.time()
train_loss,val_loss = trainer.run_trainer()
end = time.time()
print('CNN training time: ',end-start)
torch.save(cnn_model.state_dict(),log_dir / 'CNN.pt')

fig, axs = plt.subplots(1, 1)
axs.plot(train_loss)
axs.set_title('CNN Training Loss')
plt.savefig(log_dir / 'cnn-loss.png')
plt.clf()
plt.close(fig)



#=================================Data Transform with CNN=====================================
cnn_model.training = False
cnn_model.eval()
num_batches = x_train.shape[0]//cnn_batch_sz
x_train_cnn = []
for i in range(num_batches):
    x_batch = x_torch[i*cnn_batch_sz:(i+1)*cnn_batch_sz].to(device)
    x_train_cnn.append(cnn_model(x_batch.to(device)).detach())

x_train_cnn = torch.vstack(x_train_cnn)
torch.save(x_train_cnn,'x_subj_train_cnn.pt')

num_batches = x_val.shape[0]//cnn_batch_sz
x_val_cnn = []
for i in range(num_batches):
    x_batch = x_val_torch[i*cnn_batch_sz:(i+1)*cnn_batch_sz].to(device)
    x_val_cnn.append(cnn_model(x_batch.to(device)).detach())

x_val_cnn = torch.vstack(x_val_cnn)
torch.save(x_val_cnn,'x_subj_val_cnn.pt')
