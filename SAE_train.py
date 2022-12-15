import torch
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from DNN import SAE,Trainer

from sklearn.model_selection import train_test_split


np.random.seed(42)

sae_iterations = 100
sae_batch_sz = 64
sae_lr = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

now = datetime.now()
time_start = now.strftime('%H_%M')

log_dir = Path(f'logs/DNN_{time_start}')
log_dir.mkdir(parents=True,exist_ok=True)

x_cnn = torch.load('x_subj_train_cnn.pt')
x_val = torch.load('x_subj_val_cnn.pt')
#======================================Training SAE===========================================
sae_model = SAE(5760,1e-8,1e-8)
sae_model.to(device)
sae_model.train()
sae_optimizer = torch.optim.Adam(sae_model.parameters(),lr=sae_lr,betas=(0.9,0.999),eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    sae_optimizer, factor=0.8, patience=100
)
trainer = Trainer(sae_model,x_cnn,x_cnn,sae_batch_sz,device,sae_optimizer,scheduler,max_iter=sae_iterations)

start = time.time()
train_loss,val_loss = trainer.run_trainer()
end = time.time()
print('SAE training time: ',end-start)
torch.save(sae_model.state_dict(),log_dir / 'SAE.pt')
#sae_model.load_state_dict(torch.load('logs/DNN_18_07/SAE.pt'))
fig, axs = plt.subplots(1, 1)
axs.plot(train_loss)
axs.set_title('SAE Training Loss')
plt.savefig(log_dir / 'sae-loss.png')
plt.clf()
plt.close(fig)

#=================================Data Transform with SAE=====================================
sae_model.eval()
num_batches = x_cnn.shape[0]//sae_batch_sz
x_sae = []
for i in range(num_batches):
    x_batch = x_cnn[i*sae_batch_sz:(i+1)*sae_batch_sz].to(device)
    x_sae.append(sae_model(x_batch.to(device)))
x_sae = torch.vstack(x_sae)
torch.save(x_sae,'x_subj_train_sae.pt')

num_batches = x_val.shape[0]//sae_batch_sz
x_val_sae = []
for i in range(num_batches):
    x_batch = x_val[i*sae_batch_sz:(i+1)*sae_batch_sz].to(device)
    x_val_sae.append(sae_model(x_batch.to(device)))
x_val_sae = torch.vstack(x_val_sae)
torch.save(x_val_sae,'x_subj_val_sae.pt')