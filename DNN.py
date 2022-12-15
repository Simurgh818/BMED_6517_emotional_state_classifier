import torch
from torch import nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.out = nn.Softmax()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss_fn(self, y_pred, y):
        loss = self.criterion(y_pred,y)
        return loss

class Conv(nn.Module):
    def __init__(self,training: bool,binary : bool):
        super().__init__()
        self.training = training
        self.conv1 = nn.Conv2d(4,32,(3,1),1)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,64,(3,1),1)
        self.drop2 = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(3)
        self.flat = nn.Flatten()
        
        if binary:
            self.criterion = nn.BCELoss()
            self.train_out = nn.Linear(5760,1)
            self.out = nn.Sigmoid()
        else:
            self.train_out = nn.Linear(5760,4)
            self.criterion = nn.CrossEntropyLoss()
            self.out = nn.Softmax(dim=1)
        
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool(x)
        x = self.flat(x)

        if self.training:
            x = self.out(self.train_out(x))
        
        return x

    def loss_fn(self, y_pred, y):
        loss = self.criterion(y_pred,y)
        return loss

    

class SAE(nn.Module):
    def __init__(self,input_size,rho,beta):
        super().__init__()
        self.encoder1 = nn.Linear(input_size,512)
        self.encoder2 = nn.Linear(512,128)
        self.activation = nn.ReLU()
        self.decoder1 = nn.Linear(128,512)
        self.decoder2 = nn.Linear(512,input_size)
        self.kldiv = nn.KLDivLoss()
        self.mse = nn.MSELoss()
        self.rho = rho
        self.beta = beta

    def forward(self,x):

        x = self.activation(self.encoder1(x))
        x = self.activation(self.encoder2(x))
        x = self.activation(self.decoder1(x))
        x = self.activation(self.decoder2(x))
        
        return x

    # define the sparse loss function
    def loss_fn(self,y_pred,y):
        J = self.mse(y_pred,y)
        #kl = self.kl_divergence(self.rho,p)
        l1_norm = sum(p.abs().sum()
                  for p in self.parameters())
        loss = J + self.beta*l1_norm
        return loss


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 batch_sz: int,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler,
                 x_val: torch.Tensor = torch.empty((0,)),
                 y_val: torch.Tensor = torch.empty((0,)),
                 max_iter: int=2000):
        self.model = model
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val =y_val
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.max_iter=max_iter
        self.epoch=0
        self.batch_sz = batch_sz
        self.loss = []
        self.val_loss = [0]

    def run_trainer(self):
        from tqdm import tqdm, trange
        progressbar = trange(self.max_iter, desc='Progress',leave=False)
        for i in progressbar:
            self.epoch += 1 #counts iterations
            loss = self._train()
            progressbar.set_postfix({'val loss':self.val_loss[-1], 'loss':loss})
            if self.x_val.size()[0]>0:
                self._val()
        return self.loss, self.val_loss

    def _val(self):
        x = self.x_val
        y = self.y_val
        num_batches = x.shape[0]//self.batch_sz
        batch_loss = 0
        for i in range(num_batches):
            x_batch = x[i*self.batch_sz:(i+1)*self.batch_sz].to(self.device)
            y_batch = y[i*self.batch_sz:(i+1)*self.batch_sz].to(self.device)
            y_pred = self.model(x_batch)
            loss = self.model.loss_fn(y_pred,y_batch)
            batch_loss += loss

        batch_loss = batch_loss.item()/num_batches
        self.val_loss.append(batch_loss)

    def _train(self):
        from tqdm import tqdm, trange
        shuffle = torch.randperm(self.x.shape[0])
        x = self.x[shuffle].to(self.device)
        y = self.y[shuffle].to(self.device)
        num_batches = self.x.shape[0]//self.batch_sz
        batch_iter = tqdm(range(num_batches),desc='Train',leave=False)
        batch_loss = 0
        for i in batch_iter:
            x_batch = x[i*self.batch_sz:(i+1)*self.batch_sz].to(self.device)
            y_batch = y[i*self.batch_sz:(i+1)*self.batch_sz].to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.model.loss_fn(y_pred,y_batch)
            loss.backward(retain_graph=True) # has to retain graph for DNN and not for the others
            #loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200)
            self.optimizer.step()
            batch_iter.set_postfix({'loss':loss.item()})
            batch_loss += loss
        batch_loss = batch_loss.item()/num_batches
        self.loss.append(batch_loss)
        self.lr_scheduler.step((loss))
        batch_iter.close()
        return batch_loss


class DNN_concatenated(nn.Module):
    def __init__(self,device,cnn_path,sae_path,dnn_path):
        super().__init__()
        self.cnn = Conv(training=False,binary=False).to(device)
        self.cnn.load_state_dict(torch.load(cnn_path))
        self.sae = SAE(5760,1e-8,1e-8).to(device)
        self.sae.load_state_dict(torch.load(sae_path))
        self.dnn = DNN(5760,4).to(device)
        self.dnn.load_state_dict(torch.load(dnn_path))

    def eval(self):
        self.cnn.eval()
        self.sae.eval()
        self.dnn.eval()
        return self

    def forward(self,x):
        x = self.cnn(x)
        x = self.sae(x)
        x = self.dnn(x)
        return x