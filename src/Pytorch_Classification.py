#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import pickle
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

start_time = time.time()

# Set Data Directory Path
directory_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(directory_path)

os.chdir(os.path.join(directory_path, 'data'))



# Pick up device which runs fast: either mps for M1 or gpu for linux

# In[21]:


def get_device(m1):
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif m1:
        if torch.backends.mps.is_built():
            return torch.device('mps')
        else:
            print('Pytorch Version Only Supports CPU')
            return torch.device('cpu')
    else:
        return torch.device('cpu')
        
device =get_device(m1 = True)
device


# Move Data to a  Device

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[23]:


class DeviceDataLoader():
    '''Wrap a dataloader to move data to a device'''
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        '''Yield a batch of data after moving it to device'''
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        '''Number of batches'''
        return len(self.dl)


# Load the Image Dataset to Tensor
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(X_train, y_train), (X_test, y_test) = data

# Cannot use append: change the array dimension
# X = torch.from_numpy(np.concatenate((X_train, X_test), axis = 0)).to(torch.float32)
# y = torch.from_numpy(np.concatenate((y_train, y_test), axis = 0)).to(torch.float32)
X_train = torch.from_numpy(X_train).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.long)
y_test = torch.from_numpy(y_test).to(torch.long)

def split_indices(n, training_pct):
    n_training= int(n*training_pct)
    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # pick first n_val indices for the training dataset, and other values for the validation
    return idxs[:n_training], idxs[n_training:]


# In[10]:


train_indices, val_indices = split_indices(X_train.shape[0], 0.8)
len(train_indices)


# In[11]:


batch_size = 100
dataset = TensorDataset(X_train, y_train)


train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler = train_sampler)

val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler = val_sampler)


# ## Wrap Data Loader

# In[26]:


train_dl = DeviceDataLoader(train_loader, device)
valid_dl = DeviceDataLoader(val_loader, device)



# ## Create Hidden Layer Neural Network for the Classification

# In[45]:


# Feedforward Neural Network with One Layer
class FFNModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    


# In[46]:


input_size = 28*28
num_classes = 10


# ## Train the Model Functions

# In[47]:


def loss_batch(model, loss_func, xb, yb, opt=None, metric = None):
    '''
    Calculates the loss and metric value for a batch of data, 
    and optionally performs gradient descent if an optimizer is provided
    '''
    preds = model(xb)
    loss = loss_func(preds, yb)
    if opt:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        #Reset gradients
        opt.zero_grad()
    #Evalaute metric
    metric_result = None
    if metric:
        metric_result = metric(preds, yb)
    return loss.item(), len(xb), metric_result


# In[48]:


def evaluate(model, loss_func, valid_dl, metric = None):
    '''
    Calculates the overall loss (and a metric, if provided) for the validation set
    '''
    with torch.no_grad():
        results = [loss_batch(model, loss_func, xb, yb, opt=None, metric = metric) for xb, yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums))/total  #nums is used to calculate weighted average
        avg_metric = None
        if metric:
            avg_metric = np.sum(np.multiply(metrics, nums))/total
    return avg_loss, total, avg_metric


# In[53]:


def fit(num_epochs, model, loss_fn, opt, train_dl, valid_dl, metric = None):
    '''
    Training
    '''
    val_losses = [0]*num_epochs
    val_metrics = [0]*num_epochs
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Predict+ Calculate loss+Train
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)

            # 2. Evaluation
            val_loss, total, val_metric = evaluate(model, loss_fn, valid_dl, metric)
            val_losses[epoch] = val_loss
            val_metrics[epoch] = val_metric
            
        # Print the progress
        if (epoch) % 2 == 0:
            if metric:
                print('Epoch [{}/{}], Loss: {:.4f}, {}:{}'.format(epoch + 1, num_epochs, val_loss, 
                                                                  metric.__name__, val_metric))
            else:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss))
    return val_losses, val_metrics


# In[54]:


# Create accuracy function for metric
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds==labels).item()/len(preds)


# ## Train Model

# In[55]:


# Convert model to appropriate device
model = FFNModel(input_size, hidden_size = 32, out_size=num_classes)
model.to(device)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_fn = F.cross_entropy


# In[56]:


losses, metrics = fit(10, model, loss_fn, optimizer, train_dl, valid_dl, accuracy)

total_run_time = time.time()-start_time
print('Total Run Time', total_run_time)

