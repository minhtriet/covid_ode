#!/usr/bin/env python
# coding: utf-8

# #todo
# 
# minmax scaler
# min: 0
# max: Canada population

# In[2]:


import torch
import torch.nn as nn
import os
from covid_ode import sidartha_ode
#from ode_nn import Dataset, train_epoch, eval_epoch, get_lr
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import random
import warnings


from tqdm.notebook import tqdm
# from ode_nn import Dataset_graph, train_epoch_graph, eval_epoch_graph, get_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


canada = pd.read_csv('canada.csv')


# In[4]:


CANADA_POPULATION = 38.01e6


# In[5]:


canada.shape


# In[6]:


TRAIN_IDX = int(canada.shape[0]*.5)
VAL_IDX = TRAIN_IDX + int(canada.shape[0]*.1)


# In[7]:


TRAIN_IDX, VAL_IDX


# In[5]:


canada


# In[6]:


canada = canada.values[:, 0:3] / CANADA_POPULATION


# In[7]:


canada[:, 0:3]


# In[8]:


test_idx = 131

# Learning Rate
lr = 0.01

# number of historic data points for fitting
input_steps = 2

# forecasting horizon
output_steps = 2

# number of epochs for training
num_epochs = 20000

# select data for training
data = canada[test_idx-input_steps:test_idx+output_steps]   # only 1 training sample
y_exact = data[:,:input_steps]


# In[9]:


model = sidartha_ode.AutoOdeSIDARTHE(len_data = output_steps+input_steps).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1000, gamma=0.9)
loss_fun = torch.nn.MSELoss()
min_loss = 1

##################################################################
for e in tqdm(range(num_epochs)):
    optimizer.zero_grad()    
    y_approx = model(I0=data[0, 0], E0=data[0, 2], H0=data[0, 1])
    i_e_h_approx = y_approx[:,2], y_approx[:,7], y_apprpox[:6]
    loss = loss_fun(y_approx, y_exact)
    loss.backward()
    optimizer.step()
    scheduler.step()
######## Weighted Loss ########
#     loss_weight = weight_fun(input_steps, function = "sqrt", feat_weight = True)
#     loss = torch.mean(loss_weight*loss_fun(y_approx[:,:,-3:], y_exact[:,:input_steps,-3:])) 

######## A few constraints that can potential improve the model ########
#     positive_constraint = loss_fun(F.relu(-model.beta), torch.tensor(0.0).float().to(device))
#     diagonal_constraint = loss_fun(torch.diagonal(model.A, 0),torch.tensor(1.0).float().to(device))
#     initial_constraint = loss_fun(model.init_S + model.init_E + model.init_I + model.init_R + model.init_U, torch.tensor(1.0).float().to(device))
#     loss += initial_constraint + positive_constraint + diagonal_constraint 
   
    if loss.item() < min_loss:
        best_model = model
        min_loss = loss.item()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
#     if e%1000 == 0:
#         y_approx2 = model(data.shape[1]).data.numpy()
#         y_exact2 = data.data.numpy()
#         print(list_csv[test_idx][:10])
#         #torch.mean(torch.abs(y_approx - y_exact)[:,-7:]).data, torch.mean(torch.abs(y_approx - y_exact)[:,30:]).data
#         for i in range(3):
#             print(np.mean(np.abs(y_approx2*scaler - y_exact2*scaler)[:,-7:, i]))


# In[ ]:


data, y_exact

