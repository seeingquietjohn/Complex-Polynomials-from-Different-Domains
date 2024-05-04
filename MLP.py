#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_channels): 
        super(MLP, self).__init__()
        layers = []
        for i in range(len(num_hidden)): 
            if i == 0: 
                layers.append(nn.Linear(num_inputs, num_hidden[i]))
            else: 
                layers.append(nn.Linear(num_hidden[i-1], num_hidden[i]))
            layers.append(nn.LeakyReLU())
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_hidden[-1], num_channels)
        
    def forward(self,x): 
        x = self.hidden(x)
        x = self.output(x)
        return x



# In[ ]:




