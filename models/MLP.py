import torch,math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicMLP(nn.Module):
    def __init__(self,channels,hidden_dims):
        super(BasicMLP, self).__init__()
        self.channels=channels
        self.hidden_dims=hidden_dims
        self.extractor=nn.Sequential()
        self.extractor.add_module("linear1",nn.Linear(self.channels,self.hidden_dims))
        self.extractor.add_module("relu",nn.ELU(inplace=True))
        self.extractor.add_module("linear2",nn.Linear(self.hidden_dims,self.channels))
    def forward(self,x):
        if x.ndim>2:
            x=x.view(x.shape[0],-1)
        x=self.extractor(x)+x
        return x

class MLP(nn.Module):
    def __init__(self,channels,hidden_dims,num_layer=1):
        super(MLP,self).__init__()
        self.extractor=[]
        for i in range(num_layer):
            self.extractor.append(BasicMLP(channels,hidden_dims))
        self.extractor=nn.ModuleList(self.extractor)
    def forward(self,x):
        for layer in self.extractor:
            x=layer(x)
        return x

