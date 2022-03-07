import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import einops

class INF(nn.Module):
    def __init__(self, in_channels, band_num, time_out, hidden_dim, d_model):
        super(INF, self).__init__()
        self.in_channels = in_channels
        self.band_num = band_num
        self.time_out = time_out
        # 上采样操作，提高分辨率
        self.interploate = lambda x: F.interpolate(x, [d_model, d_model], mode="bilinear")
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(in_channels * d_model, hidden_dim)

    def forward(self, x):
        # bs,ic,bn,to -> bs,ic,to,to
        x = self.interploate(x)
        x = einops.rearrange(x, "b i h w -> b w (i h)")
        x = self.linear(x)
        x = einops.rearrange(x, "b i c -> b c i")
        x = self.bn(self.relu(x))
        x=x.unsqueeze(-1)
        xs=x.repeat(1,1,1,x.shape[2])
        return xs



class Net(nn.Module):
    def __init__(self, num_classes, out, band_nums, time_dim,
                 hidden_dim, d_model, num_head=0, grad_rate=1.0, use_inf=False):
        super(Net, self).__init__()
        """
        bs,band_nums,out_channels,time_dim
        """
        self.classes = num_classes
        self.use_inf = use_inf
        self.time_out = time_dim
        self.band_nums = band_nums
        self.in_channels = out // band_nums
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.inf = INF(self.in_channels, self.band_nums, self.time_out, self.hidden_dim, self.d_model)
        self.blocks = []
        self.blocks.append(nn.Identity())
        self.blocks = nn.ModuleList(self.blocks)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.grad_rate = grad_rate

    def forward(self, data):
        data = self.inf(data).float()
        for block in self.blocks:
            data = block(data)
        m = data.shape[-1]
        data = data.view(data.shape[0], -1)/(m*self.grad_rate)
        return data
