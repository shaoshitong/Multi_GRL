import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.clone()
        return output, None



# Scaled dot product attention
def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    # dropout设置参数，避免p_atten过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(3)])
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value):
        batches = query.size(0)
        residual = query
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.(B,5,180,62)
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batches, -1, self.h * self.d_k)
        # linear out + dropout(用新的线性层)(B,180,310)
        x = self.dropout(self.linear(x))
        # Residual connection + layerNorm
        x += residual
        x = self.layer_norm(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(AttentionLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.position = PositionalEncoding(d_model=310, dropout=0.5, max_len=180)
        self.time_attention1 = AttentionLayer(h=10, d_model=310, dropout=0.5)
        self.feed1 = FeedForward(d_model=310, d_ff=512, dropout=0.5)
        self.encoder_layer1 = nn.Sequential(self.time_attention1, self.feed1)

    def forward(self, x):
        x_reshape = rearrange(x, 'b c h w -> b w (c h)')
        x_pos = self.position(x_reshape)
        out = self.encoder_layer1(x_pos)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.featureExtraction = nn.Sequential(FeatureExtraction())
    def forward(self, x):
        return self.featureExtraction(x)
