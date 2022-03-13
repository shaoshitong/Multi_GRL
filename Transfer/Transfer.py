import torch, math,random
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.mmd import mmd
from torch.utils.checkpoint import checkpoint
class crossentropy(object):
    def __init__(self):
        pass

    def __call__(self, pred, gold, smoothing=0.1, *args, **kwargs):
        gold = gold.to(pred.device)
        if gold.ndim == 1:
            one_hot = torch.full_like(pred, fill_value=0.).to(pred.device)  # 0.0111111
            one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.)  # 0.9
            gold = one_hot
        gold = gold.float()
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        return torch.nn.functional.kl_div(input=log_prob.float(), target=gold, reduction='none').sum(dim=-1).mean()

    def cuda(self, ):
        return self


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.clone()
        return output, None

def normalize(feature,eps):
    norm=torch.norm(feature,2,1,keepdim=True)+eps
    feature=torch.div(feature,norm)
    return feature

class ResMLP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResMLP,self).__init__()
        self.mlp=nn.Sequential(nn.Linear(in_channels,32),
                               nn.ReLU(inplace=True),
                               nn.Linear(32,out_channels))
        self.shortcut=nn.Linear(in_channels,out_channels)
        self.in_channels=in_channels
        self.out_channels=out_channels
    def forward(self,x):
        y=self.mlp(x)+self.shortcut(x)
        return y

class Multi_GRL(nn.Module):
    def __init__(self, extractor, in_channels, subject_nums, num_classes):
        super(Multi_GRL, self).__init__()
        self.extractor = nn.Sequential(extractor, nn.Flatten())
        self.in_channels = in_channels
        self.subject_nums = subject_nums
        self.subject_addblock=nn.ModuleList([])
        for i in range(subject_nums):
            layer=nn.Linear(in_channels,512)
            self.subject_addblock.append(layer)
        self.subject_class_predictor=nn.ModuleList([])
        for i in range(subject_nums):
            layer=nn.Sequential(nn.ReLU(inplace=False),nn.Linear(512,num_classes))
            self.subject_class_predictor.append(layer)
        self.subject_domain_predictor=ResMLP(512,subject_nums)
        self.betas = 1

    def forward(self, x, y, subject_label, subject):
        """
        :param x: 输入样本
        :param y: 情绪类别
        :param subject_mask: 身份类别软标签
        :param subject_label: 身份类别硬标签
        :param subject: 当前目标域身份
        :return: 在训练阶段会返回两个loss和一个correct,测试阶段只返回一个correct
        """
        x = self.extractor(x)
        return self.TL(x,y,subject_label,subject)
    def TL(self,x,y,subject_label,subject):
        target_mask = torch.eq(torch.LongTensor([subject]).to(subject_label.device), subject_label)
        subject_batch_num = target_mask.sum().item()
        outs = []
        for i in range(self.subject_nums):
            if i == subject:
                continue
            subject_x = x[i * subject_batch_num:(i + 1) * subject_batch_num]
            subject_x = self.subject_addblock[i](subject_x)
            outs.append(subject_x)
        target_outs = []
        for i in range(self.subject_nums):
            if i == subject:
                continue
            subject_target = x[subject * subject_batch_num:(subject + 1) * subject_batch_num]
            subject_target = self.subject_addblock[i](subject_target)
            target_outs.append(subject_target)
        logits = []
        target_logits = []
        for i in range(self.subject_nums - 1):
            subject_logit = outs[i]
            # subject_logit=x[i*subject_batch_num:(i+1)*subject_batch_num]
            subject_logit = self.subject_class_predictor[i](subject_logit)
            logits.append(subject_logit)
            target_logit = target_outs[i]
            # target_logit=x[subject*subject_batch_num:(subject+1)*subject_batch_num]
            target_logit = self.subject_class_predictor[i](target_logit)
            target_logits.append(target_logit)
        "====================================================MMD LOSS====================================================="
        mmd_loss = 0.
        for source_out, target_out in zip(outs, target_outs):
            mmd_loss_one = mmd(source_out, target_out)
            mmd_loss = mmd_loss + mmd_loss_one
        mmd_loss = mmd_loss / (self.subject_nums - 1)
        "====================================================DISC LOSS====================================================="
        disc_loss = 0.
        for i in range(self.subject_nums - 1):
            for j in range(i + 1, self.subject_nums - 1):
                disc_loss = disc_loss + (F.softmax(outs[i], dim=1) - F.softmax(outs[j], dim=1)).abs().mean()
        disc_loss = disc_loss * 2 / ((self.subject_nums - 1) * (self.subject_nums - 2))
        "====================================================CLS LOSS====================================================="
        count = 0
        cls_loss = 0.
        for i in range(self.subject_nums):
            if i == subject:
                continue
            count += 1
            subject_target = y[i * subject_batch_num:(i + 1) * subject_batch_num]
            cls_loss = cls_loss + crossentropy()(logits[count - 1], subject_target)
        cls_loss = cls_loss / (self.subject_nums - 1)
        "====================================================PRED ACC====================================================="
        target_logit = torch.softmax(torch.stack(target_logits, -1), 1).mean(-1)
        target_true = y[subject * subject_batch_num:(subject + 1) * subject_batch_num]
        pred = torch.eq(torch.argmax(target_logit, 1), torch.argmax(target_true, 1))
        pred, nums = pred.sum(), pred.shape[0]
        return mmd_loss * 0.1, disc_loss * 0.4, cls_loss, pred, nums