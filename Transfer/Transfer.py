import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
class Multi_GRL(nn.Module):
    def __init__(self, extractor, in_channels, subject_nums, num_classes):
        super(Multi_GRL, self).__init__()
        self.extractor = nn.Sequential(extractor, nn.Flatten())
        self.in_channels = in_channels
        self.subject_nums = subject_nums
        self.class_predictor = nn.Sequential(*[
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        ])
        self.subject_predictor = nn.Sequential(*[
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, subject_nums)
        ])
        self.betas = 0.5

    def forward(self, x, y, subject_mask, subject_label, subject):
        """
        :param x: 输入样本
        :param y: 情绪类别
        :param subject_mask: 身份类别软标签
        :param subject_label: 身份类别硬标签
        :param subject: 当前目标域身份
        :return: 在训练阶段会返回两个loss和一个correct,测试阶段只返回一个correct
        """
        if self.training == True:
            x = self.extractor(x)
            label_out = self.class_predictor(x)
            subject_out = self.subject_predictor(ReverseLayerF.apply(x, 0.95))  # b,15
            # subject_out=rearrange(subject_out, "(s i) v -> s i v", s=subject_mask.shape[1]).mean(1)
            subject_out = torch.nn.functional.softmax(subject_out, dim=1)
            subject_mask = -torch.log(1 / (subject_mask) - 1)
            subject_mask = subject_mask - subject_mask.min(dim=-1, keepdim=True)[0]
            # subject_mask = rearrange(subject_mask, "(s i) v -> s i v", s=subject_mask.shape[1]).mean(1)
            subject_mask = (subject_mask.max(dim=-1, keepdim=True)[0] - subject_mask)
            subject_mask = subject_mask / subject_mask.sum(-1, keepdim=True)
            yin_subject_mask = torch.eye(subject_mask.shape[1], dtype=torch.float).to(subject_mask.device).unsqueeze(
                1).repeat(1, int(subject_mask.shape[0] / subject_mask.shape[1]), 1).view(-1, subject_mask.shape[-1])
            loss_subject_1 = F.mse_loss(subject_out, subject_mask,size_average=False, reduction="none")
            loss_subject_2 = F.mse_loss(subject_out, yin_subject_mask, size_average=False,reduction="none")
            loss_subject = self.betas * loss_subject_1 + (1. - self.betas) * loss_subject_2
            use = ~torch.eq(torch.LongTensor([subject]).to(subject_label.device), subject_label)
            y = y.long()[use]
            label_out = label_out[use]
            loss_class = crossentropy()(label_out, y)
            pred = torch.argmax(label_out, dim=1)
            correct = 100 * torch.eq(pred, y.argmax(1)).float().sum().item() / pred.shape[0]
            return loss_class, loss_subject*0.5, correct
        else:
            x = self.extractor(x)
            label_out = self.class_predictor(x)
            pred = torch.argmax(label_out, dim=1)
            correct = 100 * torch.eq(pred, y.argmax(1)).float().sum().item() / pred.shape[0]
            return correct
