import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from Sampler.Dataloader import get_data_seed
from WideLinear import Net
from Transfer.Transfer import Multi_GRL
from Optimizer.AdaBoud import AdaBound

parser = argparse.ArgumentParser()  # 创建对象
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nEpoch', type=int, default=200)

args = parser.parse_args()


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


def print_list(result_log):
    result_log: dict
    return_list = []
    for key, value in result_log.items():
        if len(value) == 0:
            continue
        value = np.array(value[-20:])
        mean = round(value.mean(), 3)
        max = round(value.max(), 3)
        std = round(value.std(), 3)
        print(f"{key} : mean:{mean}, std:{std}, best:{max}\n")
        return_list.append((key, mean, std, max))
    return return_list


def save_csv(filename, **kwargs):
    columns = []
    value_list = []
    # 遍历字典中的所有键
    for keys in kwargs.keys():
        columns.append(keys)
        # 访问字典中的值
        value = kwargs[keys]
        value_list.append(value)
    # 转换为矩阵形式
    value_list = np.array(value_list, dtype=np.float32).T
    dataframe = pd.DataFrame(value_list, columns=columns)
    dataframe.to_csv(filename, index=True, header=True)


def test(myNet, test_data_loader):
    myNet.eval()
    correct = 0
    with torch.no_grad():
        for t_sample, t_label in test_data_loader:
            t_sample, t_label = t_sample.cuda(), t_label.cuda()
            class_output = myNet(t_sample, t_label, None, None, None)
            correct += class_output
    acc = correct / len(test_data_loader)
    return acc


def train(myNet, source_loader, subject):
    myNet.train()
    len_sample = 0
    correct = 0
    loss = 0
    """
    source_data:样本
    source_label:标签
    subject_label:人标签
    subject_mark:分数
    """
    for source_data, source_label, subject_label, subject_mark in source_loader:
        optimizer.zero_grad()
        source_data, source_label, subject_label, subject_mark = source_data.cuda(), source_label.cuda(), subject_label.cuda(), subject_mark.cuda()
        """
        这里一共输入五个变量，分别是样本，样本情绪标签，身份软标签，身份硬标签，当前目标域身份，
        """
        with torch.cuda.amp.autocast(enabled=False):
            loss1, loss2, acc = myNet(source_data, source_label, subject_mark, subject_label, subject)
        loss_total = loss1+loss2*0.1

        loss_total.backward(retain_graph=False)
        optimizer.step()
        loss += loss_total.clone().detach().item()
        correct += acc
    train_acc = correct / len(source_loader)
    train_acc1 = round(train_acc, 3)
    train_accuracy.append(train_acc1)
    item_pr = 'Train Epoch: [{}/{}], total_loss: {:.3f} epoch{}_Acc: {:.3f}' \
        .format(epoch, args.nEpoch, loss / len(source_loader), epoch, train_acc)
    print(item_pr)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    mode = 'SEED'
    sample_path = 'G:/Alex/SEED_experiment/Three sessions sample/DE_4D/'
    path = "G:/Alex/SEED_experiment/Backbone_contrast/Transformer/result/experiment1"
    if not os.path.isdir(path):
        os.mkdir(path)
    if mode == 'DEAP':
        total = 32
    elif mode == 'SEED':
        total = 15
    else:
        raise NotImplementedError
    ALL_SUBJECT_ACC = []
    ALL_BSET_ACC=[]
    for i in range(total):
        scaler = torch.cuda.amp.GradScaler()
        """
        首先获取数据集，第一个参数是路径，第二个参数是人数，第三个是现在目标域是第几个人，第四个类别数，第五个batchsize
        """
        source_loader, target_loader = get_data_seed(sample_path, 15, i, 3, 256)
        """
        生成特征提取器
        """
        extractor = Net(3, 310, 5, 180, 64, 100, 0, 0.9, True)
        """
        生成完整架构，第一个参数是特征提取器，第二个参数是特征提取器输出的特征数量，第三个是人数，第四个是类别数
        """
        myNet = Multi_GRL(extractor, 100 * 100 * 64, 15, 3)
        myNet.class_predictor = nn.Linear(100 * 100 * 64, 3)
        myNet.subject_predictor = nn.Linear(100 * 100 * 64, 15)
        myNet = myNet.cuda()
        # optimizer = optim.AdamW(myNet.parameters(), lr=4 * 0.0001)
        optimizer = AdaBound(myNet.parameters(), lr=4 * 0.0001,betas=(0.5,0.999),)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [600], 0.1)
        train_accuracy = []
        test_accuracy = []
        e_error = []
        for epoch in range(1, args.nEpoch + 1):
            train(myNet, source_loader, i)
            #scheduler.step()
            test_acc = test(myNet, target_loader)
            test_accuracy.append(round(test_acc, 3))
            test_info = 'Test acc Epoch{}: {:.3f}'.format(epoch, test_acc)
            print(test_info)
        last_ACC = test_accuracy[-1]
        best_acc = max(test_accuracy)
        ALL_SUBJECT_ACC.append(last_ACC)
        ALL_BSET_ACC.append(best_acc)
        print(f"last acc is:{last_ACC}%,best acc is:{best_acc}")
    ALL_SUBJECT_ACC = np.array(ALL_SUBJECT_ACC)
    ALL_BSET_ACC=np.array(ALL_BSET_ACC)
    print(f"the last mean acc is {ALL_SUBJECT_ACC.mean()}%,the last std acc is {ALL_SUBJECT_ACC.std()}%\n"
          f"the best mean acc is {ALL_BSET_ACC.mean()}%,the best std acc is {ALL_BSET_ACC.std()}%")
