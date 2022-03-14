import random

import torch,math
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import os
from Sampler.Dataloader import get_data_seed
from models.Transformer import Transformer
from Transfer.Transfer import Multi_GRL
from models.MLP import MLP
from Optimizer.AdaBoud import AdaBound
import logging
parser = argparse.ArgumentParser()  # 创建对象
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nEpoch', type=int, default=50)

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
            class_output = myNet(t_sample,t_label,None,None,None)
            correct+=class_output
    acc = correct / len(test_data_loader)
    return acc


def train(myNet, source_loader,subject):
    myNet.train()
    global now_i,iteration,scaler
    correct=0
    cls_total_loss=0
    mmd_total_loss=0
    disc_total_loss=0
    """
    source_data:样本
    source_label:标签
    subject_label:人标签
    subject_mark:分数
    """
    len_data=0
    global a_w,b_w
    for source_data, source_label,subject_label,subject_mark in source_loader:
        optimizer.zero_grad()
        source_data,source_label,subject_label = source_data.cuda(), source_label.cuda(),subject_label.cuda()
        """
        这里一共输入五个变量，分别是样本，样本情绪标签，身份软标签，身份硬标签，当前目标域身份，
        """
        global_weight = (2 / (1 + math.exp(-10 * (now_i) / (iteration))) - 1)
        # optimizer.param_groups[0]['lr'] = 0.002 / math.pow((1 + 2 * (now_i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[1]['lr'] = 0.002 / math.pow((1 + 2 * (now_i - 1) / (iteration)), 0.75)
        # optimizer.param_groups[2]['lr'] = 0.002 / math.pow((1 + 2 * (now_i - 1) / (iteration)), 0.75)
        now_i+=1
        global top_num
        with torch.cuda.amp.autocast(enabled=True):
            mmd_loss,disc_loss,cls_loss,pred,nums= myNet(source_data,source_label,subject_label,subject,top_num)
        loss_total=cls_loss+global_weight*(mmd_loss*a_w+disc_loss*b_w)
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        cls_total_loss=cls_total_loss+cls_loss.clone().detach().cpu().item()
        mmd_total_loss=mmd_total_loss+mmd_loss.clone().detach().cpu().item()
        disc_total_loss=disc_total_loss+disc_loss.clone().detach().cpu().item()
        correct=correct+pred.clone().detach().cpu().item()
        len_data+=nums
    correct=correct*100
    train_acc = correct / len_data
    train_acc1 = round(train_acc, 3)
    train_accuracy.append(train_acc1)
    item_pr = 'Train Epoch: [{}/{}], cls_loss: {:.3f}, mmd_loss: {:.3f}, disc_loss: {:.3f}, epoch{}_Acc: {:.3f}%, iter:{}' \
        .format(epoch, args.nEpoch, cls_total_loss/len(source_loader),mmd_total_loss/len(source_loader),disc_total_loss/len(source_loader), epoch, train_acc,global_weight)
    print(item_pr)
    logger.info(item_pr)
    return train_acc

if __name__ == '__main__':

    mode = 'SEED'
    sample_path = '/data/EEG/DE_4D/'
    path = '/home/sst/log/EEG/'
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename='/home/sst/log/EEG/HUANWEI_EXPERIMENT/TMSFAN_MLP_SEED',
                        filemode='w')
    logger = logging.getLogger('MLP')
    if not os.path.isdir(path):
        os.mkdir(path)
    if mode == 'DEAP':
        total = 32
    elif mode == 'SEED':
        total = 15
    else:
        raise NotImplementedError
    best_weight=[]
    for m in range(100):
        a_w,b_w=random.random()*2/3,random.random()
        top_num = 5
        ALL_SUBJECT_ACC = []
        ALL_BSET_ACC = []
        for i in range(total):
            logger.info(f"train: {i}")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            """
            首先获取数据集，第一个参数是路径，第二个参数是人数，第三个是现在目标域是第几个人，第四个类别数，第五个batchsize
            """
            scaler=torch.cuda.amp.GradScaler()
            source_loader,target_loader=get_data_seed(sample_path,15,i,3,225)
            """
            生成特征提取器
            """
            extractor=MLP(310*180,1024,3)
            """
            生成完整架构，第一个参数是特征提取器，第二个参数是特征提取器输出的特征数量，第三个是人数，第四个是类别数
            """
            iteration = args.nEpoch * len(source_loader)+1
            now_i=0
            myNet = Multi_GRL(extractor,310*180,15,3).cuda()
            optimizer = AdaBound([
                {'params':myNet.extractor.parameters(),'lr':0.00004,'final_lr':0.004},
                {'params': myNet.subject_class_predictor.parameters(), 'lr': 0.00004,'final_lr':0.004},
                {'params': myNet.subject_addblock.parameters(), 'lr': 0.00004,'final_lr':0.004},
            ],betas=(0,0.9))

            # optimizer = optim.SGD([
            #     {'params':myNet.extractor.parameters(),'lr':0.00004,'final_lr':0.004},
            #     {'params': myNet.subject_class_predictor.parameters(), 'lr': 0.00004,'final_lr':0.004},
            #     {'params': myNet.subject_addblock.parameters(), 'lr': 0.00004,'final_lr':0.004},
            # ],lr=0.001)
            train_accuracy = []
            test_accuracy = []
            e_error = []
            best_acc = -float('inf')
            for epoch in range(1, args.nEpoch + 1):
                test_acc=train(myNet, source_loader,i)
                test_accuracy.append(test_acc)
            last_ACC = sum(test_accuracy[-3:])/3
            best_acc = max(test_accuracy)
            ALL_SUBJECT_ACC.append(last_ACC)
            ALL_BSET_ACC.append(best_acc)
            print(f"last acc is:{last_ACC}%,best acc is:{best_acc}")
            logger.info(f"last acc is:{last_ACC}%,best acc is:{best_acc}")
        ALL_SUBJECT_ACC = np.array(ALL_SUBJECT_ACC)
        ALL_BSET_ACC = np.array(ALL_BSET_ACC)
        print(f"the last mean acc is {ALL_SUBJECT_ACC.mean()}%,the last std acc is {ALL_SUBJECT_ACC.std()}%\n"
              f"the best mean acc is {ALL_BSET_ACC.mean()}%,the best std acc is {ALL_BSET_ACC.std()}%")
        logger.info(f"the last mean acc is {ALL_SUBJECT_ACC.mean()}%,the last std acc is {ALL_SUBJECT_ACC.std()}%\n"
              f"the best mean acc is {ALL_BSET_ACC.mean()}%,the best std acc is {ALL_BSET_ACC.std()}%")
        logger.info(f"a_w is {a_w},b_W is {b_w}")
        best_weight.append((ALL_SUBJECT_ACC.mean(),ALL_SUBJECT_ACC.std(),a_w,b_w))
    best_weight=sorted(best_weight,key=lambda x:x[0],reverse=True)
    print(best_weight)
    logger.info(best_weight)

# ===================================================
# train_accuracy : mean:99.979, std:0.063, best:100.0
#
# test_accuracy : mean:82.059, std:0.669, best:82.815
# ===================================================
'''
the last mean acc is 78.51833333333335%,the last std acc is 8.419625650163367%
the best mean acc is 80.59266666666667%,the best std acc is 8.052062424552412%

the last mean acc is 76.59266666666666%,the last std acc is 8.809446238114074%
the best mean acc is 79.40740000000001%,the best std acc is 7.675218136313779%

the last mean acc is 75.55553333333334%,the last std acc is 8.114436664502485%
the best mean acc is 78.22220000000002%,the best std acc is 7.917329231502249%

the last mean acc is 78.07400000000001%,the last std acc is 8.149562172288766%
the best mean acc is 80.44440000000002%,the best std acc is 7.791522673607087%

the last mean acc is 76.148%,the last std acc is 8.133346830589893%
the best mean acc is 79.2592%,the best std acc is 7.197060827865775%



the last mean acc is 78.22226666666667%,the last std acc is 10.010749452241605%
the best mean acc is 81.48140000000002%,the best std acc is 9.16856621142768%

the last mean acc is 76.8888%,the last std acc is 7.942140980869076%
the best mean acc is 79.85186666666668%,the best std acc is 7.095725181324942%

the last mean acc is 76.14819999999999%,the last std acc is 8.566817924994087%
the best mean acc is 79.2592%,the best std acc is 8.261748178200543%

the last mean acc is 76.14806666666667%,the last std acc is 8.372744866264322%
the best mean acc is 79.11113333333334%,the best std acc is 8.385728033324769%

the last mean acc is 76.44446666666667%,the last std acc is 9.696503636993192%
the best mean acc is 79.11100000000002%,the best std acc is 8.807021146032673%
'''



"""
the last mean acc is 81.03713333333334%,the last std acc is 7.861702282726194%
the best mean acc is 83.25933333333334%,the best std acc is 7.86152514182557%

the last mean acc is 81.92593333333335%,the last std acc is 7.734902925197073%
the best mean acc is 84.14806666666667%,the best std acc is 7.114144881540968%

the last mean acc is 81.62953333333333%,the last std acc is 6.859701294436143%
the best mean acc is 83.85186666666667%,the best std acc is 6.615330416204135%

the last mean acc is 80.59253333333332%,the last std acc is 7.233548883424296%
the best mean acc is 83.11106666666667%,the best std acc is 6.730510109114234%

the last mean acc is 82.07386666666666%,the last std acc is 6.413343775459275%
the best mean acc is 84.0%,the best std acc is 6.450829538387549%
==============================================================================

the last mean acc is 80.74059999999999%,the last std acc is 6.724006908582608%
the best mean acc is 83.1112%,the best std acc is 6.379004145894039%

the last mean acc is 80.59253333333332%,the last std acc is 7.095808695905554%
the best mean acc is 82.3704%,the best std acc is 7.886729113305887%

the last mean acc is 81.18506666666667%,the last std acc is 7.7349718764122795%
the best mean acc is 83.11113333333334%,the best std acc is 7.428094379822833%

the last mean acc is 80.14806666666667%,the last std acc is 7.84487055739113%
the best mean acc is 83.11113333333333%,the best std acc is 6.970885490061902

the last mean acc is 79.70379999999999%,the last std acc is 7.431150249232392%
the best mean acc is 82.07406666666665%,the best std acc is 6.8597862402717915%
"""




