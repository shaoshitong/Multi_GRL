import numpy as np
import sys,os
import scipy.io as sio
from Sampler.Dataset import TrainDataset
from torch.utils.data import DataLoader,Dataset
from Sampler.Sampler import RandomIdentitySampler
def get_data_deap(sample_path, subject_number, i, num_classes,batchsize=128):
    source_sample = []
    source_elabel = []
    people_source_elabel=[]
    print("train:", i)
    index = [j for j in range(subject_number)]
    for j in index:
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'person_%d label_V.npy' % j)
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    l2=source_elabel.shape[0]
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    source = TrainDataset(source_sample, source_elabel,i,subject_number,num_classes,training=True,name="seed")
    target = TrainDataset(source_sample, source_elabel,i,subject_number,num_classes,training=False,name="seed")
    source_loader = DataLoader(source, batch_size=batchsize, shuffle=True, drop_last=False)
    target_loader = DataLoader(target, batch_size=batchsize, shuffle=True, drop_last=False)
    return source_loader, target_loader

def get_data_seed(sample_path, subject_number, i, num_classes,batchsize=128):
    source_sample = []
    source_elabel = []
    people_source_elabel=[]
    print("train:", i)
    index = [j for j in range(subject_number)]
    for j in index:
        t_source_sample = np.load(sample_path + 'person_%d data.npy' % j)
        t_source_elabel = np.load(sample_path + 'label.npy')
        source_sample.append(t_source_sample)
        source_elabel.append(t_source_elabel)
    source_elabel = np.concatenate(source_elabel, axis=0)
    source_sample = np.concatenate(source_sample, axis=0)
    l2=source_elabel.shape[0]
    print("source 3:{:.2f}%,2:{:.2f}%,1:{:.2f}%,0:{:.2f}%".format(np.sum(source_elabel == 3)/l2,
          np.sum(source_elabel == 2)/l2,
          np.sum(source_elabel == 1)/l2,
          np.sum(source_elabel == 0)/l2))
    source = TrainDataset(source_sample, source_elabel,i,subject_number,num_classes,training=True,name="seed")
    target = TrainDataset(source_sample, source_elabel,i,subject_number,num_classes,training=False,name="seed")
    nums_instance=int(batchsize//subject_number)
    batchsize=nums_instance*subject_number
    Sampler=RandomIdentitySampler(source,subject_number,nums_instance)
    source_loader = DataLoader(source, sampler=Sampler,batch_size=batchsize, drop_last=False)
    target_loader = DataLoader(target, batch_size=batchsize, shuffle=True, drop_last=False)
    return source_loader, target_loader