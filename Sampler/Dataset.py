from torch.utils.data import Dataset
import torch
import numpy as np
import random


class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, data):
        ''' img: Tensor image of size (C, H, W) '''
        _, h = data.shape
        mask = np.ones((h), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            mask[y1: y2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(data)
        return data * mask



class TrainDataset(Dataset):
    def __init__(self, data, label, test_index, subject_number,num_classes=2, training=True,name="seed"):
        super(Dataset,self).__init__()
        self.data = data
        self.label = label.astype(np.longlong)
        self.name=name
        self.num_classes = num_classes
        self.test_index=test_index
        self.training = training
        if self.name=="seed":
            self.subject_mark=np.load("./HelpFile/subject_label.npy").astype(float)
        elif self.name=="deap":
            raise NotImplementedError("Not import")
        else:
            raise NotImplementedError("Not import")
        self.one_subject_number=(self.data.shape[0]//subject_number)
        self.subject_label=torch.from_numpy(np.array([[i]*self.one_subject_number for i in range(subject_number)]).reshape(-1)).long()
        if self.training==False:
            sl=slice(test_index*self.one_subject_number,(test_index+1)*self.one_subject_number)
            self.data=self.data[sl]
            self.label=self.label[sl]
            self.subject_label=self.subject_label[sl]
            self.subject_mark=self.subject_mark[sl]
    def __getitem__(self, index):
        if self.training:
            sample = self.data[index, ...]
            sample = torch.Tensor(sample)
            label = torch.zeros(self.num_classes)
            label[self.label[index]] = 1
            subject_label=self.subject_label[index]
            subject_mark=torch.from_numpy(self.subject_mark[self.subject_label[index].item(),...]).float()
            return sample,label,subject_label,subject_mark
        else:
            sample = self.data[index, ...]
            sample = torch.Tensor(sample)
            label = torch.zeros(self.num_classes)
            label[self.label[index]] = 1
            return sample,label
    def __len__(self):
        return len(self.label)

