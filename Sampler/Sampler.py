import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity, randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, subject_nums, num_instances):
        self.data_source = data_source
        self.batch_size = subject_nums * num_instances
        self.num_instances = num_instances
        self.subject_nums=subject_nums
        assert self.batch_size % self.num_instances == 0, "batch size must exclude the number of subjects"
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, subject, _) in enumerate(self.data_source):
            self.index_dic[subject.item()].append(index)
        self.subjects = list(self.index_dic.keys())
        self.length = 0
        for subject in self.subjects:
            idxs = self.index_dic[subject]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for subject in self.subjects:
            idxs = copy.deepcopy(self.index_dic[subject])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[subject].append(batch_idxs)
                    batch_idxs = []
            if len(batch_idxs)!=0:
                batch_idxs_dict[subject].append(batch_idxs)
        avai_subjects = copy.deepcopy(self.subjects)
        final_idxs = []
        copy_avai_subjects=copy.deepcopy(avai_subjects)
        while len(avai_subjects)!=0:
            for subject in copy_avai_subjects:
                batch_idxs = batch_idxs_dict[subject].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[subject]) == 0:
                    avai_subjects.remove(subject)
        assert len(avai_subjects)==0,"there are subjects whose samples are not entered"
        self.length = len(final_idxs)
        return iter(final_idxs)
    def __len__(self):
        return self.length
