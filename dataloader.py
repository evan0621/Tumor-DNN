import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(self, txtdir):
        self.file = open(txtdir, 'r')
        self.datalist = self.file.readlines()

    def __getitem__(self, item):
        data = self.datalist[item]
        data = data.split(' ')
        input_data = np.array(data[5:-1], dtype='float')
        intensor = torch.from_numpy(input_data)
        intensor = Variable(intensor).float().cuda()
        label = np.array(data[0], dtype='int')
        label_tensor = torch.from_numpy(label)
        label_tensor = Variable(label_tensor).cuda()

        return intensor, label_tensor

    def __len__(self):
        return len(self.datalist)
