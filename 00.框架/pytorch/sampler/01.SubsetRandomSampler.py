import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self):
        self.data = np.arange(21)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)


trainset = MyDataset()

indices = np.arange(len(trainset))
# np.random.shuffle(indices)
split = len(indices) // 3

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

train_loader = DataLoader(dataset=trainset, sampler=train_sampler)
valid_loader = DataLoader(dataset=trainset, sampler=valid_sampler)

for data, label in valid_loader:
    print(data, ": ", label)

valid_samples = []
for i in valid_sampler:
    valid_samples.append(i)
print(valid_samples)
