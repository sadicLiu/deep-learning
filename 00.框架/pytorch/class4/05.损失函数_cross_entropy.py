import numpy as np
import torch
import torch.nn as nn

# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- CrossEntropy loss: reduction -----------------------------------
# flag = 0
flag = 1
if flag:
    # def loss function
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print("Cross Entropy Loss:\n\tnone: {}\n\tsum: {}\n\tmean: {}".format(loss_none, loss_sum, loss_mean))

# ----------------------------------- weight(各个类别的loss使用不同权重) ---------------------------------
# flag = 0
flag = 1
if flag:
    # def loss function
    weights = torch.tensor([1, 2], dtype=torch.float)
    # weights = torch.tensor([0.7, 0.3], dtype=torch.float)

    loss_f_none_w = nn.CrossEntropyLoss(weight=weights, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print("Cross Entropy Loss with weights: ")
    print("\tweights: ", weights)
    print("\tnone: {}\n\tsum: {}\n\tmean: {}".format(loss_none, loss_sum, loss_mean))
