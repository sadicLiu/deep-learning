import numpy as np
import torch
import torch.nn as nn

# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- NLLLoss -----------------------------------
# 这个loss就是对inputs中的每个样本中真实标签的那个值取反
# flag = 0
flag = 1
if flag:
    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.NLLLoss(weight=weights, reduction='none')
    loss_f_sum = nn.NLLLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.NLLLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print("NLL Loss: ")
    print("\tweights: ", weights)
    print("\tnone: {}\n\tsum: {}\n\tmean: {}".format(loss_none_w, loss_sum, loss_mean))
