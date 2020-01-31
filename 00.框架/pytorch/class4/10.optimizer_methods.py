import os

import torch
import torch.optim as optim
from common_tools import set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

set_seed(1)  # 设置随机种子

weight = torch.randn((2, 2), requires_grad=True)
weight.grad = torch.ones((2, 2))

optimizer = optim.SGD([weight], lr=0.1)

# ----------------------------------- step -----------------------------------
# 通过调用 optimizer.step() 来更新参数
flag = 0
# flag = 1
if flag:
    print("weight before step:\n\t{}".format(weight.data))
    optimizer.step()  # 修改lr=1 0.1观察结果
    print("weight after step:\n\t{}".format(weight.data))

# ----------------------------------- zero_grad -----------------------------------
flag = 0
# flag = 1
if flag:
    print("weight before step:\n\t{}".format(weight.data))
    optimizer.step()  # 修改lr=1 0.1观察结果
    print("weight after step:\n\t{}".format(weight.data))

    print("\nweight memory addr in optimizer:{}".format(id(optimizer.param_groups[0]['params'][0])))
    print("weight memory addr in weight:{}\n".format(id(weight)))

    print("weight.grad before zero_grad() is \n\t{}".format(weight.grad))
    optimizer.zero_grad()
    print("weight.grad after optimizer.zero_grad() is\n\t{}".format(weight.grad))

# ----------------------------------- add_param_group -----------------------------------
flag = 0
# flag = 1
if flag:
    print("optimizer.param_groups is:")
    for a_group in optimizer.param_groups:
        print("\t{}".format(a_group))

    w2 = torch.ones((2, 3), requires_grad=True)
    optimizer.add_param_group({"params": w2, 'lr': 0.0001})

    print("\noptimizer.param_groups is:")
    for a_group in optimizer.param_groups:
        print("\t{}".format(a_group))

# ----------------------------------- state_dict -----------------------------------
flag = 0
# flag = 1
if flag:
    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n\t", opt_state_dict)

    for i in range(10):
        optimizer.step()

    print("state_dict after step:\n\t", optimizer.state_dict())

    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

# -----------------------------------load state_dict -----------------------------------
flag = 0
# flag = 1
if flag:
    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

    print("state_dict before load state:\n\t", optimizer.state_dict())
    optimizer.load_state_dict(state_dict)
    print("state_dict after load state:\n\t", optimizer.state_dict())
