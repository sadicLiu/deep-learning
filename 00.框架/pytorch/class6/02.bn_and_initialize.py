"""使用BN可以不用特别关注模型参数的初始化"""
import torch
import torch.nn as nn

from common_tools import set_seed

set_seed(1)  # 设置随机种子


# exp1: 不用BN,也不用初始化函数,模型输出值很小
# exp2: 不用BN,使用不太好的初始化方法(relu函数,标准正态分布),会产生nan
# exp3: 不用BN,使用kaiming初始化,输出值正常
# exp4: 使用BN,怎么都正常


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{}, std:{}".format(i, x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # method 1
                nn.init.normal_(m.weight.data, std=1)  # normal: mean=0, std=1

                # method 2 kaiming
                # nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)
