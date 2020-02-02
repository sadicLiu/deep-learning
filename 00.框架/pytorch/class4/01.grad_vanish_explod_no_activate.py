import numpy as np
import torch
import torch.nn as nn

from common_tools import set_seed

"""使用100层的全连接网络来演示梯度消失与梯度爆炸的问题 
    从运行结果可以看到,对每一层的参数随机初始化之后,每一层输出结果的方差都在变大,
    这就导致梯度爆炸的产生.
    因此,要想解决梯度消失和梯度爆炸的问题,就要保证每层输出数据的标准差数值合理[方差一致性原则]
    
    对于全连接层,若神经元个数为n,使用标准正态分布初始化,可以推导出输出数据的方差为n
"""

set_seed(1)


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 1)随机初始化,会产生梯度爆炸
                # nn.init.normal_(m.weight.data)

                # 2)初始化时每个参数除以sqrt(n),保证每层输出的方差是1
                nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))  # normal: mean=0, std=1


layer_nums = 100
neural_nums = 256
batch_size = 16

net = MLP(neural_nums, layer_nums)
net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)
