import torch
import torch.nn as nn

x = torch.randn(10000)
out = torch.tanh(x)

gain = x.std() / out.std()
print('gain:{}'.format(gain))

tanh_gain = nn.init.calculate_gain('tanh')
print('tanh_gain in PyTorch:', tanh_gain)