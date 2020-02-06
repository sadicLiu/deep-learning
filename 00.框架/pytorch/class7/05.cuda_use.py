import torch
import torch.nn as nn

# %% device

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# %% tensor to cuda

# 非inplace操作,重新创建变量
x_cpu = torch.ones((3, 3))
print("x_cpu:\n\tdevice: {}, is_cuda: {}, id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))

x_gpu = x_cpu.to(device)
print("x_gpu:\n\tdevice: {}, is_cuda: {}, id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))

# 弃用
# x_gpu = x_cpu.cuda()

# %% module to cuda

# inplace操作
net = nn.Sequential(nn.Linear(3, 3))
print("\nid:{}, is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

net.to(device)
print("\nid:{}, is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

# ========================== forward in cuda
output = net(x_gpu)
print("output is_cuda: {}".format(output.is_cuda))

# output = net(x_cpu)

# %% 查看当前gpu 序号，尝试修改可见gpu，以及主gpu

current_device = torch.cuda.current_device()
print("current_device: ", current_device)

torch.cuda.set_device(1)
current_device = torch.cuda.current_device()
print("current_device: ", current_device)

cap = torch.cuda.get_device_capability(device=None)
print("capability: ", cap)

name = torch.cuda.get_device_name()
print("device name: ", name)

is_available = torch.cuda.is_available()
print("is avaliable: ", is_available)

num_device = torch.cuda.device_count()
print("device number: ", num_device)

# %% seed

seed = 2
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

current_seed = torch.cuda.initial_seed()
print(current_seed)

s = torch.cuda.seed()
s_all = torch.cuda.seed_all()
