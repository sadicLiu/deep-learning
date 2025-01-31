import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# %% 指数衰减函数的定义
torch.manual_seed(1)


def exp_w_func(beta, time_list):
    # β^t * (1-β)
    return [(1 - beta) * np.power(beta, exp) for exp in time_list]


beta = 0.9
num_point = 100
time_list = np.arange(num_point).tolist()

# %% exponential weight
weights = exp_w_func(beta, time_list)

plt.plot(time_list, weights, '-ro', label="Beta: {}\ny = β^t * (1-β)".format(beta))
plt.xlabel("time")
plt.ylabel("weight")
plt.legend()
plt.title("exponentially weighted average")
plt.show()

print(np.sum(weights))

# %% multi weights
beta_list = [0.98, 0.95, 0.9, 0.8]
w_list = [exp_w_func(beta, time_list) for beta in beta_list]
for i, w in enumerate(w_list):
    plt.plot(time_list, w, label="Beta: {}".format(beta_list[i]))
    plt.xlabel("time")
    plt.ylabel("weight")
plt.legend()
plt.show()


# %% SGD momentum
def func(x):
    return torch.pow(2 * x, 2)  # y = (2x)^2 = 4*x^2        dy/dx = 8x


iteration = 100
m = 0.9  # .9 .63

lr_list = [0.01, 0.03]

momentum_list = list()
loss_rec = [[] for l in range(len(lr_list))]
iter_rec = list()

for i, lr in enumerate(lr_list):
    x = torch.tensor([2.], requires_grad=True)

    momentum = 0. if lr == 0.03 else m
    momentum_list.append(momentum)

    optimizer = optim.SGD([x], lr=lr, momentum=momentum)

    for iter in range(iteration):
        y = func(x)
        y.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_rec[i].append(y.item())

for i, loss_r in enumerate(loss_rec):
    plt.plot(range(len(loss_r)), loss_r, label="LR: {} M:{}".format(lr_list[i], momentum_list[i]))
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss value')
plt.show()
