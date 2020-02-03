import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from common_tools import set_seed

set_seed(1)  # 设置随机种子

# %% SummaryWriter

# 指定了 log_dir 属性之后, comment 属性就失效了
writer = SummaryWriter(log_dir="./train_log", comment='_comment', filename_suffix="pow_2_x")
# writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")

for x in range(100):
    writer.add_scalar('y=pow_2_x', 2 ** x, x)

writer.close()

# %% scalar and scalars

max_epoch = 100

writer = SummaryWriter(log_dir="./log", filename_suffix="_test")

for x in range(max_epoch):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x)}, x)

writer.close()

# %% histogram

writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

for x in range(2):
    np.random.seed(x)

    data_union = np.arange(100)
    data_normal = np.random.normal(size=1000)

    writer.add_histogram('distribution union', data_union, x)
    writer.add_histogram('distribution normal', data_normal, x)

    plt.subplot(121).hist(data_union, label="union")
    plt.subplot(122).hist(data_normal, label="normal")
    plt.legend()
    plt.show()

writer.close()
