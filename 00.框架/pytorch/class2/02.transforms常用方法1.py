import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from my_dataset import RMBDataset
from utils import *

set_seed(1)  # 设置随机种子

# 参数设置
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}

# ============================ step 1/5 数据 ============================
split_dir = "/home/liuhy/res/deep-learning/00.框架/pytorch/class2/01.rmb/dataset/rmb_split"
train_dir = os.path.join(split_dir, "train")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(512),  # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop, 先随机裁剪(size随机,ratio随机)再resize到指定大小
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# ============================ step 5/5 训练 ============================
for i, data in enumerate(train_loader):
    inputs, labels = data  # B C H W

    img_tensor = inputs[0, ...]  # C H W
    img = transform_invert(img_tensor, train_transform)
    plt.imshow(img)
    plt.show()
    plt.pause(0.5)
    plt.close()

    # bs, ncrops, c, h, w = inputs.shape
    # for n in range(ncrops):
    #     img_tensor = inputs[0, n, ...]  # C H W
    #     img = transform_invert(img_tensor, train_transform)
    #     plt.imshow(img)
    #     plt.show()
    #     plt.pause(1)

    if i > 5: break
