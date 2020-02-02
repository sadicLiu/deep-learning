import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from common_tools import set_seed
import torchvision.models as models

# ----------------------------------- feature map visualization -----------------------------------
writer = SummaryWriter(comment='_lena', filename_suffix="_lena")

# 数据
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]

norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform
])

img_pil = Image.open("lena.png").convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)  # chw --> bchw

# 模型
alexnet = models.alexnet(pretrained=True)

# 注册hook
fmap_dict = dict()
for name, sub_module in alexnet.named_modules():

    if isinstance(sub_module, nn.Conv2d):
        key_name = str(sub_module.weight.shape)
        fmap_dict.setdefault(key_name, list())

        n1, n2 = name.split(".")


        def hook_func(m, i, o):
            key_name = str(m.weight.shape)
            fmap_dict[key_name].append(o)


        alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

# forward
output = alexnet(img_tensor)

# add image
for layer_name, fmap_list in fmap_dict.items():
    fmap = fmap_list[0]
    fmap.transpose_(0, 1)

    nrow = int(np.sqrt(fmap.shape[0]))
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=322)
