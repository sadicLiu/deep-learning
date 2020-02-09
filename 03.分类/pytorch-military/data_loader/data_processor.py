import os
import shutil
from pathlib import Path


# %% 
def get_roi_data(data_path='/media/liuhy/data/数据集/ImageNet'):
    path_dst = Path.cwd().parent / 'data/ImageNet'
    path_label_file = Path(data_path) / 'labels'
    path_imgs = Path(data_path) / 'ILSVRC2012_img_train'

    tars = []
    with open(path_label_file) as f:
        line = f.readline()
        while line:
            tar_name = line.split(' ')[0] + '.tar'
            tars.append(tar_name)
            line = f.readline()

    for a_tar in tars:
        src = path_imgs / a_tar
        dst = path_dst / a_tar
        shutil.copyfile(src, dst)
    print('done copy')


get_roi_data()

# %%
