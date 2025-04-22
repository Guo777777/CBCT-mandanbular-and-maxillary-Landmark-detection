# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random

# 存放数据的文件夹路径
load_data_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'

# 获取到对应的案例列表（加一个排序，防止在不同平台上的划分不一致）
all_case_list = os.listdir(load_data_dir_path)
all_case_list = sorted(all_case_list)

# 打乱列表
random.seed(2023)
random.shuffle(all_case_list)

# 划分训练-验证-测试集，并且要保存下来保证结果可以复现
save_split_file_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data'

# 这里49个数据，暂时以：30：9：10进行划分
train_list = all_case_list[:80]
valid_list = all_case_list[80:80+10]
test_list = all_case_list[80+10:]

# 写入到文件中，方便进行读取
with open(os.path.join(save_split_file_dir_path, 'train.txt'), mode='w', encoding='utf-8') as f:
    for filename in train_list:
        f.write('%s\n' % filename)

with open(os.path.join(save_split_file_dir_path, 'valid.txt'), mode='w', encoding='utf-8') as f:
    for filename in valid_list:
        f.write('%s\n' % filename)

with open(os.path.join(save_split_file_dir_path, 'test.txt'), mode='w', encoding='utf-8') as f:
    for filename in test_list:
        f.write('%s\n' % filename)

