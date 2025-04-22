# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import numpy as np
import SimpleITK as sitk

base_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData/070_shenxinlin'

nrrd_image_filename = [x for x in os.listdir(os.path.join(base_dir_path)) if x.endswith('.nrrd')][0]
nrrd_image_path = os.path.join(base_dir_path, nrrd_image_filename)

sitk_image = sitk.ReadImage(nrrd_image_path)

# 原始坐标
origin = sitk_image.GetOrigin()
print('原点坐标：', origin)

# 图像体素
spacing = sitk_image.GetSpacing()
print('图像体素：', spacing)

# 图像大小
shape = sitk_image.GetSize()
print('图像大小：', shape)

# 关键点标注文件
all_json_landmark_filename = [x for x in os.listdir(os.path.join(base_dir_path)) if x.endswith('.json')]

# 关键点的真实物理坐标
for json_landmark_filename in all_json_landmark_filename:
    # 读取对应的标注文件
    with open(os.path.join(base_dir_path, json_landmark_filename), mode='r', encoding='utf-8') as f:
        landmark_dict = json.load(f)
    # 解析出物理坐标
    markup = landmark_dict['markups'][0]
    landmark_position = markup['controlPoints'][0]['position']
    # 判断坐标系统方向
    coordinate_system = markup['coordinateSystem']
    if coordinate_system != 'LPS':
        raise ValueError('请认真检查标注文件的坐标系统是否为 LPS 系统，避免坐标转换发生错误！')
    # 物理坐标转体素坐标
    voxel_coord = (np.asarray(landmark_position) - np.asarray(origin)) / (np.asarray(spacing))
    print(json_landmark_filename, voxel_coord)


