import os
import math
import json
import random
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import measure
import csv

tooth_region_path = '/home/user16/sharedata/GXE/SkullWidth/data/tooth_region'
origin_root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
landmark_path = '/home/user16/sharedata/GXE/SkullWidth/data/test_save'
save_origin_crop_image_path = '/home/user16/sharedata/GXE/SkullWidth/data/four_point_save_128'
csv_save_path = '/home/user16/sharedata/GXE/SkullWidth/data/landmark.csv'

if not os.path.exists(save_origin_crop_image_path):
    os.makedirs(save_origin_crop_image_path)


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file)


def generate_3d_heatmap(prev_heatmap, z0, x0, y0):
    """生成3d的高斯热图"""
    radius = 5
    diameter = 2 * radius + 1
    shape = (diameter, diameter, diameter)
    sigma = 2
    m, n, k = [(ss - 1) / 2 for ss in shape]
    zz, xx, yy = np.ogrid[-m:m + 1, -n:n + 1, -k:k + 1]
    # 生成热图
    h = np.exp(-(xx * xx + yy * yy + zz * zz) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    # 赋值范围
    z_limit, x_limit, y_limit = prev_heatmap.shape[:3]
    z_min, z_max = min(z0, radius), min(z_limit - z0 + 1, radius + 1)
    x_min, x_max = min(x0, radius), min(x_limit - x0 + 1, radius + 1)
    y_min, y_max = min(y0, radius), min(y_limit - y0 + 1, radius + 1)

    # 赋值操作
    masked_heatmap = prev_heatmap[int(z0 - z_min):int(z0 + z_max), int(x0 - x_min):int(x0 + x_max),
                     int(y0 - y_min):int(y0 + y_max)]
    masked_gaussian = h[int(radius - z_min):int(radius + z_max), int(radius - x_min):int(radius + x_max),
                      int(radius - y_min):int(radius + y_max)]

    # Ensure dimensions are compatible for element-wise comparison
    masked_gaussian = masked_gaussian[:masked_heatmap.shape[0], :masked_heatmap.shape[1], :masked_heatmap.shape[2]]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    print(np.sum(masked_heatmap), np.sum(prev_heatmap))
    return prev_heatmap


def get_mask_bounds(mask):
    # 找到mask中所有非零元素的索引
    indices = np.nonzero(mask)
    # 在每个维度上找到最大和最小的索引
    min_x, max_x = np.min(indices[1]), np.max(indices[1])
    min_y, max_y = np.min(indices[2]), np.max(indices[2])
    min_z, max_z = np.min(indices[0]), np.max(indices[0])
    return min_x, max_x, min_y, max_y, min_z, max_z


def crop_image(image, min_x, max_x, min_y, max_y, min_z, max_z):
    # 使用numpy数组的切片功能截取图像
    return image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]


# case_json_list = []
# z_list = []
# y_list = []
# x_list = []
# 遍历裁剪出来的图像文件夹
whole_list = []
for patient_name in sorted(os.listdir(os.path.join(landmark_path))):
    number, name, pred = patient_name.split("_")
    data_list = []
    data_list.append(number + '_' + name)
    if "label" in patient_name:
        continue
    print(number, "begin")

    # 提取原始图像对应的关键点标注坐标，文件后缀为json后缀
    # # 获取文件夹下的所有子文件夹
    # subfolders = [f.path for f in os.scandir(os.path.join(origin_root_path, number + '_' + name)) if f.is_dir()]
    all_json_landmark_filename = [x for x in os.listdir(os.path.join(origin_root_path, number + '_' + name)) if
                                  x.endswith('.json')]

    # 提取到对应的 nrrd 后缀的图像文件
    nrrd_image_filename = \
        [x for x in os.listdir(os.path.join(origin_root_path, number + '_' + name)) if x.endswith('.nrrd')][0]
    nrrd_image_file_path = os.path.join(origin_root_path, number + '_' + name, nrrd_image_filename)
    sitk_image = sitk.ReadImage(nrrd_image_file_path)

    # 原始坐标
    origin = sitk_image.GetOrigin()
    print('原点坐标：', origin)

    # 图像体素
    # YXZ
    spacing = sitk_image.GetSpacing()
    print('图像体素：', spacing)

    # 图像大小
    # YXZ
    shape = sitk_image.GetSize()
    print('图像GetSize大小：', shape)

    origin_image_array = sitk.GetArrayFromImage(sitk_image)
    # 读取原数据的图像shape
    origin_image_z_shape, origin_image_x_shape, origin_image_y_shape = np.shape(origin_image_array)

    # 关键点的真实物理坐标
    for i, json_landmark_filename in enumerate(all_json_landmark_filename):
        # 读取对应的标注文件
        with open(os.path.join(origin_root_path, number + '_' + name, json_landmark_filename), mode='r',
                  encoding='utf-8') as f:
            landmark_dict = json.load(f)
        # 解析出物理坐标
        markup = landmark_dict['markups'][0]
        if 'controlPoints' in markup and len(markup['controlPoints']) > 0:
            landmark_position = markup['controlPoints'][0]['position']
            # 在这里可以使用 landmark_position 进行进一步的处理
        else:
            # 处理列表为空的情况
            continue
        # 判断坐标系统方向
        coordinate_system = markup['coordinateSystem']
        if coordinate_system != 'LPS':
            raise ValueError('请认真检查标注文件的坐标系统是否为 LPS 系统，避免坐标转换发生错误！')
        # 物理坐标转体素坐标
        voxel_coord = (np.asarray(landmark_position) - np.asarray(origin)) / (np.asarray(spacing))
        print(json_landmark_filename, voxel_coord)

        # TODO: 注意这里，第三个位置才是Z轴方向
        origin_landmark_y = voxel_coord[0]
        origin_landmark_x = voxel_coord[1]
        origin_landmark_z = voxel_coord[2]

        data_list.append(origin_landmark_y)
        data_list.append(origin_landmark_x)
        data_list.append(origin_landmark_z)

    # 关键点的真实物理坐标
    other_json_landmark_filename = [x for x in os.listdir(os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/json_save')) if
                                    number in x]
    for json_landmark_filename in other_json_landmark_filename:
        # 读取对应的标注文件
        with open(os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/json_save', json_landmark_filename), mode='r',
                  encoding='utf-8') as f:
            landmark_dict = json.load(f)
        # 解析出物理坐标
        markup = landmark_dict['markups'][0]
        if 'controlPoints' in markup and len(markup['controlPoints']) > 0:
            landmark_position = markup['controlPoints'][0]['position']
            # 在这里可以使用 landmark_position 进行进一步的处理
        else:
            # 处理列表为空的情况
            continue
        # 判断坐标系统方向
        coordinate_system = markup['coordinateSystem']
        if coordinate_system != 'LPS':
            raise ValueError('请认真检查标注文件的坐标系统是否为 LPS 系统，避免坐标转换发生错误！')
        # 物理坐标转体素坐标
        voxel_coord = (np.asarray(landmark_position) - np.asarray(origin)) / (np.asarray(spacing))
        print(json_landmark_filename, voxel_coord)

        # TODO: 注意这里，第三个位置才是Z轴方向
        origin_landmark_y = voxel_coord[0]
        origin_landmark_x = voxel_coord[1]
        origin_landmark_z = voxel_coord[2]

        data_list.append(origin_landmark_y)
        data_list.append(origin_landmark_x)
        data_list.append(origin_landmark_z)

    whole_list.append(data_list)
    print('>>' * 50)
    # save to csv
# 创建 header 列表
header_list = ["case", "F0-x", "F0-y", "F0-z", "F1-x", "F1-y", "F1-z", "F2-x", "F2-y", "F2-z", "F3-x", "F3-y", "F3-z",
               "F0-x-pred", "F0-y-pred",
               "F0-z-pred",
               "F1-x-pred", "F1-y-pred", "F1-z-pred", "F2-x-pred", "F2-y-pred", "F2-z-pred",
               "F3-x-pred", "F3-y-pred", "F3-z-pred"]
# 以写方式打开文件。注意添加 newline=""，否则会在两行数据之间都插入一行空白。
with open("/home/user16/sharedata/GXE/SkullWidth/data/landmark1.csv", mode="w", encoding="utf-8-sig", newline="") as f:
    # 基于打开的文件，创建 csv.writer 实例
    writer = csv.writer(f)

    # 写入 header。
    # writerow() 一次只能写入一行。
    writer.writerow(header_list)

    # 写入数据。
    # writerows() 一次写入多行。
    writer.writerows(whole_list)

    # # 保存对应的裁剪图像和热图
    # sitk_origin_crop_image = sitk.GetImageFromArray(zoom_image_array)
    # sitk.WriteImage(sitk_origin_crop_image,
    #                 os.path.join(save_origin_crop_image_path, number + "_" + name + "_image.nii.gz"))
    # sitk_origin_crop_image_heatmap = sitk.GetImageFromArray(origin_crop_image_heatmap)
    # sitk.WriteImage(sitk_origin_crop_image_heatmap,
    #                 os.path.join(save_origin_crop_image_path, number + "_" + name + "_heatmap_%s.nii.gz" % i))

# 保存对应关系为CSV文件，方便以后进行回顾性查看
# dict_data = {'case': case_json_list, 'z': z_list, 'y': y_list, 'x': x_list}
# df = pd.DataFrame(dict_data)
# df.to_csv(csv_save_path)
