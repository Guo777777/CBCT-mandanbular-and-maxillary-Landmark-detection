# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import tqdm
import json
import random
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch_input):
        for t in self.transforms:
            batch_input = t(**batch_input)
        return batch_input


class SkullWidthCBCTDataset(Dataset):
    def __init__(self, data_root_dir, txt_file_path, zoom_target_size=(144, 48, 40)):
        self.data_root_dir = data_root_dir
        with open(txt_file_path, mode='r', encoding='utf-8') as f:
            all_case_list = f.readlines()
        self.all_case_list = all_case_list
        self.zoom_target_size = zoom_target_size

    def __getitem__(self, index):
        # 获取对应的文件名 以及 对应的标签
        case_name = self.all_case_list[index].strip()
        #
        # # 提取到对应的 nrrd 后缀的图像文件
        # # nrrd_image_filename = \
        # #     [x for x in os.listdir(os.path.join(self.data_root_dir, case_name)) if x.endswith('.nrrd')][0]
        # # nrrd_image_file_path = os.path.join(self.data_root_dir, case_name, nrrd_image_filename)
        nrrd_image_file_path = os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/four_point_save_128',
                                            case_name + "_image.nii.gz")
        sitk_image = sitk.ReadImage(nrrd_image_file_path)
        image_array = sitk.GetArrayFromImage(sitk_image)

        heatmap_array = np.zeros_like(image_array)
        # 获取文件夹下的所有子文件夹
        all_json_landmark_filename = [x for x in os.listdir(os.path.join('/home/user16/sharedata/GXE/SkullWidth/data'
                                                                         '/four_point_save_128')) if
                                      "heatmap" in x and x.startswith(case_name)]
        for json_landmark_filename in all_json_landmark_filename:
            # 热图重叠生成
            heatmap_image = sitk.ReadImage(os.path.join('/home/user16/sharedata/GXE/SkullWidth/data'
                                                        '/four_point_save_128', json_landmark_filename))
            heatmap_array1 = sitk.GetArrayFromImage(heatmap_image)
            heatmap_array = np.maximum(heatmap_array1, heatmap_array)

        # 直接缩放（0填充）到统一的大小，
        # z_dim, x_dim, y_dim = np.shape(image_array)
        # z_scale, x_scale, y_scale = self.zoom_target_size[0] / z_dim, self.zoom_target_size[1] / x_dim, \
        #                             self.zoom_target_size[2] / y_dim
        # zoom_image_array = ndimage.zoom(image_array, (z_scale, x_scale, y_scale))
        zoom_image_array = image_array

        # 最大最小值归一化操作，分母加上一个epsilon防止出现除0的出现
        epsilon = 1e-5
        image_array = zoom_image_array / (np.max(zoom_image_array) - np.min(zoom_image_array) + epsilon)

        # 直接缩放（0填充）到统一的大小，
        # z_dim, x_dim, y_dim = np.shape(heatmap_array)
        # z_scale, x_scale, y_scale = self.zoom_target_size[0] / z_dim, self.zoom_target_size[1] / x_dim, \
        #                             self.zoom_target_size[2] / y_dim
        # zoom_heatmap_array = ndimage.zoom(heatmap_array, (z_scale, x_scale, y_scale))
        zoom_heatmap_array = heatmap_array

        # 最大最小值归一化操作，分母加上一个epsilon防止出现除0的出现
        # epsilon = 1e-5
        # heatmap_array = zoom_heatmap_array / (np.max(zoom_heatmap_array) - np.min(zoom_heatmap_array) + epsilon)

        # # 提取对应的关键点标注坐标，文件后缀为json后缀
        # # 获取文件夹下的所有子文件夹
        # subfolders = [f.path for f in os.scandir(os.path.join(self.data_root_dir, case_name)) if f.is_dir()]
        # all_json_landmark_filename = [x for x in subfolders if x.endswith('.json')]
        #
        # # 获取到图片的spacing大小获取到才能从真实的物理坐标转换为体素坐标
        # sitk_image_spacing = sitk_image.GetSpacing()

        # # 加载已生成的热图
        # heatmap_array = np.zeros_like(image_array)
        # for json_landmark_filename in all_json_landmark_filename:
        #     # 读取对应的标注文件
        #     with open(os.path.join(self.data_root_dir, case_name, json_landmark_filename), mode='r',
        #               encoding='utf-8') as f:
        #         landmark_dict = json.load(f)
        #     # 解析出物理坐标
        #     markup = landmark_dict['markups'][0]
        #     landmark_position = markup['controlPoints'][0]['position']
        #     # 判断坐标系统方向
        #     coordinate_system = markup['coordinateSystem']
        #     if coordinate_system != 'LPS':
        #         raise ValueError('请认真检查标注文件的坐标系统是否为 LPS 系统，避免坐标转换发生错误！')
        #     # 物理坐标转体素坐标
        #     z = landmark_position[2] / sitk_image_spacing[0]
        #     x = landmark_position[0] / sitk_image_spacing[1]
        #     y = landmark_position[1] / sitk_image_spacing[2]
        #     # 缩放到目标大小的位置
        #     z, x, y = z * z_scale, x * x_scale, y * y_scale
        #     # 生成对应的热图
        #     heatmap_array = self.generate_3d_heatmap(heatmap_array, z, y, x)

        # 增加一个通道维度
        image_array = image_array[np.newaxis, ...]
        heatmap_array = heatmap_array[np.newaxis, ...]

        patient_info = case_name

        return image_array, heatmap_array, patient_info

    def __len__(self):
        return len(self.all_case_list)

    def generate_3d_heatmap(self, prev_heatmap, z0, x0, y0):
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
        z_min, z_max = min(z0, radius), min(z_limit - z0, radius + 1)
        x_min, x_max = min(x0, radius), min(x_limit - x0, radius + 1)
        y_min, y_max = min(y0, radius), min(y_limit - y0, radius + 1)

        # 赋值操作
        masked_heatmap = prev_heatmap[int(z0 - z_min):int(z0 + z_max), int(x0 - x_min):int(x0 + x_max),
                         int(y0 - y_min):int(y0 + y_max)]
        masked_gaussian = h[int(radius - z_min):int(radius + z_max), int(radius - x_min):int(radius + x_max),
                          int(radius - y_min):int(radius + y_max)]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

        return prev_heatmap


if __name__ == '__main__':
    data_root_dir = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
    txt_file_path = '/home/user16/sharedata/GXE/SkullWidth/data/train.txt'
    dataset = SkullWidthCBCTDataset(data_root_dir, txt_file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 取出一个数据来观察一下，生成的热图是否正确
    for batch_input in dataloader:
        image = batch_input[0]
        heatmap = batch_input[1]
        np_image = image.cpu().numpy()[0, 0, ...]
        np_heatmap = heatmap.cpu().numpy()[0, 0, ...]
        print(np.shape(np_image))
        print(np.sum(np_heatmap))
        # 保存为SimpleITK的nii.gz格式，用于在ITK-SNAP中打开查看
        sitk_image = sitk.GetImageFromArray(np_image)
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_image, '/home/user16/sharedata/GXE/SkullWidth/data/1/1.nii.gz')
        sitk.WriteImage(sitk_heatmap, '/home/user16/sharedata/GXE/SkullWidth/data/1/2.nii.gz')
        break
