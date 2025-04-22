import os
import tqdm
import json
import nrrd
import random
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

tooth_region_path = '/home/user16/sharedata/GXE/SkullWidth/data/tooth_region'
root_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_tooth_region'
origin_root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
save_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_heatmap_resize'

save_origin_crop_image_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_origin_image_and_heatmap'
if not os.path.exists(save_origin_crop_image_path):
    os.makedirs(save_origin_crop_image_path)

def generate_3d_heatmap(prev_heatmap, z0, x0, y0):
    """生成3d的高斯热图"""
    radius = 7
    diameter = 2 * radius + 1
    shape = (diameter, diameter, diameter)
    sigma = 1.5
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
    masked_heatmap = prev_heatmap[int(z0 - z_min):int(z0 + z_max), int(x0 - x_min):int(x0 + x_max), int(y0 - y_min):int(y0 + y_max)]
    masked_gaussian = h[int(radius - z_min):int(radius + z_max), int(radius - x_min):int(radius + x_max), int(radius - y_min):int(radius + y_max)]

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

# 遍历裁剪出来的图像文件夹
for patient_name in os.listdir(os.path.join(root_path)):
    number, name, pred = patient_name.split("_")
    print(number, "begin")

    # 提取原始图像对应的关键点标注坐标，文件后缀为json后缀
    all_json_landmark_filename = [x for x in os.listdir(os.path.join(origin_root_path, number + '_' + name)) if x.endswith('.json')]

    # 提取到对应的 nrrd 后缀的图像文件
    nrrd_image_filename = [x for x in os.listdir(os.path.join(origin_root_path, number + '_' + name)) if x.endswith('.nrrd')][0]
    nrrd_image_file_path = os.path.join(origin_root_path, number + '_' + name, nrrd_image_filename)

    image, img_header = nrrd.read(nrrd_image_file_path)

    print(img_header)