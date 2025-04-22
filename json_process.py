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
size_list = [0, 0, 0]
for patient_name in sorted(os.listdir(os.path.join(origin_root_path))):
    number, name = patient_name.split("_")
    data_list = []
    data_list.append(number + '_' + name)
    # if "label" in patient_name:
    #     continue
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
    shape_list = list(shape)
    k = 0
    for i in shape_list:
        size_list[k] += i
        k += 1

    # origin_image_array = sitk.GetArrayFromImage(sitk_image)
    # # 读取原数据的图像shape
    # origin_image_z_shape, origin_image_x_shape, origin_image_y_shape = np.shape(origin_image_array)
    #
    # # 关键点的真实物理坐标
    # for i, json_landmark_filename in enumerate(all_json_landmark_filename):
    #     # 读取对应的标注文件
    #     with open(os.path.join(origin_root_path, number + '_' + name, json_landmark_filename), mode='r',
    #               encoding='utf-8') as f:
    #         landmark_dict = json.load(f)
    #     # 解析出物理坐标
    #     markup = landmark_dict['markups'][0]
    #     if 'controlPoints' in markup and len(markup['controlPoints']) > 0:
    #         landmark_position = markup['controlPoints'][0]['position']
    #         # 在这里可以使用 landmark_position 进行进一步的处理
    #     else:
    #         # 处理列表为空的情况
    #         continue
    #     # 判断坐标系统方向
    #     coordinate_system = markup['coordinateSystem']
    #     if coordinate_system != 'LPS':
    #         raise ValueError('请认真检查标注文件的坐标系统是否为 LPS 系统，避免坐标转换发生错误！')
    #     # 物理坐标转体素坐标
    #     voxel_coord = (np.asarray(landmark_position) - np.asarray(origin)) / (np.asarray(spacing))
    #     print(json_landmark_filename, voxel_coord)
    #
    #     # TODO: 注意这里，第三个位置才是Z轴方向
    #     origin_landmark_y = voxel_coord[0]
    #     origin_landmark_x = voxel_coord[1]
    #     origin_landmark_z = voxel_coord[2]
    #
    #     assert 0 <= origin_landmark_z < origin_image_z_shape
    #     assert 0 <= origin_landmark_x < origin_image_x_shape
    #     assert 0 <= origin_landmark_y < origin_image_y_shape
    #     # # TODO: 读取二值化的128*128*128的牙齿区域分割结果
    #     data_itkimage = sitk.ReadImage(os.path.join(tooth_region_path, patient_name))
    #     mask = sitk.GetArrayFromImage(data_itkimage)
    #     z_index, x_index, y_index = np.where(mask == 1)
    #     min_scale_tooth_z, max_scale_tooth_z = np.min(z_index), np.max(z_index)
    #     min_scale_tooth_x, max_scale_tooth_x = np.min(x_index), np.max(x_index)
    #     min_scale_tooth_y, max_scale_tooth_y = np.min(y_index), np.max(y_index)
    #
    #     # tooth_region本身存在缩放，要还原回去
    #     min_origin_tooth_z = int((min_scale_tooth_z / 72) * origin_image_z_shape)
    #     max_origin_tooth_z = int((max_scale_tooth_z / 72) * origin_image_z_shape)
    #     min_origin_tooth_x = int((min_scale_tooth_x / 72) * origin_image_x_shape)
    #     max_origin_tooth_x = int((max_scale_tooth_x / 72) * origin_image_x_shape)
    #     min_origin_tooth_y = int((min_scale_tooth_y / 72) * origin_image_y_shape)
    #     max_origin_tooth_y = int((max_scale_tooth_y / 72) * origin_image_y_shape)
    #
    #     assert 0 <= min_origin_tooth_z
    #     assert 0 <= min_origin_tooth_x
    #     assert 0 <= min_origin_tooth_y
    #     assert max_origin_tooth_z < origin_image_z_shape
    #     assert max_origin_tooth_x < origin_image_x_shape
    #     assert max_origin_tooth_y < origin_image_y_shape
    #
    #     # 稍微将粗分割结果稍微外扩10个体素，max和min操作是防止越界
    #     min_origin_tooth_z = max(0, min_origin_tooth_z - 10)
    #     max_origin_tooth_z = min(origin_image_z_shape, max_origin_tooth_z + 10)
    #     min_origin_tooth_x = max(0, min_origin_tooth_x - 10)
    #     max_origin_tooth_x = min(origin_image_x_shape, max_origin_tooth_x + 10)
    #     min_origin_tooth_y = max(0, min_origin_tooth_y - 10)
    #     max_origin_tooth_y = min(origin_image_y_shape, max_origin_tooth_y + 10)
    #     print('牙齿区域裁剪范围：', [min_origin_tooth_y, max_origin_tooth_y], [min_origin_tooth_x, max_origin_tooth_x],
    #           [min_origin_tooth_z, max_origin_tooth_z])
    #
    #     # 裁剪出原图上的图像
    #     origin_crop_image_z_size = max_origin_tooth_z - min_origin_tooth_z
    #     origin_crop_image_x_size = max_origin_tooth_x - min_origin_tooth_x
    #     origin_crop_image_y_size = max_origin_tooth_y - min_origin_tooth_y
    #     print('Crop图像大小：', origin_crop_image_y_size, origin_crop_image_x_size, origin_crop_image_z_size)
    #
    #     # 更新关键点的坐标
    #     new_landmark_z = origin_landmark_z - min_origin_tooth_z
    #     new_landmark_x = origin_landmark_x - min_origin_tooth_x
    #     new_landmark_y = origin_landmark_y - min_origin_tooth_y
    #     print('Crop图像坐标系下的关键点坐标：', new_landmark_y, new_landmark_x, new_landmark_z)
    #     assert 0 <= new_landmark_z < origin_crop_image_z_size
    #     assert 0 <= new_landmark_y < origin_crop_image_y_size
    #     assert 0 <= new_landmark_x < origin_crop_image_x_size
    #
    #     # TODO: 注意这里，传入生成热图的顺序是：z, y, x
    #     origin_crop_image = origin_image_array[min_origin_tooth_z:max_origin_tooth_z,
    #                         min_origin_tooth_x:max_origin_tooth_x, min_origin_tooth_y:max_origin_tooth_y]
    #     origin_crop_image_heatmap = np.zeros((128, 128, 128), dtype=float)
    #     origin_crop_image_heatmap = generate_3d_heatmap(origin_crop_image_heatmap,
    #                                                     (new_landmark_z / origin_crop_image_z_size) * 128,
    #                                                     (new_landmark_x / origin_crop_image_x_size) * 128,
    #                                                     (new_landmark_y / origin_crop_image_y_size) * 128)
    #     print('热图元素和：', np.sum(origin_crop_image_heatmap))
    #
    #     # 直接缩放（0填充）到统一的大小，
    #     z_dim, x_dim, y_dim = np.shape(origin_crop_image)
    #     z_scale, x_scale, y_scale = 128 / z_dim, 128 / x_dim, \
    #                                 128 / y_dim
    #     zoom_image_array = ndimage.zoom(origin_crop_image, (z_scale, x_scale, y_scale))
    #
    #     #####
    #     nrrd_image_file_path = os.path.join(landmark_path, patient_name)
    #     sitk_image = sitk.ReadImage(nrrd_image_file_path)
    #     array_image = sitk.GetArrayFromImage(sitk_image)
    #     # 卡阈值
    #     array_image[array_image <= 0.1] = 0
    #     array_image[array_image >= 0.1] = 1
    #     # 标记连通域
    #     labels = measure.label(array_image)
    #     # 获取连通区域的属性
    #     props = measure.regionprops(labels)
    #     # 获取每个连通区域的面积
    #     areas = [region.area for region in props]
    #     # 根据面积对区域进行排序
    #     sorted_regions = np.argsort(areas)[::-1]
    #     # 筛选出面积最大的四个区域
    #     top_regions = sorted_regions[:4]
    #
    #
    #     # 按左上、右上、左下、右下的顺序排序
    #     def custom_sort(region_idx):
    #         region = props[region_idx]
    #         centroid = region.centroid
    #         return centroid[0] + centroid[1]
    #
    #
    #     sorted_top_regions = sorted(top_regions, key=custom_sort)
    #     # 存储标签和中心点坐标的列表
    #     result_list = []
    #     # 输出筛选结果并存储到列表中
    #     for region_idx in sorted_top_regions:
    #         region = props[region_idx]
    #         centroid = list(region.centroid)
    #         centroid[0] = ((centroid[0] / 128) * origin_crop_image_y_size + min_origin_tooth_y) * spacing[0] + origin[0]
    #         centroid[1] = ((centroid[1] / 128) * origin_crop_image_x_size + min_origin_tooth_x) * spacing[1] + origin[1]
    #         centroid[2] = ((centroid[2] / 128) * origin_crop_image_z_size + min_origin_tooth_z) * spacing[2] + origin[2]
    #         result_list.append(centroid)
    # # 打印结果列表
    # print('>>' * 50)
    # num = 0
    # for item in result_list:
    #     # 随便打开一个
    #     with open(os.path.join(origin_root_path, '004' + '_' + 'lianghanbin', 'F.mrk.json'), mode='r',
    #               encoding='utf-8') as f:
    #         landmark_dict = json.load(f)
    #     # 解析出物理坐标
    #     landmark_dict['markups'][0]['controlPoints'][0]['position'] = item
    #     # 将修改后的数据转换回 JSON 格式
    #     modified_json = json.dumps(landmark_dict, indent=4)
    #     # 保存为新文件
    #     new_filename = os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/json_save',
    #                                 number + '_' + name + '_' + 'F' + '_' + str(num) + "_pred.mrk.json")
    #     with open(new_filename, 'w') as file:
    #         file.write(modified_json)
    #     print(item)
    #     distance0 = []
    #     # 关键点的真实物理坐标
    #     for i, json_landmark_filename in enumerate(all_json_landmark_filename):
    #         # 读取对应的标注文件
    #         with open(os.path.join(origin_root_path, number + '_' + name, json_landmark_filename), mode='r',
    #                   encoding='utf-8') as f:
    #             landmark_dict = json.load(f)
    #         # 解析出物理坐标
    #         markup = landmark_dict['markups'][0]
    #         if 'controlPoints' in markup and len(markup['controlPoints']) > 0:
    #             landmark_position = markup['controlPoints'][0]['position']
    #         # 计算欧几里得距离
    #         distance = math.sqrt(
    #             (landmark_position[0] - item[0]) ** 2 + (landmark_position[1] - item[1]) ** 2 + (
    #                     landmark_position[2] - item[2]) ** 2)
    #         distance0.append(distance)
    #     print("与金标准的距离为:", min(distance0))
    #     data_list.append(min(distance0))
    #     num += 1
    # distance1 = []
    # distance1.append(math.sqrt(
    #     (result_list[0][0] - result_list[1][0]) ** 2 + (result_list[0][1] - result_list[1][1]) ** 2 + (
    #             result_list[0][2] - result_list[1][2]) ** 2))
    # distance1.append(math.sqrt(
    #     (result_list[2][0] - result_list[3][0]) ** 2 + (result_list[2][1] - result_list[3][1]) ** 2 + (
    #             result_list[2][2] - result_list[3][2]) ** 2))
    # for i in distance1:
    #     print("所需的距离指标：", i)
    #     data_list.append(i)
    #
    # whole_list.append(data_list)
    print('>>' * 50)
print(size_list)
# # save to csv
# # 创建 header 列表
# header_list = ["case", "F0与金标准的距离", "F1与金标准的距离", "F2与金标准的距离", "F3与金标准的距离",
#                "距离指标1与金标准差值", "距离指标2与金标准差值"]
# # 以写方式打开文件。注意添加 newline=""，否则会在两行数据之间都插入一行空白。
# with open("/home/user16/sharedata/GXE/SkullWidth/data/landmark.csv", mode="w", encoding="utf-8-sig", newline="") as f:
#     # 基于打开的文件，创建 csv.writer 实例
#     writer = csv.writer(f)
#
#     # 写入 header。
#     # writerow() 一次只能写入一行。
#     writer.writerow(header_list)
#
#     # 写入数据。
#     # writerows() 一次写入多行。
#     writer.writerows(whole_list)

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
