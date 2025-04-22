import os
import tqdm
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

# 加载原始的CBCT图像路径
load_origin_image_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'

# 加载72*72*72牙齿分割结果图路径
load_72_tooth_seg_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/tooth_region'

# 保存原始图像大小的牙齿分割结果路径
save_origin_tooth_seg_dir_path = '/home/user16/sharedata/GXE/SkullWidth/data/origin_tooth_region'

for patient_name in tqdm.tqdm(os.listdir(load_72_tooth_seg_dir_path)):
    number, name, pred = patient_name.split("_")
    # 加载 Nrrd 图像
    nrrd_image_filename = \
        [x for x in os.listdir(os.path.join(load_origin_image_dir_path,  number + '_' + name)) if x.endswith('.nrrd')][0]
    nrrd_image_file_path = os.path.join(load_origin_image_dir_path,  number + '_' + name, nrrd_image_filename)
    sitk_image = sitk.ReadImage(nrrd_image_file_path)
    sitk_array = sitk.GetArrayFromImage(sitk_image)
    # 加载 72*72*72 分割结果
    data_72_itkimage = sitk.ReadImage(os.path.join(load_72_tooth_seg_dir_path,  number + '_' + name + '_' + pred))
    array_72 = sitk.GetArrayFromImage(data_72_itkimage)
    print(np.unique(array_72))
    # 计算缩放系数
    ori_x = sitk_array.shape[0]
    ori_y = sitk_array.shape[1]
    ori_z = sitk_array.shape[2]
    a_x = ori_x / 72
    a_y = ori_y / 72
    a_z = ori_z / 72
    # 缩放矩阵
    zoomed_image = zoom(array_72, zoom=(a_x, a_y, a_z), order=0)
    print(np.unique(zoomed_image))
    # 将矩阵保存为 nii.gz 图像
    image = sitk.GetImageFromArray(zoomed_image)
    image.CopyInformation(sitk_image)
    sitk.WriteImage(image, os.path.join(save_origin_tooth_seg_dir_path,  number + '_' + name + ".nii.gz"))

# # 原始二值图像
# image = np.array([
#     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
# ])
#
# # 设置插值倍数
# zoom_factor = 2
#
# # 进行最近邻插值
# zoomed_image = zoom(image, zoom=(zoom_factor, zoom_factor, zoom_factor), order=0)
#
# # 输出插值后的图像
# print(zoomed_image)
