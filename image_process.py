import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage

root_path = '/home/user16/sharedata/GXE/SkullWidth/data/tooth_region'
origin_root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
save_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_tooth_region'


def get_mask_bounds(mask):
    # 找到mask中所有非零元素的索引
    indices = np.nonzero(mask)
    # 在每个维度上找到最大和最小的索引
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_z, max_z = np.min(indices[2]), np.max(indices[2])
    return min_x, max_x, min_y, max_y, min_z, max_z


def crop_image(image, min_x, max_x, min_y, max_y, min_z, max_z):
    # 使用numpy数组的切片功能截取图像
    return image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]


for patient_name in os.listdir(os.path.join(root_path)):
    try:
        number, name, pred = patient_name.split("_")
        print(number, "begin")
        # 读取原数据
        data_itkimage = sitk.ReadImage(os.path.join(root_path, patient_name))
        mask = sitk.GetArrayFromImage(data_itkimage)
        min_x, max_x, min_y, max_y, min_z, max_z = get_mask_bounds(mask)

        # 提取到对应的 nrrd 后缀的图像文件
        nrrd_image_filename = \
            [x for x in os.listdir(os.path.join(origin_root_path, number + '_' + name)) if x.endswith('.nrrd')][0]
        nrrd_image_file_path = os.path.join(origin_root_path, number + '_' + name, nrrd_image_filename)
        sitk_image = sitk.ReadImage(nrrd_image_file_path)
        image_array = sitk.GetArrayFromImage(sitk_image)
        x = image_array.shape[0]
        y = image_array.shape[1]
        z = image_array.shape[2]
        print(f"Image shape: {image_array.shape}")

        # 缩放回原image大小
        min_x = (min_x / 128) * x
        max_x = (max_x / 128) * x
        min_y = (min_y / 128) * y
        max_y = (max_y / 128) * y
        min_z = (min_z / 128) * z
        max_z = (max_z / 128) * z

        min_x = int(round(min_x))
        max_x = int(round(max_x))
        min_y = int(round(min_y))
        max_y = int(round(max_y))
        min_z = int(round(min_z))
        max_z = int(round(max_z))

        # 假设 image 是你的原始图像，下列函数调用将返回你需要的截取的图像
        cropped_image = crop_image(image_array, min_x, max_x, min_y, max_y, min_z, max_z)
        image = sitk.GetImageFromArray(cropped_image)
        sitk.WriteImage(image, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))
    except Exception as e:
        print(f"Error processing image: {patient_name}")
        print(f"Error message: {str(e)}")

        continue




root_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_tooth_region'
save_path = '/home/user16/sharedata/GXE/SkullWidth/data/crop_tooth_region_resize'
for patient_name in os.listdir(os.path.join(root_path)):
    number, name, pred = patient_name.split("_")
    print(number, "begin")
    data_itkimage = sitk.ReadImage(os.path.join(root_path, patient_name))
    data = sitk.GetArrayFromImage(data_itkimage)
    # 直接缩放（0填充）到统一的大小，
    z_dim, x_dim, y_dim = np.shape(data)
    z_scale, x_scale, y_scale = 128 / z_dim, 128 / x_dim, \
                                128 / y_dim
    zoom_image_array = ndimage.zoom(data, (z_scale, x_scale, y_scale))
    image = sitk.GetImageFromArray(zoom_image_array)
    sitk.WriteImage(image, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))
