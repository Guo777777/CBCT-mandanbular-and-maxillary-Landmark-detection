from sklearn.cluster import KMeans

import numpy as np
import os
import SimpleITK as sitk
import json

tooth_region_path = '/home/user5/sharedata/GXE/SkullWidth/data/tooth_region'
origin_root_path = '/home/user5/sharedata/GXE/SkullWidth/data/imageStandardData'
save_path = '/home/user5/sharedata/GXE/SkullWidth/data/whole_heatmap_save'

for patient_name in sorted(os.listdir(os.path.join(tooth_region_path))):
    number, name, pred = patient_name.split("_")
    points = []
    print(number, "begin")

    # 获取文件夹下的所有子文件夹
    nrrd_image_file_path = os.path.join('/home/user5/sharedata/GXE/SkullWidth/data/crop_origin_image_and_heatmap01',
                                        number + '_' + name + "_image.nii.gz")
    sitk_image = sitk.ReadImage(nrrd_image_file_path)
    origin_image_array = sitk.GetArrayFromImage(sitk_image)

    # 获取文件夹下的所有子文件夹
    all_json_landmark_filename = [x for x in os.listdir(os.path.join('/home/user5/sharedata/GXE/SkullWidth/data'
                                                                     '/crop_origin_image_and_heatmap01')) if
                                  "heatmap" in x and x.startswith(number + '_' + name)]
    heatmap_array = np.zeros_like(origin_image_array)
    for json_landmark_filename in all_json_landmark_filename:
        # 热图重叠生成
        heatmap_image = sitk.ReadImage(os.path.join('/home/user5/sharedata/GXE/SkullWidth/data'
                                                    '/crop_origin_image_and_heatmap01', json_landmark_filename))
        heatmap_array1 = sitk.GetArrayFromImage(heatmap_image)
        heatmap_array = np.maximum(heatmap_array1, heatmap_array)

    # 保存对应的裁剪图像和热图
    sitk_origin_crop_image = sitk.GetImageFromArray(origin_image_array)
    sitk.WriteImage(sitk_origin_crop_image,
                    os.path.join(save_path, number + "_" + name + "_image.nii.gz"))
    sitk_origin_crop_image_heatmap = sitk.GetImageFromArray(heatmap_array)
    sitk.WriteImage(sitk_origin_crop_image_heatmap,
                    os.path.join(save_path, number + "_" + name + "_heatmap.nii.gz"))
