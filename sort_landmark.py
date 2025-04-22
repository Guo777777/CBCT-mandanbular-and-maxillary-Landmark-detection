from sklearn.cluster import KMeans

import numpy as np
import os
import SimpleITK as sitk
import json

tooth_region_path = '/home/user5/sharedata/GXE/SkullWidth/data/tooth_region'
origin_root_path = '/home/user5/sharedata/GXE/SkullWidth/data/imageStandardData'
save_data_path = '/home/user5/sharedata/GXE/SkullWidth/data/sort_three_region/data/G3'
save_label_path = '/home/user5/sharedata/GXE/SkullWidth/data/sort_three_region/label/G3'

for patient_name in sorted(os.listdir(os.path.join(tooth_region_path))):
    number, name, pred = patient_name.split("_")
    points = []
    print(number, "begin")

    # 获取文件夹下的所有子文件夹
    nrrd_image_file_path = os.path.join('/home/user5/sharedata/GXE/SkullWidth/data/crop_origin_image_and_heatmap01',
                                        number + '_' + name + "_image.nii.gz")
    sitk_image = sitk.ReadImage(nrrd_image_file_path)
    origin_image_array = sitk.GetArrayFromImage(sitk_image)
    # 读取原数据的图像shape
    origin_image_z_shape, origin_image_x_shape, origin_image_y_shape = np.shape(origin_image_array)
    origin_array_x_shape, origin_array_y_shape, origin_array_z_shape = np.shape(origin_image_array)
    # 图像大小
    # YXZ
    shape = sitk_image.GetSize()
    print('图像GetSize大小：', shape)
    array_shape = origin_image_array.shape
    print('图像array大小：', array_shape)

    all_json_landmark_filename = [x for x in os.listdir(os.path.join('/home/user5/sharedata/GXE/SkullWidth/data'
                                                                     '/crop_origin_image_and_heatmap01')) if
                                  "heatmap" in x and x.startswith(number + '_' + name)]
    for json_landmark_filename in all_json_landmark_filename:
        heatmap_image = sitk.ReadImage(os.path.join('/home/user5/sharedata/GXE/SkullWidth/data'
                                                    '/crop_origin_image_and_heatmap01', json_landmark_filename))
        heatmap_array1 = sitk.GetArrayFromImage(heatmap_image)
        # 获取三维热图中心点的坐标
        center_points = np.argwhere(heatmap_array1 == 1)
        lst = center_points.tolist()
        points.append(lst[0])

    points = np.asarray(points)
    # print(points.shape)
    # print(points)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
    print(kmeans.labels_)

    # 找到属于特定聚类的数据点
    cluster_0_indices = np.where(kmeans.labels_ == 0)[0]
    cluster_1_indices = np.where(kmeans.labels_ == 1)[0]
    cluster_2_indices = np.where(kmeans.labels_ == 2)[0]

    # 输出聚类结果
    print("Cluster 0 data points:")
    print(points[cluster_0_indices])

    print("Cluster 1 data points:")
    print(points[cluster_1_indices])

    print("Cluster 2 data points:")
    print(points[cluster_2_indices])

    # 计算几何中点
    center0 = np.mean(points[cluster_0_indices], axis=0)
    center1 = np.mean(points[cluster_1_indices], axis=0)
    center2 = np.mean(points[cluster_2_indices], axis=0)
    center0 = np.round(center0).astype(int)
    center1 = np.round(center1).astype(int)
    center2 = np.round(center2).astype(int)

    # 通过几何中点外扩来截图
    min_origin_tooth_x0 = max(0, center2[2] - 20)
    max_origin_tooth_x0 = min(origin_array_z_shape, center2[2] + 20)
    min_origin_tooth_z0 = max(0, center2[0] - 72)
    max_origin_tooth_z0 = min(origin_array_x_shape, center2[0] + 72)
    min_origin_tooth_y0 = max(0, center2[1] - 36)
    max_origin_tooth_y0 = min(origin_array_y_shape, center2[1] + 36)
    print('牙齿区域裁剪范围：', [min_origin_tooth_x0, max_origin_tooth_x0], [min_origin_tooth_y0, max_origin_tooth_y0],
          [min_origin_tooth_z0, max_origin_tooth_z0])

    # 截原图
    origin_crop_image0 = origin_image_array[min_origin_tooth_z0:max_origin_tooth_z0,
                         min_origin_tooth_y0:max_origin_tooth_y0, min_origin_tooth_x0:max_origin_tooth_x0]
    if origin_crop_image0.shape[0] != 144:
        origin_crop_image0 = np.pad(origin_crop_image0, ((0, 144 - origin_crop_image0.shape[0]), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0)
    if origin_crop_image0.shape[1] != 72:
        origin_crop_image0 = np.pad(origin_crop_image0, ((0, 0), (0, 72 - origin_crop_image0.shape[1]), (0, 0)),
                                    mode='constant', constant_values=0)
    if origin_crop_image0.shape[2] != 40:
        origin_crop_image0 = np.pad(origin_crop_image0, ( (0, 0), (0, 0), (0, 40 - origin_crop_image0.shape[2])),
                                    mode='constant', constant_values=0)
    sitk_origin_crop_image0 = sitk.GetImageFromArray(origin_crop_image0)
    sitk.WriteImage(sitk_origin_crop_image0,
                    os.path.join(save_data_path, number + "_" + name + "_image.nii.gz"))

    # 截热图
    heatmap_array = np.zeros_like(origin_image_array)
    for json_landmark_filename in all_json_landmark_filename:
        # 热图重叠生成
        heatmap_image = sitk.ReadImage(os.path.join('/home/user5/sharedata/GXE/SkullWidth/data'
                                                    '/crop_origin_image_and_heatmap01', json_landmark_filename))
        heatmap_array1 = sitk.GetArrayFromImage(heatmap_image)
        heatmap_array = np.maximum(heatmap_array1, heatmap_array)
    heatmap_array0 = heatmap_array[min_origin_tooth_z0:max_origin_tooth_z0,
                     min_origin_tooth_y0:max_origin_tooth_y0, min_origin_tooth_x0:max_origin_tooth_x0]
    if heatmap_array0.shape[0] != 144:
        heatmap_array0 = np.pad(heatmap_array0, ((0, 144 - heatmap_array0.shape[0]), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0)
    if heatmap_array0.shape[1] != 72:
        heatmap_array0 = np.pad(heatmap_array0, ((0, 0), (0, 72 - heatmap_array0.shape[1]), (0, 0)),
                                    mode='constant', constant_values=0)
    if heatmap_array0.shape[2] != 40:
        heatmap_array0 = np.pad(heatmap_array0, ((0, 0), (0, 0), (0, 40 - heatmap_array0.shape[2])),
                                    mode='constant', constant_values=0)
    sitk_origin_heatmap_array0 = sitk.GetImageFromArray(heatmap_array0)
    sitk.WriteImage(sitk_origin_heatmap_array0,
                    os.path.join(save_label_path, number + "_" + name + "_label.nii.gz"))
