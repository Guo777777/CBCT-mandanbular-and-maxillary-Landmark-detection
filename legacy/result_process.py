from skimage import measure
import numpy as np
import SimpleITK as sitk
import os

landmark_path = '/home/user16/sharedata/GXE/SkullWidth/data/test_save'
origin_root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'

for patient_name in sorted(os.listdir(os.path.join(landmark_path))):
    if "label" in patient_name:
        continue
    number, name, pred = patient_name.split("_")
    print(number, "begin")
    nrrd_image_file_path = os.path.join(landmark_path, patient_name)
    sitk_image = sitk.ReadImage(nrrd_image_file_path)
    array_image = sitk.GetArrayFromImage(sitk_image)
    # 卡阈值
    array_image[array_image < 0.2] = 0
    array_image[array_image >= 0.2] = 1
    # 标记连通域
    labels = measure.label(array_image)
    print(np.sum(labels))
    # 获取连通区域的属性
    props = measure.regionprops(labels)


    # 输出每个连通区域的中心点坐标
    for region in props:
        centroid = region.centroid
        print("Label:", region.label)
        print("Centroid:", centroid)
