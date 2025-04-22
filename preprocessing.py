# 预处理
import numpy as np
import cv2
import SimpleITK as sitk


# 二值化预处理，t是阈值
def thresholding(data, t):
    data_0 = np.zeros_like(data)
    data_0[data < t] = 0
    data_0[data >= t] = 1
    return data_0


# 归一化预处理，a-b范围
def normalization(data, a, b):
    smooth = 1e-10
    data[data < a] = a
    data[data > b] = b
    data_max = data.max()
    data_min = data.min()
    deta = data_max - data_min
    data = (data - data_min) / (deta + smooth)
    return data


# 方差均值标准化预处理，a-b范围
def standardization(data, a, b):
    smooth = 1e-10
    data_0 = data
    data_0[data_0 < a] = a
    data_0[data_0 > b] = b
    data_0 = (data_0 - np.mean(data_0)) / (np.std(data_0) + smooth)
    return data_0


# 图像膨胀
def dilation(data):
    kernel = np.ones((5, 5), np.uint8)
    data_0 = cv2.dilate(data, kernel)
    return data_0


# 尺寸中心裁剪
def cutsize(data, newsize):
    image_x, image_y, image_z = data.shape
    begin = ((image_x - newsize) // 2)
    end = image_x - ((image_x - newsize) - begin)
    newdata = data[begin:end, begin:end, begin:end]
    return newdata


# 图片四周填充
def padsize(data_1, newsize_1):
    image_x_1, image_y_1, image_z_1 = data_1.shape
    begin_1 = ((newsize_1 - image_x_1) // 2)
    end_1 = newsize_1 - ((newsize_1 - image_x_1) - begin_1)
    newdata_1 = np.pad(data_1, pad_width=((begin_1, newsize_1 - end_1),  # 向上填充a个维度，向下填充b个维度
                                          (begin_1, newsize_1 - end_1),  # 向左填充a个维度，向右填充b个维度
                                          (begin_1, newsize_1 - end_1)),
                       mode="constant",  # 填充模式
                       constant_values=(0, 0)  # 第一个维度（就是向上和向左）填充6，第二个维度（向下和向右）填充5
                       )
    return newdata_1


# 重采样
# sitk.sitkNearestNeighbor邻近插值，用于mask；sitk.sitkLinear线性插值用于ct
def resize_image_itk(itkimage, newSize, resamplemethod):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int64)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


# 将三维图像填充为1：1：1
def pad(data):
    image_x, image_y, image_z = data.shape
    image_max = max(image_x, image_y, image_z)
    x_new = (image_max - image_x)
    y_new = (image_max - image_y)
    z_new = (image_max - image_z)
    data1 = np.pad(data, ((x_new, 0), (y_new, 0), (z_new, 0)), mode='constant', constant_values=(0, 0))
    return data1


if __name__ == "__main__":
    a = np.ones((667, 667))
    b = cutsize(a, 656)
    c = padsize(b, 667)
