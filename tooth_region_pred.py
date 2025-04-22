import torch
import os
from torch.utils.data import DataLoader
from preprocessing import normalization

# 我们自定义的一些模型和数据相关操作
from UNet_xhy import UNet3D_simple
from SkullWidthCBCT import SkullWidthCBCTDataset

# 屏蔽相关的警告，如果不确定，那就屏蔽下面的代码
import warnings

warnings.filterwarnings('ignore')

import SimpleITK as sitk

sitk.ProcessObject.SetGlobalWarningDisplay(False)
# GPU显卡设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
model_path = '/home/user16/sharedata/GXE/SkullWidth/model/tooth_best_model/bestmodel.pth'
net = UNet3D_simple(n_class=1).to(device)
net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0")))
# 开始预测
net.eval()
root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
save_path = '/home/user16/sharedata/GXE/SkullWidth/data/tooth_region'
target_size = (72, 72, 72)
image_xsize = 0
image_ysize = 0
image_zsize = 0

# 训练集
train_dataset = SkullWidthCBCTDataset(data_root_dir=root_path,
                                      txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/train.txt',
                                      zoom_target_size=target_size)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
with torch.no_grad():
    net.eval()
    for iteration, (image, heatmap, patient_info) in enumerate(train_dataloader):
        number, name = patient_info[0].split("_")
        image = image.float().to(device)
        image = normalization(image, torch.quantile(image, 0.01), torch.quantile(image, 0.99))
        pred = net(image)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        image_xsize += pred.shape[0]
        image_ysize += pred.shape[0]
        image_zsize += pred.shape[0]
        np_heatmap = pred.cpu().numpy()[0, 0, ...]
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_heatmap, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))
        print(number, "save")
print("train all save")

# 测试集
test_dataset = SkullWidthCBCTDataset(data_root_dir=root_path,
                                     txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/test.txt',
                                     zoom_target_size=target_size)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
with torch.no_grad():
    net.eval()
    for iteration, (image, heatmap, patient_info) in enumerate(test_dataloader):
        number, name = patient_info[0].split("_")
        image = image.float().to(device)
        image = normalization(image, torch.quantile(image, 0.01), torch.quantile(image, 0.99))
        pred = net(image)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        image_xsize += pred.shape[0]
        image_ysize += pred.shape[0]
        image_zsize += pred.shape[0]
        np_heatmap = pred.cpu().numpy()[0, 0, ...]
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_heatmap, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))
        print(number, "save")
print("test all save")

# 验证集
valid_dataset = SkullWidthCBCTDataset(data_root_dir=root_path,
                                      txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/valid.txt',
                                      zoom_target_size=target_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
with torch.no_grad():
    net.eval()
    for iteration, (image, heatmap, patient_info) in enumerate(valid_dataloader):
        number, name = patient_info[0].split("_")
        image = image.float().to(device)
        image = normalization(image, torch.quantile(image, 0.01), torch.quantile(image, 0.99))
        pred = net(image)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        image_xsize += pred.shape[0]
        image_ysize += pred.shape[0]
        image_zsize += pred.shape[0]
        np_heatmap = pred.cpu().numpy()[0, 0, ...]
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_heatmap, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))
        print(number, "save")
print("valid all save")
