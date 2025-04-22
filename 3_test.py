import torch
import os
from torch.utils.data import DataLoader
import time

# 我们自定义的一些模型和数据相关操作
from UNet import UNet3d
from SkullWidthCBCT import SkullWidthCBCTDataset

# 屏蔽相关的警告，如果不确定，那就屏蔽下面的代码
import warnings

warnings.filterwarnings('ignore')

import SimpleITK as sitk

# 记录开始时间
start_time = time.time()

sitk.ProcessObject.SetGlobalWarningDisplay(False)
# GPU显卡设备选择
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
model_path = '/home/user16/sharedata/GXE/SkullWidth/model/UNet3d_stage1/Unet_model08.pt'
net = UNet3d(n_class=1, act='relu').to(device)
state_dict = torch.load(model_path, map_location=device)['state_dict']
net.load_state_dict(state_dict, strict=True)
# 测试
net.eval()
root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
save_path = '/home/user16/sharedata/GXE/SkullWidth/data/test_save'
target_size = (144, 72, 40)
test_dataset = SkullWidthCBCTDataset(data_root_dir=root_path,
                                     txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/test.txt',
                                     zoom_target_size=target_size)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# 模型的测试
with torch.no_grad():
    net.eval()
    for iteration, (image, heatmap, patient_info) in enumerate(test_dataloader):
        number, name = patient_info[0].split("_")
        image = image.float().to(device)
        pred = net(image)
        np_heatmap = pred.cpu().numpy()[0, 0, ...]
        # 保存为SimpleITK的nii.gz格式，用于在ITK-SNAP中打开查看
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_heatmap, os.path.join(save_path, number + "_" + name + "_pred.nii.gz"))

        np_heatmap = heatmap.cpu().numpy()[0, 0, ...]
        # 保存为SimpleITK的nii.gz格式，用于在ITK-SNAP中打开查看
        sitk_heatmap = sitk.GetImageFromArray(np_heatmap)
        sitk.WriteImage(sitk_heatmap, os.path.join(save_path, number + "_" + name + "_label.nii.gz"))

        print(number, "save")
print("all save")

# 记录结束时间
end_time = time.time()

# 计算并打印程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time} 秒")