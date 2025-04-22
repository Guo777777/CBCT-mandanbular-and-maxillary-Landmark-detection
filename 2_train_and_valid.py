# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# 数据扩增对应的操作
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform

# 我们自定义的一些模型和数据相关操作
from UNet import UNet3d
from heatmap import focal_loss
from SkullWidthCBCT import DatasetTransforms, SkullWidthCBCTDataset

# 屏蔽相关的警告，如果不确定，那就屏蔽下面的代码
import warnings

warnings.filterwarnings('ignore')

import SimpleITK as sitk

sitk.ProcessObject.SetGlobalWarningDisplay(False)

# GPU显卡设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# 优化器
lr = 1e-3
model = UNet3d(n_class=1, act='relu').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)

# 数据加载
target_size = (144, 72, 40)
data_root_path = '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData'
train_dataset = SkullWidthCBCTDataset(data_root_dir=data_root_path,
                                      txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/train.txt',
                                      zoom_target_size=target_size)
valid_dataset = SkullWidthCBCTDataset(data_root_dir=data_root_path,
                                      txt_file_path='/home/user16/sharedata/GXE/SkullWidth/data/valid.txt',
                                      zoom_target_size=target_size)

# 构建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

# # 数据扩增的配置
transforms = []
# 对比度调节
brightness_transform = ContrastAugmentationTransform((0.3, 3.), data_key='data', preserve_range=True)
transforms.append(brightness_transform)
# 大小缩放

spatial_transform = SpatialTransform(target_size, (target_size[0] // 2, target_size[1] // 2, target_size[2] // 2),
                                     do_elastic_deform=False,
                                     do_rotation=True,
                                     angle_x=(0, 0.1 * np.pi), angle_y=(0, 0.1 * np.pi),
                                     do_scale=True, scale=(0.9, 1.1),
                                     border_mode_data='constant', border_cval_data=0, order_data=1,
                                     random_crop=False)
# transforms.append(spatial_transform)
# 镜像翻转
mirror_transform = MirrorTransform(axes=(1, 2))
transforms.append(mirror_transform)
# 数据的数据扩增器
train_transforms = DatasetTransforms(transforms)

# 模型训练的损失函数
loss_fn = focal_loss

# 训练参数
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
best_loss = 100000
start_epoch = 0
train_max_epoch = 5000
num_epoch_no_improvement = 0
epoch_patience = 15

model_save_folder_path = '/home/user16/sharedata/GXE/SkullWidth/model/UNet3d_stage1'
if not os.path.exists(model_save_folder_path):
    os.makedirs(model_save_folder_path)

# 如果中途断了，可以开启下面的代码则继续训练，不用重头开始
# if os.path.exists(os.path.join(model_save_folder_path, 'Unet_model03.pt')):
#     checkpoint_state = torch.load(os.path.join(model_save_folder_path, 'Unet_model03.pt'))
#     start_epoch = checkpoint_state['epoch']
#
#     # 加载之前的模型权重
#     model_dict = model.state_dict()
#     # 将pretrained_dict里不属于model_dict的键剔除掉
#     pretrained_dict = checkpoint_state['state_dict']
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 更新现有的model_dict
#     model_dict.update(pretrained_dict)
#     # 加载我们真正需要的state_dict
#     model.load_state_dict(model_dict)
#     optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
# 如果中途断了，可以开启下面的代码则继续训练，不用重头开始
# if os.path.exists(os.path.join(model_save_folder_path, 'Unet_model06.pt')):
#     checkpoint_state = torch.load(os.path.join(model_save_folder_path, 'Unet_model07.pt'))
#     start_epoch = checkpoint_state['epoch']
#     model.load_state_dict(checkpoint_state['state_dict'])
#     optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])

# 开始训练
for epoch in range(start_epoch, train_max_epoch):
    scheduler.step(epoch)

    # 模型的训练
    model.train()
    for (image, heatmap, i) in tqdm(train_dataloader):
        # 数据扩增: 转为numpy，并封装成batchgenerators的格式
        numpy_image, numpy_heatmap = image.numpy(), heatmap.numpy()
        batch_input = {'data': numpy_image, 'seg': numpy_heatmap}

        # 数据扩增
        batch_augment_input = train_transforms(batch_input)

        # 转为torch格式进行模型的训练
        image, heatmap = torch.from_numpy(batch_augment_input['data']), torch.from_numpy(batch_augment_input['seg'])
        image, heatmap = image.float().to(device), heatmap.float().to(device)
        # image, heatmap = torch.from_numpy(batch_input['data']), torch.from_numpy(batch_input['seg'])
        image, heatmap = image.float().to(device), heatmap.float().to(device)

        # 模型输出
        outputs = model(image)
        # 损失反馈
        loss = loss_fn(outputs, heatmap)
        # 梯度传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))

    # 模型的验证
    with torch.no_grad():
        model.eval()
        print("Validating....")
        for iteration, (image, heatmap, i) in enumerate(valid_dataloader):
            image, heatmap = image.float().to(device), heatmap.float().to(device)
            outputs = model(image)
            loss = loss_fn(outputs, heatmap)
            valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss, train_loss))
    train_losses = []
    valid_losses = []

    # 早停机制，防止过拟合
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_save_folder_path, "Unet_model08.pt"))
        print("Saving model ", os.path.join(model_save_folder_path, "Unet_model08.pt"), '\n')
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {} \n".format(best_loss,
                                                                                                     num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == epoch_patience:
        print("Early Stopping \n")
        break
