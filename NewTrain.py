import os
import shutil
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from Networks.Mainnet import SimpleRegNet
from Dataset.DataSet import NifitDataSet
from T4T.Utility.CallBacks import EarlyStopping

import datetime
import time


def time_printer():
    now = datetime.datetime.now()
    ts = now.strftime('%Y-%M-%D %H:%M:%S')
    print('do func time : ', ts)


def ShowLossCurve(loss_dict, save_folder, title):
    color_list = sns.color_palette(
        ['#e50000', '#fffd01', '#87fd05', '#00ffff', '#152eff', '#ff08e8', '#ff5b00', '#9900fa']) \
                 + sns.color_palette('deep')
    plt.title(title)
    plt.xlabel("epoch") #x轴标签
    plt.ylabel("loss value") #y轴标签
    for index, key in enumerate(loss_dict.keys()):
        y = loss_dict[key]
        x = np.arange(1, len(y)+1)
        plt.plot(x, y, c=color_list[index], label=key)

    plt.legend(loc='best')
    # plt.show()
    plt.savefig(save_folder, bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.close()


def train():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    ##########
    # Prepare
    ##########
    # Get train data
    device = 'cuda:1'
    batch_size = 2
    is_min = True
    is_box = False
    flow_compute = 'single'
    image_loss = ['corr', 'mse']
    image_loss_dict = dict(zip(image_loss, [0.25, 0.25]))
    flow_loss = ['epe', 'smooth']
    flow_loss_dict = dict(zip(flow_loss, [0.75, 0.75]))
    model_folder = '/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1129'
    random_3d_augment = {
        # 'zoom': [1, 1.25],  # 缩放？
        # 'horizontal_flip': True,  # 翻转
        'volume_percent': [0.05, 0.5],
        'rotate_angle': [0, 5],
        'rotate_axis': [1, 0, 1]
    }
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    else:
        shutil.rmtree(model_folder)
        os.makedirs(model_folder)

    train_dataset = NifitDataSet(data_path=r'/data/data1/zyh/Data/CTLung/CropData',
                                 csv_path=r'/data/data1/zyh/Data/CTLung/CropData/train.csv',
                                 transforms=random_3d_augment, shuffle=True, is_min=is_min,
                                 fixed_key='rigid',
                                 moving_key='inhale')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    train_batches = np.ceil(len(train_dataset) / batch_size)

    # Get val data
    val_dataset = NifitDataSet(data_path=r'/data/data1/zyh/Data/CTLung/CropData',
                               csv_path=r'/data/data1/zyh/Data/CTLung/CropData/val.csv',
                               transforms=None, shuffle=True, is_min=is_min,
                               fixed_key='rigid',
                               moving_key='inhale'
                               )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_batches = np.ceil(len(val_dataset) / batch_size)

    # ModelSetting
    simple_regnet = SimpleRegNet(shape=[256, 256, 256],
                                 n_deform=1,
                                 n_recursive=1,
                                 in_channels=2,
                                 affine_loss_dict={},
                                 image_loss_dict=image_loss_dict,
                                 flow_loss_dict=flow_loss_dict,
                                 deform_net='UNetComplex',
                                 is_affine=False,
                                 is_min=is_min,
                                 is_box=is_box).to(device)

    # simple_regnet.load_state_dict(torch.load(r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1031_batch2/81-0.054856.pt'))

    num_params = 0
    for param in simple_regnet.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    # Optimizer
    optimizer = torch.optim.Adam(simple_regnet.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5,
                                                           verbose=True)

    early_stopping = EarlyStopping(store_path=str(os.path.join(model_folder, '{}-{:.6f}.pt')), patience=100, verbose=True)

    ##########
    # Train
    ##########
    deform_loss = image_loss + flow_loss
    train_loss_record = dict(zip(deform_loss, [[] for idx in range(len(deform_loss))]))
    val_loss_record = dict(zip(deform_loss, [[] for idx in range(len(deform_loss))]))
    for epoch in range(0, 2000):
        print('Epoch: {}'.format(epoch + 1))
        train_losses_dict, val_losses_dict = {}, {}
        simple_regnet.train()
        print('Train:', end=' ')
        for i, (moving, fixed, field, volume, moving_mask, _, _, _) in enumerate(train_loader):
            moving = moving.to(device)  # (batch, 1, *)
            fixed = fixed.to(device)
            field = field.to(device)
            volume = volume.to(device)
            moving_mask = moving_mask.to(device)

            img_p, _, loss_dict = simple_regnet(moving, fixed, volume, moving_mask, field, _, flow_compute=flow_compute)

            # Optimize
            optimizer.zero_grad()
            loss_dict['all'].backward()
            optimizer.step()
            train_losses_dict = dict(Counter(loss_dict) + Counter(train_losses_dict))
            print('->{}'.format(i), end='')
        print('\nVal:', end=' ')
        simple_regnet.eval()
        with torch.no_grad():
            for i, (moving, fixed, field, volume, moving_mask, _, _, _) in enumerate(val_loader):
                moving = moving.to(torch.float32).to(device)  # (batch, 1, *)
                fixed = fixed.to(torch.float32).to(device)
                field = field.to(torch.float32).to(device)
                volume = volume.to(torch.float32).to(device)
                moving_mask = moving_mask.to(device)

                img_p, _, loss_dict = simple_regnet(moving, fixed, volume, moving_mask, field, _, flow_compute=flow_compute)
                val_losses_dict = dict(Counter(loss_dict) + Counter(val_losses_dict))
                print('->{}'.format(i), end='')
        # Recorder
        print('\ntrain:', end=' ')
        for key in train_losses_dict:
            if key == 'all': continue
            print('{}: {:3f}'.format(key, train_losses_dict[key] / train_batches), end=' ')
        print('total train loss: {}'.format(train_losses_dict['all'] / train_batches))
        print('val:', end=' ')
        for key in val_losses_dict:
            if key == 'all': continue
            print('{}: {:3f}'.format(key, val_losses_dict[key] / val_batches), end=' ')
        print('total val loss: {}'.format(val_losses_dict['all'] / val_batches))

        for key in train_loss_record.keys():
            train_loss_record[key].append(train_losses_dict[key].cpu().detach().numpy().squeeze())
            val_loss_record[key].append(val_losses_dict[key].cpu().detach().numpy().squeeze())

        scheduler.step(val_losses_dict['all'] / val_batches)
        early_stopping(val_losses_dict['all'] / val_batches,
                       simple_regnet,
                       (epoch + 1, val_losses_dict['all'] / val_batches))

        if early_stopping.early_stop:
            print("Early stopping")
            break
        # break
    ShowLossCurve(train_loss_record, os.path.join(model_folder, 'train.jpg'), 'Train Loss')
    ShowLossCurve(val_loss_record, os.path.join(model_folder, 'val.jpg'), 'Validation Loss')


def test():
    ##########
    # Prepare
    ##########
    # Get train data
    device = 'cuda:7'
    batch_size = 1
    model_folder = '/data/data1/zyh/ModelSetting/RespiratoryCompensation/checkpoints/RespiratoryCompensation_0823_VTN'

    test_dataset = NifitDataSet(data_path=r'/data/data1/zyh/Data/CTLung/Test/Train', transforms=None, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # ModelSetting
    simple_regnet = SimpleRegNet(shape=[256, 256, 256],
                                 n_deform=3,
                                 n_recursive=3,
                                 in_channels=2,
                                 deform_loss_dict={},
                                 deform_net='UNetComplex').to(device)


    simple_regnet.eval()
    with torch.no_grad():
        for i, (moving, fixed, field, volume) in enumerate(test_loader):
            moving = moving.to(torch.float32).to(device)  # (batch, 1, *)
            fixed = fixed.to(torch.float32).to(device)
            field = field.to(torch.float32).to(device)
            volume = volume.to(torch.float32).to(device)

            image_pred, field_pred, _ = simple_regnet(moving, fixed, volume, field)


if __name__ == '__main__':
    while True:
        time.sleep(0)
        time_printer()
        train()
        break

