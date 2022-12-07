import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm
from time import sleep

import multiprocessing as Pool
from tqdm import tqdm


def GetVolume():
    volume_dict = {}
    pbar = tqdm(total=len(os.listdir(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/image')), ncols=80)
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/image')):
        sleep(0.1)
        inhale_path = os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale', 'mask', case)
        inhale_mask = sitk.ReadImage(inhale_path)
        exhale_path = os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale', 'mask', case)
        exhale_mask = sitk.ReadImage(exhale_path)
        inhale_volume = np.count_nonzero(sitk.GetArrayFromImage(inhale_mask))
        exhale_volume = np.count_nonzero(sitk.GetArrayFromImage(exhale_mask))
        volume = (inhale_volume - exhale_volume) / inhale_volume
        volume_dict[case.split('.nii.gz')[0]] = [inhale_volume, exhale_volume, volume]
        pbar.update()
    pbar.close()
    df = pd.DataFrame(volume_dict, index=['inhale_volume', 'exhale_volume', 'volume_percent']).T
    df.to_csv(r'/data/data1/zyh/Data/volume.csv')
# GetVolume()


def CheckExchange():
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/NewHydra/inhale/mask')):
        inhale = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/NewHydra/inhale/mask', case)))
        exhale = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/NewHydra/exhale/mask', case)))
        if np.sum(inhale) <= np.sum(exhale):
            print(case.replace(' ', '_'))
# CheckExchange()

def ExchangeFile(source, target):
    temp = os.path.join(os.path.dirname(target), 'temp.nii.gz')
    shutil.move(source, temp)
    shutil.move(target, source)
    shutil.move(temp, target)


# case_list = ['20191112_chen_hong_qing.nii.gz','20191115_zhu_jin_ping.nii.gz','20191118_tian_zuo_hai.nii.gz',
#              '20191119_li_yong_ran.nii.gz','20200610_xu_ju_fang.nii.gz','20200624_xu_quan_hai.nii.gz',
#              '20200627_xu_zhong_qing.nii.gz','20200724_tang_yi_zhao.nii.gz','20200727_wang_shao_yun.nii.gz',
#              '20210319_shi_ying.nii.gz','20210426_qu_yu_cai.nii.gz','20210520_sun_xiu_ying.nii.gz',
#              '20210531_li_shu_dong.nii.gz','20210629_guan_xin_guo.nii.gz','20211213_xia_jin_gen.nii.gz',
#              '20220114_fan_fu_tian.nii.gz','20220207_li_yi_lan.nii.gz','20220613_yang_mei_ling.nii.gz',
#              '20220727_gu_zhao_yang.nii.gz','Case_0.nii.gz','Case_10.nii.gz','Case_24.nii.gz','Case_33.nii.gz',
#              'Case_45.nii.gz','Case_48.nii.gz','Case_53.nii.gz','Case_6.nii.g','Case_113.nii.gz','Case_75.nii.gz',
#              'Case_85.nii.gz','Case_91.nii.gz','20220824_jiang_hai_tao.nii.gz','20220825_xi_yu_mei.nii.gz',
#              '20220829_sun_ke_xiang.nii.gz','20220907_fan_xi_yuan.nii.gz','20220613_yang_mei_ling.nii.gz']

# for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule')):
#     if case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/deform_flow'):
#         moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/moving_mask', case)))
#         fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/fixed_mask', case)))
#         if np.sum(moving_mask) <= np.sum(fixed_mask):
#             print(case)
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/fixed_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/moving_nodules', case))
#         else:
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/moving_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/fixed_nodules', case))
#     elif case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/deform_flow'):
#         moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/moving_mask', case)))
#         fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/fixed_mask', case)))
#         if np.sum(moving_mask) <= np.sum(fixed_mask):
#             print(case)
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/fixed_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/moving_nodules', case))
#         else:
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/moving_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/fixed_nodules', case))
#     elif case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/deform_flow'):
#         moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/moving_mask', case)))
#         fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed_mask', case)))
#         if np.sum(moving_mask) <= np.sum(fixed_mask):
#             print(case)
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/moving_nodules', case))
#         else:
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/inhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/moving_nodules', case))
#             shutil.copy(os.path.join(r'/data/data1/zyh/Data/CTLung/CropDataAll1.35Fliped/exhale/nodule', case),
#                         os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed_nodules', case))

# for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed'):
#     image = sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed', case))
#     mask = sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed_mask', case))
#     if image.GetOrigin() == mask.GetOrigin(): continue
#     else: print(case)

# image = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/fixed/20191115_tan_fen.nii.gz')
# ref_image = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/moving/20191115_tan_fen.nii.gz')
# image.CopyInformation(ref_image)
# sitk.WriteImage(image, r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/test.nii.gz')


# from scipy.ndimage import binary_closing, binary_erosion, median_filter
# data_root = r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/OnlyDeformNodule'
# pbar = tqdm(total=len(os.listdir(data_root)), ncols=80)
# for case in sorted(os.listdir(data_root)):
#     data_path = os.path.join(data_root, case, 'result.nii.gz')
#     if not os.path.exists(data_path): continue
#     image = sitk.ReadImage(data_path)
#     data = sitk.GetArrayFromImage(image)
#     closing_data = binary_closing(data, structure=np.ones((3, 3, 3))).astype(np.int32)
#     closing_data = binary_erosion(closing_data, structure=np.ones((1, 1, 1))).astype(np.int32)
#     closing_data = median_filter(closing_data, size=3).astype(np.int32)
#     closing_image = sitk.GetImageFromArray(closing_data)
#     closing_image.CopyInformation(image)
#     sitk.WriteImage(closing_image, os.path.join(data_root, case, 'result_closing.nii.gz'))
#     pbar.update()
# pbar.close()


def SplitVolume():
    volume_df = pd.read_csv(r'/data/data1/zyh/Data/volume.csv', index_col=0)
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/deform_flow')]
    val_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/deform_flow')]
    test_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/deform_flow')]

    for index in volume_df.index:
        new_index = index.replace(' ', '_')
        if new_index in train_list:
            train_df = train_df.append(volume_df.loc[index])
        elif new_index in val_list:
            val_df = val_df.append(volume_df.loc[index])
        elif new_index in test_list:
            test_df = test_df.append(volume_df.loc[index])
        else:
            print(index)
    train_df.to_csv(r'/data/data1/zyh/Data/train.csv')
    val_df.to_csv(r'/data/data1/zyh/Data/val.csv')
    test_df.to_csv(r'/data/data1/zyh/Data/test.csv')
# SplitVolume()

def Stats():
    import scipy.stats
    volume_df = pd.read_csv(r'/data/data1/zyh/Data/volume.csv', index_col=0)

    volume = volume_df.loc[:, 'volume_percent'].values
    hosiptal = []
    hydra = []
    for index in volume_df.index:
        if 'Case' in index:
            hydra.append(volume_df.loc[index, 'volume_percent'])
        else:
            hosiptal.append(volume_df.loc[index, 'volume_percent'])
    print(scipy.stats.normaltest(volume))
    print(scipy.stats.normaltest(hosiptal))
    print(scipy.stats.normaltest(hydra))
    print(scipy.stats.mannwhitneyu(hosiptal, hydra, alternative='greater'))

    train, val, test = [], [], []
    train_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Train/deform_flow')]
    val_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Val/deform_flow')]
    test_list = [case.split('.nii.gz')[0] for case in os.listdir(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix0927/Test/deform_flow')]
    for index in volume_df.index:
        new_index = index.replace(' ', '_')
        if new_index in train_list:
            train.append(volume_df.loc[index, 'volume_percent'])
        elif new_index in val_list:
            val.append(volume_df.loc[index, 'volume_percent'])
        elif new_index in test_list:
            test.append(volume_df.loc[index, 'volume_percent'])
        else:
            print(index)
    print(scipy.stats.mannwhitneyu(train, test, alternative='greater'))
    print(scipy.stats.mannwhitneyu(train, val, alternative='greater'))
    print(scipy.stats.mannwhitneyu(val, test, alternative='greater'))
# Stats()


def GetCohortCSV():
    import random
    all_case = os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving')

    train_df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/train.csv', index_col=0).squeeze()
    train_list = sorted(train_df.index.tolist())
    train_list = ['{}.nii.gz'.format(case) for case in train_list]

    test_df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/test.csv', index_col=0).squeeze()
    test_list = sorted(test_df.index.tolist())
    test_list = ['{}.nii.gz'.format(case) for case in test_list]

    val_df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/val.csv', index_col=0).squeeze()
    val_list = sorted(val_df.index.tolist())
    val_list = ['{}.nii.gz'.format(case) for case in val_list]

    new_case = [case for case in all_case if case not in train_list and case not in test_list and case not in val_list]
    random.shuffle(new_case)

    total_case = len(all_case)
    train_num = int(total_case * 0.64)
    val_num = int(total_case * 0.16)
    test_num = total_case - train_num - val_num
    print([case for case in train_list if (case in val_list or case in test_list)])
    print([case for case in val_list if (case in train_list or case in test_list)])
    print([case for case in test_list if (case in val_list or case in train_list)])

    for case in new_case:
        if len(train_list) < train_num:
            train_list.append(case)
        elif len(val_list) < val_num:
            val_list.append(case)
        else:
            test_list.append(case)
    print(len(train_list), len(val_list), len(test_list))
    print([case for case in train_list if (case in val_list or case in test_list)])
    print([case for case in val_list if (case in train_list or case in test_list)])
    print([case for case in test_list if (case in val_list or case in train_list)])

    train_list = [case.split('.nii.gz')[0] for case in train_list]
    df = pd.DataFrame(train_list)
    print(len(df.index))
    df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/train.csv', index=False)

    val_list = [case.split('.nii.gz')[0] for case in val_list]
    df = pd.DataFrame(val_list)
    print(len(df.index))
    df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/val.csv', index=False)

    test_list = [case.split('.nii.gz')[0] for case in test_list]
    df = pd.DataFrame(test_list)
    print(len(df.index))
    df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/test.csv', index=False)
# GetCohortCSV()

# nodule_original = os.listdir(r'/data/data1/zyh/Data/CTLung/RawData/anotatedNoduleImage')
# nodule_original = [case.replace(' ', '_') for case in nodule_original if not case.endswith('.csv')]
# nodule_now = os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving_nodule')
# nodule_now = [case.split('.nii.gz')[0].replace(' ', '_') for case in nodule_now]
# print([case for case in nodule_original if case not in nodule_now])
# ['20191118_tian_zuo_hai']


# for root, dirs, files in os.walk(r'/data/data1/zyh/Data/CTLung/CNNData'):
#     if len(dirs) == 0 and len(files) > 0:
#         [os.rename(os.path.join(root, file), os.path.join(root, file.replace(' ', '_'))) for file in files]


from scipy import ndimage
from scipy.ndimage import median_filter

def KeepLargestTwo(mask):
    new_mask = np.zeros(mask.shape)
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    print(max_volume)
    index = np.argmax(max_volume)
    new_mask[label_im == index + 1] = 1
    max_volume.remove(max_volume[index])
    second_index = np.argmax(max_volume)
    if second_index >= index: second_index += 1
    new_mask[label_im == second_index + 1] = 1
    return new_mask

# exhale_mask_path = r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew/exhale/mask/20200603_chen_jian_ming.nii.gz'
# exhale_image = sitk.ReadImage(exhale_mask_path)
# exhale_mask = sitk.GetArrayFromImage(exhale_image)
# new_mask = KeepLargestTwo(exhale_mask)
# new_image = sitk.GetImageFromArray(new_mask)
# new_image.CopyInformation(exhale_image)
# sitk.WriteImage(new_image,
#                 r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew/exhale/mask/20200603_chen_jian_ming_new.nii.gz')


# for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving_mask')):
#     moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/moving_mask', case)))
#     moving_mask = median_filter(moving_mask, size=5).astype(np.int32)
#     fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/fixed_mask', case)))
#     fixed_mask = median_filter(fixed_mask, size=5).astype(np.int32)
#     if np.sum(moving_mask) <= np.sum(fixed_mask):
#         print('moving: {}\tfixed: {}'.format(np.sum(moving_mask), np.sum(fixed_mask)))
#         print(case)

# for case in os.listdir(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii'):
#     if not os.path.isdir(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case)):
#         continue
#     case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case)
#     exhale_path = os.path.join(case_folder, 'exhale', 'image.nii.gz')
#     inhale_path = os.path.join(case_folder, 'inhale', 'image.nii.gz')
#     shutil.copyfile(exhale_path, os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', '{}_exhale_0000.nii.gz'.format(case)))
#     shutil.copyfile(inhale_path, os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', '{}_inhale_0000.nii.gz'.format(case)))

# for case in os.listdir(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii'):
#     if not case.endswith('.nii.gz'): continue
#     case_name = '{}_{}'.format(case.split('_')[0], case.split('_')[1])
#     type = case[-13:-7]
#     if not os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case_name, type)):
#         os.makedirs(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case_name, type))
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case),
#                 os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Hydra1021_nii', case_name, type, 'lobe.nii.gz'))
# for case in os.listdir(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew'):
#     if os.path.isdir(os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew', case)): continue
#     case_name = case[0: -14]
#     type = case[-13:-7]
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew', case),
#                 os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew', type, 'mask', '{}.nii.gz'.format(case_name.replace(' ', '_'))))
# for root, dirs, files in os.walk(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/CropNew'):
#     if len(files) > 0:
#         [os.rename(os.path.join(root, file), os.path.join(root, file.replace(' ', '_'))) for file in files]


# print(np.sum(sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CNNData/moving_mask/20191119_li_yong_ran.nii.gz'))))
# print(np.sum(sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CNNData/fixed_mask/20191119_li_yong_ran.nii.gz'))))


def MakeDir(path):
    if not os.path.exists(path): os.makedirs(path)

def Normalization(image, min, max):
    """
    Normalize an image to 0 - 1 (8bits)
    """
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(max)
    resacleFilter.SetOutputMinimum(min)

    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-255

    return image

def CopyData(image_root, save_root, operate, case_list=[]):
    if len(case_list) == 0: case_list = sorted(os.listdir(image_root))
    pbar = tqdm(total=len(case_list), ncols=80)
    # save_root = os.path.join(os.path.dirname(image_root), 'Registration')
    MakeDir(os.path.join(save_root, '{}_result'.format(operate)))
    MakeDir(os.path.join(save_root, '{}_flow'.format(operate)))
    MakeDir(os.path.join(save_root, '{}_mask'.format(operate)))
    MakeDir(os.path.join(save_root, '{}_mask_closing'.format(operate)))
    MakeDir(os.path.join(save_root, '{}_nodule'.format(operate)))
    MakeDir(os.path.join(save_root, '{}_nodule_closing'.format(operate)))
    for case in case_list:
        if not case.endswith('.nii.gz'):
            case_name = case
        else:
            case_name = case.split('.nii.gz')[0]

        mask_root = os.path.join(os.path.dirname(image_root), '{}Mask'.format(os.path.basename(image_root)))
        nodule_root = os.path.join(os.path.dirname(image_root), '{}Nodule'.format(os.path.basename(image_root)))

        shutil.copyfile(os.path.join(image_root, case_name, 'result.nii.gz'),
                        os.path.join(save_root, '{}_result'.format(operate), '{}.nii.gz'.format(case_name)))
        shutil.copyfile(os.path.join(image_root, case_name, 'deformationField.nii.gz'),
                        os.path.join(save_root, '{}_flow'.format(operate), '{}.nii.gz'.format(case_name)))
        shutil.copyfile(os.path.join(mask_root, case_name, 'result.nii.gz'),
                        os.path.join(save_root, '{}_mask'.format(operate), '{}.nii.gz'.format(case_name)))

        image = sitk.ReadImage(os.path.join(image_root, case_name, 'result.nii.gz'))
        image = Normalization(image, min=0, max=255)
        sitk.WriteImage(image, os.path.join(save_root, '{}_result'.format(operate), '{}.nii.gz'.format(case_name)))

        mask = sitk.ReadImage(os.path.join(mask_root, case_name, 'result.nii.gz'))
        mask_arr = sitk.GetArrayFromImage(mask)
        mask_arr[mask_arr > 0] = 1
        new_mask_arr = ndimage.binary_closing(mask_arr, structure=np.ones((3, 3, 3))).astype(int)
        new_mask = sitk.GetImageFromArray(new_mask_arr)
        new_mask.CopyInformation(mask)
        sitk.WriteImage(new_mask, os.path.join(save_root, '{}_mask_closing'.format(operate), '{}.nii.gz'.format(case_name)))

        if os.path.exists(os.path.join(nodule_root, case_name, 'result.nii.gz')):
            shutil.copyfile(os.path.join(nodule_root, case_name, 'result.nii.gz'),
                            os.path.join(save_root, '{}_nodule'.format(operate), '{}.nii.gz'.format(case_name)))
            nodule = sitk.ReadImage(os.path.join(nodule_root, case_name, 'result.nii.gz'))
            nodule_arr = sitk.GetArrayFromImage(nodule)
            nodule_arr[nodule_arr > 0] = 1
            new_nodule_arr = ndimage.binary_closing(nodule_arr, structure=np.ones((3, 3, 3))).astype(int)
            new_nodule = sitk.GetImageFromArray(new_nodule_arr)
            new_nodule.CopyInformation(nodule)
            sitk.WriteImage(new_nodule,
                            os.path.join(save_root, '{}_nodule_closing'.format(operate), '{}.nii.gz'.format(case_name)))
        pbar.update()
    pbar.close()

# CopyData(image_root=r'/data/data1/zyh/Data/CTLung/RegistrationDeform/Rigid',
#          save_root='/data/data1/zyh/Data/CTLung/CNNData',
#          operate='rigid')
# CopyData(image_root=r'/data/data1/zyh/Data/CTLung/RegistrationDeform/RigidDeform',
#          save_root='/data/data1/zyh/Data/CTLung/CNNData',
#          operate='rigid_deform')
#

# restore_file = ['deformationField.nii.gz', 'result.nii.gz', 'TransformParameters.0.txt']
# for root, dirs, files in os.walk(r'/data/data1/zyh/Data/CTLung/RegistrationDeform'):
#     if len(dirs) == 0 and len(files) > 0:
#         [os.remove(os.path.join(root, file)) for file in files if file not in restore_file]
# root_list = [r'/data/data1/zyh/Data/CTLung/RegistrationDeform/OnlyDeformMask',
#              r'/data/data1/zyh/Data/CTLung/RegistrationDeform/OnlyDeformNodule',
#              r'/data/data1/zyh/Data/CTLung/RegistrationDeform/RigidMask',
#              r'/data/data1/zyh/Data/CTLung/RegistrationDeform/RigidNodule',
#              r'/data/data1/zyh/Data/CTLung/RegistrationDeform/RigidDeformMask',
#              r'/data/data1/zyh/Data/CTLung/RegistrationDeform/RigidDeformNodule']
# for root in root_list:
#     for case in os.listdir(root):
#         os.remove(os.path.join(root, case, 'deformationField.nii.gz'))



def Norm(data):
    normal_data = np.asarray(data, dtype=np.float32)
    if normal_data.max() - normal_data.min() < 1e-6:
        return np.zeros_like(normal_data)
    normal_data = np.clip(normal_data, a_min=0, a_max=255)
    normal_data = normal_data - np.min(normal_data)
    normal_data = normal_data / np.max(normal_data)
    normal_data = normal_data * 255
    return normal_data

# root = r'/data/data1/zyh/Data/CTLung/RegistrationElastix/OnlyDeform'
# pbar = tqdm(total=len(os.listdir(root)), ncols=80)
# for case in sorted(os.listdir(root)):
#     image = sitk.ReadImage(os.path.join(root, case, 'result.nii.gz'))
#     image_arr = sitk.GetArrayFromImage(image)
#     # print(np.min(image_arr), np.max(image_arr))
#     image_arr = Norm(image_arr)
#     new_image = sitk.GetImageFromArray(image_arr)
#     new_image.CopyInformation(image)
#     sitk.WriteImage(new_image, os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/rigid_deform', '{}.nii.gz'.format(case)))
#
#     image = sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/RegistrationElastix/Rigid', case, 'result.nii.gz'))
#     image_arr = sitk.GetArrayFromImage(image)
#     image_arr = Norm(image_arr)
#     new_image = sitk.GetImageFromArray(image_arr)
#     new_image.CopyInformation(image)
#     sitk.WriteImage(new_image,
#                     os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/rigid', '{}.nii.gz'.format(case)))
#     pbar.update()
# pbar.close()

# checkpoint = r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1029/39-0.054048.pt'
# weights = torch.load(checkpoint)
# print()

# case_list = os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/inhale/image')
# restore_case = [case.replace(' ', '_') for case in case_list]
# restore_case = list(set(restore_case))
# remove_case = [case for case in case_list if case not in restore_case]
# print(remove_case)
def CheckResampelData():
    from MeDIT.SaveAndLoad import LoadImage
    import matplotlib.pyplot as plt
    c_list = ['inhale', 'exhale']
    for c in c_list:
        image_root = os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', c, 'figure')
        if not os.path.exists(image_root): os.makedirs(image_root)
        pbar = tqdm(total=len(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/image'.format(c))), ncols=80)
        for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/image'.format(c))):
            _, image, _ = LoadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/image'.format(c), case))
            _, mask, _ = LoadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/mask'.format(c), case))
            if os.path.exists((os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/nodule'.format(c), case))):
                _, nodule, _ = LoadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/nodule'.format(c), case))

            max_slice = np.argmax(np.sum(mask, axis=(1, 2)))
            plt.title(c)
            plt.imshow(image[max_slice], cmap='gray')
            plt.contour(mask[max_slice], colors='r', linewidths=0.25)
            if os.path.exists((os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/{}/nodule'.format(c), case))):
                plt.contour(nodule[max_slice], colors='g', linewidths=0.25)
            plt.axis('off')
            plt.savefig(os.path.join(image_root, '{}.jpg'.format(case.split('.nii.gz')[0])),
                        bbox_inches='tight', dpi=100)
            plt.close()
            pbar.update()
        pbar.close()


def LogExchangeFile(source):
    from MeDIT.SaveAndLoad import LoadImage
    import pandas as pd
    pbar = tqdm(total=len(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/{}/inhale/image'.format(source))), ncols=80)
    case_list = []
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/{}/inhale/image'.format(source))):
        _, inhale_mask, _ = LoadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/{}/inhale/mask'.format(source), case))
        _, exhale_mask, _ = LoadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/{}/exhale/mask'.format(source), case))
        if np.count_nonzero(exhale_mask) >= np.count_nonzero(inhale_mask):
            case_list.append(case)
        pbar.update()
    pbar.close()
    df = pd.DataFrame(case_list)
    df.to_csv(r'/data/data1/zyh/Data/CTLung/Resample/{}/ExchangeFile.csv'.format(source), header=False, index=False)
# LogExchangeFile('SHCH')
# LogExchangeFile('Hydra')

def CheckData():
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CropData')):
        case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/CropData', case)
        if os.path.isdir(case_folder):
            case_list = os.listdir(case_folder)
            print(case, end=': ')
            if 'rigid_exhale.nii.gz' not in case_list:
                print('rigid_exhale', end=', ')
            if 'rigid_exhale_mask.nii.gz' not in case_list:
                print('rigid_exhale_mask', end=', ')
            if ('rigid_exhale_nodule.nii.gz' not in case_list) and ('inhale_nodule.nii.gz' in case_list):
                print('rigid_exhale_nodule', end=', ')
            if 'rigid_field.nii.gz' not in case_list:
                print('rigid_field', end=', ')
            if 'deform_exhale.nii.gz' not in case_list:
                print('deform_exhale', end=', ')
            if 'deform_exhale_mask.nii.gz' not in case_list:
                print('deform_exhale_mask', end=', ')
            if ('deform_exhale_nodule.nii.gz' not in case_list) and ('inhale_nodule.nii.gz' in case_list):
                print('deform_exhale_nodule', end=', ')
            if 'deform_field.nii.gz' not in case_list:
                print('deform_field', end=', ')
            if 'RigidTransformParameters.0.txt' not in case_list:
                print('RigidTransformParameters')
            if 'TransformParameters.0.txt' not in case_list:
                print('TransformParameters')
            print()
# CheckData()

# root = r'/data/data1/zyh/Data/CTLung/CNNData'
# for case in sorted(os.listdir(root)):
#     case_folder = os.path.join(root, case)
#     # if str(case_folder).endswith('.nii.gz'):
#     #     os.rename(case_folder, os.path.join(root, case.split('.nii.gz')[0]))
#     # if not os.path.exists(case_folder): os.makedirs(case_folder)
#     # shutil.move(os.path.join(root, 'moving', case), os.path.join(case_folder, 'inhale.ni.gz'))
#     # shutil.move(os.path.join(root, 'moving_mask', case), os.path.join(case_folder, 'inhale_mask.ni.gz'))
#     # shutil.move(os.path.join(root, 'fixed', case), os.path.join(case_folder, 'exhale.ni.gz'))
#     # shutil.move(os.path.join(root, 'fixed_mask', case), os.path.join(case_folder, 'exhale_mask.ni.gz'))
#     # shutil.move(os.path.join(root, 'rigid', case), os.path.join(case_folder, 'rigid_exhale.ni.gz'))
#     # shutil.move(os.path.join(root, 'rigid_mask', case), os.path.join(case_folder, 'rigid_exhale_mask.ni.gz'))
#     # shutil.move(os.path.join(root, 'rigid_deform', case), os.path.join(case_folder, 'rigid_deform.ni.gz'))
#     # shutil.move(os.path.join(root, 'rigid_deform_flow', case), os.path.join(case_folder, 'rigid_deform_flow.ni.gz'))
#     # shutil.move(os.path.join(root, 'rigid_deform_mask', case), os.path.join(case_folder, 'rigid_deform_mask.ni.gz'))
#     # shutil.move(os.path.join(root, 'deform_result', case), os.path.join(case_folder, 'deform_exhale.ni.gz'))
#     # shutil.move(os.path.join(root, 'deform_mask', case), os.path.join(case_folder, 'deform_exhale_mask.ni.gz'))
#     # shutil.move(os.path.join(root, 'deform_flow', case), os.path.join(case_folder, 'deform_flow.ni.gz'))
#     #
#     # if os.path.exists(os.path.join(root, 'moving_nodule', case)):
#     #     shutil.move(os.path.join(root, 'moving_nodule', case), os.path.join(case_folder, 'inhale_nodule.ni.gz'))
#     #     shutil.move(os.path.join(root, 'fixed_nodule', case), os.path.join(case_folder, 'exhale_nodule.ni.gz'))
#     #     shutil.move(os.path.join(root, 'rigid_nodule', case), os.path.join(case_folder, 'rigid_exhale_nodule.ni.gz'))
#     #     shutil.move(os.path.join(root, 'rigid_deform_nodule', case), os.path.join(case_folder, 'rigid_deform_nodule.ni.gz'))
#     #     shutil.move(os.path.join(root, 'deform_nodule', case), os.path.join(case_folder, 'deform_exhale_nodule.ni.gz'))
#     if os.path.isfile(case_folder): continue
#     case_list = os.listdir(case_folder)
#     for data in case_list:
#         if data.endswith('.ni.gz'):
#             os.rename(os.path.join(case_folder, data), os.path.join(case_folder, '{}.nii.gz'.format(data.split('.ni.gz')[0])))
#
# case_list = ['20220727_ye_lai_di', '20220801_yin_jian_xiang', '20220801_yuan_jia_qiao',
#              '20220802_fang_li_jing', '20220802_zhou_shi_gui', '20220805_niu_zheng_ping',
#              '20220809_liu_hai_hui', '20220809_zhu_rong_lin', '20220810_gu_jing_jie',
#              '20220810_lv_xue_xiang', '20220815_jia_dan_yu', '20220815_li_chun_hua',
#              '20220815_yu_chang_qin', '20220824_jiang_hai_tao', '20220824_pan_ru_he',
#              '20220825_cui_zong_yu', '20220829_sun_ke_xiang', '20220829_xiang_cong_ping',
#              '20220831_jiang_ming', '20220831_qian_ying', '20220902_yang_chang_fa',
#              '20220902_zheng_shou_shun', '20220907_chen_tao', '20220907_dong_qin_li',
#              '20220907_fan_xi_yuan', '20220907_li_xia', '20220907_ruan_guo_bin',
#              '20220913_sun_mei_ling', '20220913_yin_jian_xiang', '20220913_zhu_xiao_feng',
#              '20220919_gu_cui_ping', '20220919_jin_wen_cai', '20220926_gao_ping',
#              '20220926_wang_zhi_ying', '20221010_li_mei_ling']

# TODO: exchange inhale & exhale
# case_list = ['20220824_jiang_hai_tao', '20220829_sun_ke_xiang', '20220902_zheng_shou_shun',
#              '20220907_fan_xi_yuan', '20220919_gu_cui_ping', '20220919_jin_wen_cai']
# for case in case_list:
#     des_case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/ToRegistrationElastix', case)
#     if not os.path.exists(des_case_folder): os.makedirs(des_case_folder)
#     shutil.copyfile(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'inhale_nodule.nii.gz'), os.path.join(des_case_folder, 'inhale_nodule.nii.gz'))
#     shutil.copyfile(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'exhale_nodule.nii.gz'), os.path.join(des_case_folder, 'exhale_nodule.nii.gz'))
#     shutil.copyfile(os.path.join(r'/data/data1/zyh/Data/CTLung/RegistrationElastix/Rigid', case, 'TransformParameters.0.txt'), os.path.join(des_case_folder, 'RigidTransformParameters.0.txt'))
#     shutil.copyfile(os.path.join(r'/data/data1/zyh/Data/CTLung/RegistrationElastix/RigidDeform', case, 'TransformParameters.0.txt'), os.path.join(des_case_folder, 'TransformParameters.0.txt'))
#
# count = 0
# for case in os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData'):
#     if os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'inhale_nodule.nii.gz')):
#         if not os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'rigid_nodule.nii.gz')):
#             print(case)
#         count += 1
# print(count)

# case_list = ['20191115_tan_fen.nii.gz', '20200617_wu_gen_hai.nii.gz', '20200706_wu_yong_qiang.nii.gz', '20200814_zou_guo_fang.nii.gz']

# from Dataset.AllPreprocee import Normailzation, CopyInfo
from MeDIT.SaveAndLoad import LoadImage

def Crop3DEdge(mask, image, des_shape, nodule=None, center_x=None):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param array: the 3D numpy array
    :return: the cropped image.
    '''
    new_image = np.ones(shape=des_shape) * -1024
    new_mask = np.zeros(shape=des_shape)
    # new_image = np.ones(shape=[des_shape[0], des_shape[1], image.shape[2]]) * -1024
    # new_mask = np.zeros(shape=[des_shape[0], des_shape[1], mask.shape[2]])
    z, y, x = image.shape[0], image.shape[1], image.shape[2]

    is_z, is_y = False, False
    # dim0  340
    target_z = int(np.max(np.nonzero(np.sum(mask, axis=(1, 2)))))
    if target_z + 20 > z:
        is_z = True
        target_z_1 = z
        target_z_0 = max(target_z + 20 - des_shape[0], 0)
    else:
        target_z_1 = target_z + 20
        target_z_0 = max(target_z_1 - des_shape[0], 0)   #底端
    len_z = target_z_1 - target_z_0

    # dim1
    target_y = int(np.max(np.nonzero(np.sum(mask, axis=(0, 2)))))
    if target_y + 20 > y:
        is_y = True
        target_y_1 = y
        target_y_0 = max(target_y_1 + 20 - des_shape[1], 0)
    else:
        target_y_1 = target_y + 20
        target_y_0 = max(target_y_1 - des_shape[1], 0)
    len_y = target_y_1 - target_y_0

    # dim2
    if not center_x:
        nonzero_x = np.nonzero(np.sum(mask, axis=(0, 1)))[0]
        center_x = (nonzero_x[0] + nonzero_x[-1]) / 2
        target_x_0 = int(max(center_x - des_shape[2] / 2, 0))
        target_x_1 = int(min(center_x + des_shape[2] / 2, mask.shape[-1]))
    else:
        target_x_0 = center_x[0]
        target_x_1 = center_x[1]
    len_x = target_x_1 - target_x_0

    if is_z: shape_z = des_shape[0] - (target_z + 20 - z)
    else: shape_z = des_shape[0]
    if is_y: shape_y = des_shape[1] - (target_y + 20 - y)
    else: shape_y = des_shape[1]

    new_image[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y, (des_shape[2] - len_x) // 2: des_shape[2] // 2 + len_x // 2] = \
        image[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
    new_mask[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y, (des_shape[2] - len_x) // 2: des_shape[2] // 2 + len_x // 2] = \
        mask[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
    if isinstance(nodule, np.ndarray):
        new_nodule = np.zeros(shape=des_shape)
        new_nodule[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y, (des_shape[2] - len_x) // 2: des_shape[2] // 2 + len_x // 2] = \
            nodule[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
        return new_image, new_mask, new_nodule, \
               [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]
    else:
        return new_image, new_mask, nodule, \
               [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]
    # new_image[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = image[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
    # new_mask[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = mask[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
    # if isinstance(nodule, np.ndarray):
    #     new_nodule = np.zeros(shape=des_shape)
    #     new_nodule[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = nodule[target_z_0:target_z_1, target_y_0:target_y_1, target_x_0:target_x_1]
    #     return new_image, new_mask, new_nodule, \
    #            [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]
    # else:
    #     return new_image, new_mask, nodule, \
    #            [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]
    # new_image[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = \
    #     image[target_z_0:target_z_1, target_y_0:target_y_1]
    # new_mask[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = \
    #     mask[target_z_0:target_z_1, target_y_0:target_y_1]
    # if isinstance(nodule, np.ndarray):
    #     new_nodule = np.zeros(shape=[des_shape[0], des_shape[1], nodule.shape[2]])
    #     new_nodule[max(shape_z - len_z, 0):shape_z, max(shape_y - len_y, 0): shape_y] = \
    #         nodule[target_z_0:target_z_1, target_y_0:target_y_1]
    #     return new_image, new_mask, new_nodule, \
    #            [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]
    # else:
    #     return new_image, new_mask, nodule, \
    #            [target_x_0, target_y_0, target_z_0], [target_x_1, target_y_1, target_z_1]


def Crop(root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    des_shape = (256, 256, 256)
    for index, case in enumerate(sorted(case_list)):
        if case == '20201126_gu_kang_xiu.nii.gz': continue
        save_folder = os.path.join(save_root, case.split('.nii.gz')[0])
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        center_x = 0
        for c in ['inhale', 'exhale']:
            image_path = os.path.join(root, case, '{}.nii.gz'.format(c))
            mask_path = os.path.join(root, case, '{}_mask.nii.gz'.format(c))
            nodule_path = os.path.join(root, case, '{}_nodule.nii.gz'.format(c))
            try:
                image, image_arr, _ = LoadImage(image_path)
                mask, mask_arr, _ = LoadImage(mask_path)
                image_arr = np.transpose(image_arr, axes=(2, 0, 1))
                mask_arr = np.transpose(mask_arr, axes=(2, 0, 1))
                if os.path.exists(nodule_path):
                    nodule, nodule_arr, _ = LoadImage(nodule_path)
                    nodule_arr = np.transpose(nodule_arr, axes=(2, 0, 1))
                else:
                    nodule_arr = np.zeros_like(mask_arr)
                # crop mask
                if c == 'inhale':
                    crop_image_arr, crop_mask_arr, crop_nodule_arr, min_coord, max_coord = Crop3DEdge(mask_arr, image_arr, des_shape, nodule_arr)
                    center_x = [min_coord[0], max_coord[0]]
                    # print(min_coord, max_coord)
                else:
                    crop_image_arr, crop_mask_arr, crop_nodule_arr, min_coord, max_coord = Crop3DEdge(mask_arr,
                                                                                                      image_arr,
                                                                                                      des_shape,
                                                                                                      nodule_arr)
                    # print(min_coord, max_coord)
                crop_image_arr = Normailzation(crop_image_arr)

                crop_mask = CopyInfo(crop_mask_arr, image)
                crop_image = CopyInfo(crop_image_arr, image)
                sitk.WriteImage(crop_image, os.path.join(save_folder, '{}_crop.nii.gz'.format(c)))
                sitk.WriteImage(crop_mask, os.path.join(save_folder, '{}_mask_crop.nii.gz'.format(c)))

                if os.path.exists(nodule_path):
                    crop_nodule = CopyInfo(crop_nodule_arr, image)
                    sitk.WriteImage(crop_nodule, os.path.join(save_folder, '{}_nodule_crop.nii.gz'.format(c)))
            except Exception as e:
                print(case.split('.nii.gz')[0], e)
# case_list = ['20191115_tan_fen']
# Crop(root=r'/data/data1/zyh/Data/CTLung/Test',
#      save_root=r'/data/data1/zyh/Data/CTLung/Test')

# def Dice(pred, label):
#     smooth = 1
#     intersection = (pred * label).sum()
#     return (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)
#
# pred = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/20191115_tan_fen/inhale_mask_crop.nii.gz'))
# pred[pred > 1] = 1
# label = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/20191115_tan_fen/exhale_mask_crop.nii.gz'))
# label[label > 1] = 1
# print(Dice(pred, label))

# for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/inhale/image')):
#     case_name = case.split('.nii.gz')[0]
#     case_folder = os.path.join(r'D:\Data\Resample', case_name)
#     if not os.path.exists(case_folder): os.makedirs(case_folder)
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'inhale', 'image', case), os.path.join(case_folder, 'inhale.nii.gz'))
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'exhale', 'image', case), os.path.join(case_folder, 'exhale.nii.gz'))
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'inhale', 'mask', case), os.path.join(case_folder, 'inhale_mask.nii.gz'))
#     shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'exhale', 'mask', case), os.path.join(case_folder, 'exhale_mask.nii.gz'))
#     if os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'inhale', 'nodule', case)):
#         shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'inhale', 'nodule', case), os.path.join(case_folder, 'inhale_nodule.nii.gz'))
#         shutil.move(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample/SHCH', 'exhale', 'nodule', case), os.path.join(case_folder, 'exhale_nodule.nii.gz'))
# print(os.path.exists(r'D:\Data\Resample'))
# print(shutil.rmtree('D:\Data\Resample'))
# case_folder = r'/data/data1/zyh/Data/CTLung/RawData/FromHydra/Case_12'
# inhale_image = sitk.ReadImage(os.path.join(case_folder, 'inhale', 'image.nii.gz'))
# inhale_mask = sitk.ReadImage(os.path.join(case_folder, 'inhale', 'lobe.nii.gz'))
# exhale_image = sitk.ReadImage(os.path.join(case_folder, 'exhale', 'image.nii.gz'))
# exhale_image_arr = sitk.GetArrayFromImage(exhale_image)
# new_exhale_image = sitk.GetImageFromArray(exhale_image_arr)
# new_exhale_image.SetSpacing(exhale_image.GetSpacing())
# new_exhale_image.SetDirection(exhale_image.GetDirection())
# new_exhale_image.SetOrigin(inhale_image.GetOrigin())
# sitk.WriteImage(new_exhale_image, os.path.join(case_folder, 'exhale', 'image_fine_tuning.nii.gz'))
# exhale_mask = sitk.ReadImage(os.path.join(case_folder, 'exhale', 'lobe.nii.gz'))
# exhale_mask_arr = sitk.GetArrayFromImage(exhale_mask)
# new_exhale_mask = sitk.GetImageFromArray(exhale_mask_arr)
# new_exhale_mask.SetSpacing(exhale_mask.GetSpacing())
# new_exhale_mask.SetDirection(exhale_mask.GetDirection())
# new_exhale_mask.SetOrigin(inhale_image.GetOrigin())
# sitk.WriteImage(new_exhale_mask, os.path.join(case_folder, 'exhale', 'lobe_fine_tuning.nii.gz'))

# case_df = pd.read_csv(str(r'/data/data1/zyh/Data/CTLung/CropData/test.csv'), index_col=0).squeeze()
# case_list = case_df.index.tolist()
# vol_dict = {'CaseName':[], 'Volume Percent':[]}
# for case in sorted(case_list):
#     case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/CropData', case)
#     inhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'inhale_mask.nii.gz')))
#     exhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'exhale_mask.nii.gz')))
#     inhale_mask[inhale_mask > 0] = 1
#     exhale_mask[exhale_mask > 0] = 1
#     vo_per = abs(np.sum(inhale_mask) - np.sum(exhale_mask)) / np.sum(inhale_mask)
#     vol_dict['CaseName'].append(case)
#     vol_dict['Volume Percent'].append(vo_per)
# df = pd.DataFrame(vol_dict)
# df.to_csv(r'/data/data1/zyh/Data/CTLung/CropData/test_volume.csv', index='CaseName')


# inhale_image = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/inhale.nii.gz')
# flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/deform_flow.nii.gz')
# pred_flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/field_1110.nii.gz')
# flow_arr = sitk.GetArrayFromImage(flow)
# pred_flow_arr = sitk.GetArrayFromImage(pred_flow)
# diff_flow_arr = flow_arr - pred_flow_arr
# half_flow = sitk.GetImageFromArray(diff_flow_arr, isVector=True)
# half_flow.SetSpacing(flow.GetSpacing())
# half_flow.SetOrigin(flow.GetOrigin())
# half_flow.SetDirection(flow.GetDirection())
# sitk.WriteImage(half_flow, r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/deform_flow_diff_UNet.nii.gz')

# displacement_image = sitk.GetImageFromArray(half_flow_arr, isVector=True)
# displacement_image.SetOrigin()
# displacement_image.SetSpacing()
# tx = sitk.DisplacementFieldTransform(displacement_image)

# data = np.load(r'/data/data1/zyh/Data/CTLung/CropData/real_image_1118_CGAN_2D_pretrained_train.npy')
# sitk.WriteImage(sitk.GetImageFromArray(np.mean(data, axis=0).transpose(2, 1, 0)), r'/data/data1/zyh/Data/CTLung/CropData/real_image_1118_CGAN_2D_pretrained_train.nii.gz')
# data = np.load(r'/data/data1/zyh/Data/CTLung/CropData/fake_image_1118_CGAN_2D_pretrained_train.npy')
# sitk.WriteImage(sitk.GetImageFromArray(np.mean(data, axis=0).transpose(2, 1, 0)), r'/data/data1/zyh/Data/CTLung/CropData/fake_image_1118_CGAN_2D_pretrained_train.nii.gz')
# print()

from skimage.measure import compare_ssim

# inhale_image = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20210709_guo_tie/inhale.nii.gz'))
# ssim_list = []
# for i in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
#     exhale_image = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20210709_guo_tie/exhale_1117_{:.6f}.nii.gz'.format(i))
#     exhale = sitk.GetArrayFromImage(exhale_image)
#     ssim = compare_ssim(inhale_image, exhale)
#     print(i, ssim)
#     ssim_list.append(ssim)

def Data4nnUNet(data_folder, save_folder):
    for case in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, case)):
            shutil.copyfile(os.path.join(data_folder, case, 'inhale', 'image.nii.gz'),
                            os.path.join(save_folder, '{}_{}_0000.nii.gz'.format(case, 'inhale')))
            shutil.copyfile(os.path.join(data_folder, case, 'exhale', 'image.nii.gz'),
                            os.path.join(save_folder, '{}_{}_0000.nii.gz'.format(case, 'exhale')))

def nnUNet2Preprocess(data_folder):
    for case in os.listdir(data_folder):
        if case.endswith('.nii.gz'):
            case_name = case.split('.nii.gz')[0]
            shutil.move(os.path.join(data_folder, case),
                        os.path.join(data_folder,
                                     '{}_{}'.format(case.split('_')[0], case.split('_')[1]),
                                     '{}'.format(case_name.split('_')[2]),
                                     'lobe.nii.gz'))
# nnUNet2Preprocess(r'/data/data1/zyh/Data/CTLung/RawData/Hydra202211')

