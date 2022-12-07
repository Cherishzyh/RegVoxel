import math
import os
import shutil
import sys
import pandas as pd
import torch
import warnings
import numpy as np
from tqdm import tqdm
from time import sleep
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage.measure import compare_ssim
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.append('/data/data1/zyh/Project/RegVoxel')
from Utils.Transform import SpatialTransformer, AffineTransform


class ComputeMetric():
    '''
    all input images are
    '''
    def __init__(self):
        super().__init__()
        self._metric = {}

    def __Normalization(self, data, clip_ratio=0.0):
        normal_data = np.asarray(data, dtype=np.float32)
        if normal_data.max() - normal_data.min() < 1e-6:
            return np.zeros_like(normal_data)

        if clip_ratio > 1e-6:
            data_list = data.flatten().tolist()
            data_list.sort()
            normal_data.clip(data_list[int(clip_ratio / 2 * len(data_list))],
                             data_list[int((1 - clip_ratio / 2) * len(data_list))])

        normal_data = normal_data - np.min(normal_data)
        normal_data = normal_data / np.max(normal_data)
        return normal_data

    def __NormalizationImage(self, image):
        """
        Normalize an image to 0 - 1 (8bits)
        """
        normalizeFilter = sitk.NormalizeImageFilter()
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(1.0)
        resacleFilter.SetOutputMinimum(0.)

        image = normalizeFilter.Execute(image)  # set mean and std deviation
        image = resacleFilter.Execute(image)  # set intensity 0-255

        return image

    def __Image2Numpy(self, image):
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        if np.max(image) > 1 or np.min(image) < 0:
            image = self.__Normalization(image)
        return image

    def __Numpy2Image(self, array):
        if isinstance(array, np.ndarray):
            array = sitk.GetImageFromArray(array)
        if np.max(array) > 1 or np.min(array) < 0:
            array = self.__NormalizationImage(array)
        return array

    def _Dice(self, pred, label):
        smooth = 1
        intersection = (pred * label).sum()
        self._metric['Dice'] = (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)

    def _HausdorffDistanceImage(self, pred_image, label_image):
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD Image'] = hausdorff_computer.GetHausdorffDistance()

    def MaskMetric(self, pred, target):
        '''
        Evaluation metrics were calculated for one mask data, including Dice and HD
        It is best to input 【sitk.image】
        :param pred: Mask transformed by predicted deformation field
        :param target: ......
        :return: [dice, hd...]
        '''
        # TODO:  for mask & deform mask: Dice/HD
        pred[pred >= 1] = 1
        target[target >= 1] = 1
        pred_array = self.__Image2Numpy(pred)
        target_array = self.__Image2Numpy(target)
        # pred_image = self.__Numpy2Image(pred)
        # target_image = self.__Numpy2Image(target)

        self._Dice(pred_array, target_array)
        # self._HausdorffDistanceImage(pred_image, target_image)

    def _MSE(self, pred, label):
        self._metric['MSE'] = np.mean((pred - label) ** 2)

    def _PSNR(self, pred, label):
        mse = np.mean((pred - label) ** 2)
        if mse == 0:
            self._metric['PSNR'] = 100
        else:
            PIXEL_MAX = 1
            self._metric['PSNR'] = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def _SSIM(self, pred, label):
        self._metric['SSIM'] = compare_ssim(pred, label)

    def ImageMetric(self, pred, target):
        '''
        Evaluation metrics were calculated for one image data, including MSE and SSIM
        It is best to input 【sitk.image】
        :param pred: image transformed by predicted deformation iamge
        :param target: ......
        :return: [dice, hd...]
        '''
        # TODO:  image & deform image: mse/SSIM
        pred_array = self.__Image2Numpy(pred)
        target_array = self.__Image2Numpy(target)

        self._MSE(pred_array, target_array)
        self._PSNR(pred_array, target_array)
        self._SSIM(pred_array, target_array)

    def Excute(self):
        return self._metric


class ComputeMetricTensor():
    '''
    all input images are sitk.image
    '''

    def __init__(self):
        super().__init__()
        self._metric = {}

    def __Image2Tensor(self, image):
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
            image = torch.from_numpy(image)
        if torch.max(image) > 1 or torch.min(image) <0:
            image = self.__Normalization(image)
        return image

    def __Normalization(self, data):
        normal_data = torch.tensor(data, dtype=torch.float32)
        if normal_data.max() - normal_data.min() < 1e-6:
            return torch.zeros_like(normal_data)
        normal_data = normal_data - torch.min(normal_data)
        normal_data = normal_data / torch.max(normal_data)
        return normal_data

    def _Dice(self, pred, label):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = label.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        self._metric['Dice'] = (2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    def _HausdorffDistanceImage(self, pred_image, label_image):
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD Image'] = hausdorff_computer.GetHausdorffDistance()

    def MaskMetric(self, pred, target):
        '''
        Evaluation metrics were calculated for one mask data, including Dice and HD
        It is best to input 【sitk.image】
        :param pred: Mask transformed by predicted deformation field
        :param target: ......
        :return: [dice, hd...]
        '''
        # TODO:  for mask & deform mask: Dice/HD
        pred[pred >= 1] = 1
        target[target >= 1] = 1
        pred_tensor = self.__Image2Tensor(pred)
        target_tensor = self.__Image2Tensor(target)

        self._Dice(pred_tensor, target_tensor)
        # self._HausdorffDistanceImage(pred, target)

    def _MSE(self, pred, label):
        self._metric['MSE'] = torch.mean((pred - label) ** 2)

    def _PSNR(self, pred, label):
        mse = torch.mean((pred - label) ** 2)
        if mse == 0:
            self._metric['PSNR'] = 100
        else:
            PIXEL_MAX = 1
            self._metric['PSNR'] = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def _SSIM(self, pred, label):
        u_true = torch.mean(label)
        u_pred = torch.mean(pred)
        var_true = torch.var(label)
        var_pred = torch.var(pred)
        std_true = torch.sqrt(var_true)
        std_pred = torch.sqrt(var_pred)
        c1 = torch.square(torch.tensor(0.01 * 7))
        c2 = torch.square(torch.tensor(0.03 * 7))
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        self._metric['SSIM'] = ssim / denom

    def ImageMetric(self, pred, target):
        '''
        Evaluation metrics were calculated for one image data, including MSE and SSIM
        It is best to input 【sitk.image】
        :param pred: image transformed by predicted deformation iamge
        :param target: ......
        :return: [dice, hd...]
        '''
        # TODO:  image & deform image: mse/SSIM
        pred_tensor = self.__Image2Tensor(pred)
        target_tensor = self.__Image2Tensor(target)

        self._MSE(pred_tensor, target_tensor)
        self._PSNR(pred_tensor, target_tensor)
        self._SSIM(pred_tensor, target_tensor)

    def Excute(self):
        return self._metric


class ComputePointMoving():
    def __init__(self):
        super(ComputePointMoving, self).__init__()

    def GetCenter(self, mask):
        mask = np.squeeze(mask)
        assert np.ndim(mask) == 3
        if np.sum(mask) == 0:
            return [0, 0, 0]
        nonzero_x = np.nonzero(np.sum(mask, axis=(1, 2)))[0]
        nonzero_y = np.nonzero(np.sum(mask, axis=(0, 2)))[0]
        nonzero_z = np.nonzero(np.sum(mask, axis=(0, 1)))[0]
        center_0 = (nonzero_x[0] + nonzero_x[-1]) / 2
        center_1 = (nonzero_y[0] + nonzero_y[-1]) / 2
        center_2 = (nonzero_z[0] + nonzero_z[-1]) / 2
        return [center_0, center_1, center_2]

    def GetDistance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

    def Accuracy(self, point, mask):
        if mask[int(point[0]), int(point[1]), int(point[2])] == 1:
            return 1
        else:
            return 0

    def Excute(self, deform, fixed, return_center=False):
        if isinstance(deform, torch.Tensor):
            deform = deform.numpy()
        if isinstance(fixed, torch.Tensor):
            fixed = fixed.numpy()
        deform_center = self.GetCenter(deform)
        fixed_center = self.GetCenter(fixed)

        if return_center:
            return deform_center, fixed_center, \
                   self.GetDistance(deform_center, fixed_center), self.Accuracy(deform_center, fixed)
        else:
            return self.GetDistance(deform_center, fixed_center), self.Accuracy(deform_center, fixed)


class ComputePointMovingByImage():
    def __init__(self):
        super(ComputePointMovingByImage, self).__init__()
        self.intensity_statistic = sitk.LabelIntensityStatisticsImageFilter()
        self.overlap_statistic = sitk.LabelOverlapMeasuresImageFilter()
        self.cast_filter = sitk.CastImageFilter()
        self.median_filder = sitk.MedianImageFilter()
        self.statistics_filter = sitk.StatisticsImageFilter()

    def GetGravityCenter(self, mask, foreground_value=1):
        self.intensity_statistic.Execute(mask, mask==foreground_value)
        center_gravity = self.intensity_statistic.GetCenterOfGravity(foreground_value)
        center_gravity_coordiate = mask.TransformPhysicalPointToIndex(center_gravity)
        return center_gravity_coordiate

    def GetDistance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

    def GetSum(self, mask):
        self.statistics_filter.Execute(mask)
        return self.statistics_filter.GetSum()

    def Accuracy(self, point, mask):
        if mask.GetPixel(point) == 1:
            return 1
        else:
            return 0

    def KeepLargest(self, mask):
        component_mask = sitk.ConnectedComponent(mask)
        sorted_component_mask = sitk.RelabelComponent(component_mask, sortByObjectSize=True)
        largest_component_binary_mask = sorted_component_mask == 1
        return largest_component_binary_mask

    def GetDice(self, mask1, mask2):
        self.overlap_statistic.Execute(mask1, mask2)
        return self.overlap_statistic.GetDiceCoefficient()

    def CaseImage(self, image, dtype=sitk.sitkFloat32):
        self.cast_filter.SetOutputPixelType(dtype)
        return self.cast_filter.Execute(image)

    def MedianFilter(self, image):
        return sitk.BinaryMedian(image)

    def Excute(self, deform_mask, fixed_mask, return_center=False):
        deform_center = self.GetGravityCenter(deform_mask)
        fixed_center = self.GetGravityCenter(fixed_mask)

        if return_center:
            return deform_center, fixed_center, \
                   self.GetDistance(deform_center, fixed_center), self.Accuracy(deform_center, fixed_mask)
        else:
            return self.GetDistance(deform_center, fixed_center), self.Accuracy(deform_center, fixed_mask)


def KeepLargest(mask):
    new_mask = np.zeros(mask.shape)
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    index = np.argmax(max_volume)
    new_mask[label_im == index + 1] = 1
    return new_mask


def ImageTransform(moving_image, field, mask=False, affine=True, theta=None):
    warpper = SpatialTransformer(moving_image.shape[-3])
    if affine and theta:
        warp_image = AffineTransform(moving_image, theta)
    else:
        warp_image = moving_image
    wrap_image = warpper(warp_image, field, mask=mask)
    if mask:
        wrap_image = torch.round(wrap_image).numpy()
        wrap_image = median_filter(wrap_image, size=3)
    return wrap_image


def ComputeElastix(is_image=False, case_list=[], fixed_key='fixed', write_csv=True):
    data_root = '/data/data1/zyh/Data/CTLung/CNNData/'
    if write_csv:
        mse_root = r'/data/data1/zyh/Data/CTLung/CNNData/Mse.csv'
        dice_root = r'/data/data1/zyh/Data/CTLung/CNNData/Dice.csv'
        dist_root = r'/data/data1/zyh/Data/CTLung/CNNData/Distance.csv'
        new_mse_df = pd.DataFrame()
        new_dice_df = pd.DataFrame()
        new_dist_df = pd.DataFrame()
    if is_image:
        figure_root = r'/data/data1/zyh/Data/CTLung/CNNData/AImage/ImageRigid'
        if not os.path.exists(figure_root): os.makedirs(figure_root)
    deform_dict = {}
    moving_distance = ComputePointMoving()
    if len(case_list) == 0:
        case_list = os.listdir(data_root)
    len_nodule = 0
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        sleep(0.1)

        if fixed_key == 'fixed':
            fixed_folder = os.path.join(data_root, case, 'exhale.nii.gz')
            fixed_mask_folder = os.path.join(data_root, case, 'exhale_mask.nii.gz')
            fixed_nodule_folder = os.path.join(data_root, case, 'exhale_nodule.nii.gz')
            deform_folder = os.path.join(data_root, case, 'deform_result.nii.gz')
            deform_mask_folder = os.path.join(data_root, case, 'deform_mask.nii.gz')
            deform_nodule_folder = os.path.join(data_root, case, 'deform_nodule.nii.gz')
        else:
            fixed_folder = os.path.join(data_root, case, '{}_exhale.nii.gz'.format(fixed_key))
            fixed_mask_folder = os.path.join(data_root, case, '{}_exhale_mask.nii.gz'.format(fixed_key))
            fixed_nodule_folder = os.path.join(data_root, case, '{}_exhale_nodule.nii.gz'.format(fixed_key))
            deform_folder = os.path.join(data_root, case, '{}_deform.nii.gz'.format(fixed_key))
            deform_mask_folder = os.path.join(data_root, case, '{}_deform_mask.nii.gz'.format(fixed_key))
            deform_nodule_folder = os.path.join(data_root, case, '{}_deform_nodule.nii.gz'.format(fixed_key))

        ########### fixed image ###########
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_folder))
        ########### fixed mask ###########
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_folder))
        ########### fixed nodule ###########
        if os.path.exists(fixed_nodule_folder):
            fixed_nodule = sitk.GetArrayFromImage(sitk.ReadImage(fixed_nodule_folder))
            fixed_nodule = KeepLargest(fixed_nodule)
        ########### predict image ###########
        deform = sitk.GetArrayFromImage(sitk.ReadImage(deform_folder))
        ########### predict mask ###########
        deform_mask = sitk.GetArrayFromImage(sitk.ReadImage(deform_mask_folder))
        deform_mask = np.around(deform_mask)
        deform_mask = median_filter(deform_mask, size=3).astype(np.int32)
        ########### predict nodule ###########
        if os.path.exists(fixed_nodule_folder):
            deform_nodule = sitk.GetArrayFromImage(sitk.ReadImage(deform_nodule_folder))
            deform_nodule = np.around(deform_nodule)
            deform_nodule = KeepLargest(deform_nodule)
            # deform_nodule = median_filter(deform_nodule, size=3).astype(np.int32)

        evaluate = ComputeMetric()
        evaluate.ImageMetric(deform, fixed)
        evaluate.MaskMetric(deform_mask, fixed_mask)
        deform_numpy_metric = evaluate.Excute()
        if is_image:
            # 存一张mask和nodule最大层
            max_mask = np.argmax(np.sum(fixed_mask, axis=(0, 2)))
            plt.title(case.split('.nii.gz')[0])
            plt.subplot(121)
            plt.imshow(np.flip(fixed[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(deform_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(np.flip(deform[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(deform_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('mask', case.split('.nii.gz')[0])),
                        bbox_inches='tight', dpi=300)
            plt.close()
        if os.path.exists(fixed_nodule_folder):
            # Distance of nodule
            deform_center, fixed_center, nodule_distance = moving_distance.Excute(deform_nodule, fixed_nodule)
            deform_numpy_metric['Distance'] = nodule_distance
            len_nodule += 1
            if is_image:
                max_nodule = np.argmax(np.sum(fixed_nodule, axis=(0, 2)))
                plt.title(case.split('.nii.gz')[0])
                plt.subplot(121)
                plt.imshow(np.flip(fixed[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(deform_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(np.flip(deform[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(deform_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('nodule', case.split('.nii.gz')[0])),
                            bbox_inches='tight', dpi=300)
                plt.close()

        else:
            deform_numpy_metric['Distance'] = 0

        deform_dict = dict(Counter(deform_numpy_metric) + Counter(deform_dict))
        if write_csv:
            new_mse_df.loc[case.split('.nii.gz')[0], '{}'.format('Elastix')] = deform_numpy_metric['MSE']
            new_dice_df.loc[case.split('.nii.gz')[0], '{}'.format('Elastix')] = deform_numpy_metric['Dice']
            new_dist_df.loc[case.split('.nii.gz')[0], '{}'.format('Elastix')] = deform_numpy_metric['Distance'] * 1.35
        pbar.update()
    pbar.close()
    if write_csv:
        if (os.path.exists(mse_root) and os.path.exists(dice_root) and os.path.exists(dist_root)):
            mse_df = pd.read_csv(mse_root, index_col=0)
            dice_df = pd.read_csv(dice_root, index_col=0)
            dist_df = pd.read_csv(dist_root, index_col=0)
            pd.concat([mse_df, new_mse_df], axis=1).to_csv(mse_root, header=True)
            pd.concat([dice_df, new_dice_df], axis=1).to_csv(dice_root, header=True)
            pd.concat([dist_df, new_dist_df], axis=1).to_csv(dist_root, header=True)
        else:
            new_mse_df.to_csv(mse_root, header=True)
            new_dice_df.to_csv(dice_root, header=True)
            new_dist_df.to_csv(dist_root, header=True)
    print(deform_dict)
    for key in deform_dict.keys():
        if key == 'Distance':
            print(key, deform_dict[key] / len_nodule * 1.35)
        else:
            if key in ['PSNR', 'SSIM']: continue
            print(key, deform_dict[key] / len(case_list))


def ComputeTransform(suffix='0921_SingleDeform', fixed_key='exhale', is_image=False, case_list=[], write_csv=True):
    print(suffix)
    data_root = '/data/data1/zyh/Data/CTLung/CropData'
    if not fixed_key == 'exhale':
        model_suffix = suffix
        suffix = '{}_{}'.format(suffix, fixed_key)
    else:
        model_suffix = suffix
    if is_image:
        figure_root = r'/data/data1/zyh/Data/CTLung/CNNData/AImage/Image_{}'.format(suffix)
        if not os.path.exists(figure_root): os.makedirs(figure_root)
    if write_csv:
        mse_root = r'/data/data1/zyh/Data/CTLung/CropData/Mse.csv'
        dice_root = r'/data/data1/zyh/Data/CTLung/CropData/Dice.csv'
        dist_root = r'/data/data1/zyh/Data/CTLung/CropData/Distance.csv'
        acc_root = r'/data/data1/zyh/Data/CTLung/CropData/Accuracy.csv'
        new_mse_df = pd.DataFrame()
        new_dice_df = pd.DataFrame()
        new_dist_df = pd.DataFrame()
        new_acc_df = pd.DataFrame()

    deform_dict = {}
    moving_distance = ComputePointMoving()
    len_nodule = 0
    if len(case_list) == 0:
        case_list = os.listdir(os.path.join(data_root))
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        sleep(0.1)
        fixed_folder = os.path.join(data_root, case, '{}.nii.gz'.format(fixed_key))
        fixed_mask_folder = os.path.join(data_root, case, '{}_mask.nii.gz'.format(fixed_key))
        fixed_nodule_folder = os.path.join(data_root, case, '{}_nodule.nii.gz'.format(fixed_key))
        deform_folder = os.path.join(data_root, case, 'exhale_{}.nii.gz'.format(model_suffix))
        deform_mask_folder = os.path.join(data_root, case, 'lung_{}.nii.gz'.format(model_suffix))
        deform_nodule_folder = os.path.join(data_root, case, 'nodule_{}.nii.gz'.format(model_suffix))
        if not os.path.exists(deform_mask_folder): continue

        ########### fixed image ###########
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_folder))
        ########### fixed mask ###########
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_folder))
        ########### fixed nodule ###########
        if os.path.exists(fixed_nodule_folder):
            fixed_nodule = sitk.GetArrayFromImage(sitk.ReadImage(fixed_nodule_folder))
            fixed_nodule = KeepLargest(fixed_nodule)
        ########### predict image ###########
        deform = sitk.GetArrayFromImage(sitk.ReadImage(deform_folder))
        ########### predict mask ###########
        deform_mask = sitk.GetArrayFromImage(sitk.ReadImage(deform_mask_folder))
        deform_mask = np.around(deform_mask)
        deform_mask = median_filter(deform_mask, size=3).astype(np.int32)
        ########### predict nodule ###########
        if os.path.exists(deform_nodule_folder):
            deform_nodule = sitk.GetArrayFromImage(sitk.ReadImage(deform_nodule_folder))
            deform_nodule = np.around(deform_nodule)
            if np.sum(deform_nodule) == 0: continue
            deform_nodule = KeepLargest(deform_nodule)
            # if np.sum(median_filter(deform_nodule, size=3).astype(np.int32)) > 0:
            #     deform_nodule = median_filter(deform_nodule, size=3).astype(np.int32)

        evaluate = ComputeMetric()
        evaluate.ImageMetric(deform, fixed)
        evaluate.MaskMetric(deform_mask, fixed_mask)
        deform_numpy_metric = evaluate.Excute()
        if is_image:
            # 存一张mask和nodule最大层
            max_mask = np.argmax(np.sum(fixed_mask, axis=(0, 2)))
            plt.title(case.split('.nii.gz')[0])
            plt.subplot(121)
            plt.imshow(np.flip(fixed[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(deform_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(np.flip(deform[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(deform_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('mask', case.split('.nii.gz')[0])), bbox_inches='tight', dpi=300)
            plt.close()
        if os.path.exists(deform_nodule_folder):
            # Distance of nodule
            nodule_distance, puncture_accuracy = moving_distance.Excute(deform_nodule, fixed_nodule, return_center=False)
            deform_numpy_metric['Distance'] = nodule_distance
            deform_numpy_metric['Accuracy'] = puncture_accuracy
            len_nodule += 1
            if is_image:
                max_nodule = np.argmax(np.sum(fixed_nodule, axis=(0, 2)))
                plt.title(case.split('.nii.gz')[0])
                plt.subplot(121)
                plt.imshow(np.flip(fixed[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(deform_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(np.flip(deform[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(deform_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('nodule', case.split('.nii.gz')[0])), bbox_inches='tight', dpi=300)
                plt.close()

        else:
            deform_numpy_metric['Distance'] = 0
            deform_numpy_metric['Accuracy'] = 0
        # print(case, deform_numpy_metric)
        deform_dict = dict(Counter(deform_numpy_metric) + Counter(deform_dict))
        if write_csv:
            new_mse_df.loc[case.split('.nii.gz')[0], 'MSE_{}'.format(suffix)] = deform_numpy_metric['MSE']
            new_dice_df.loc[case.split('.nii.gz')[0], 'Dice_{}'.format(suffix)] = deform_numpy_metric['Dice']
            new_dist_df.loc[case.split('.nii.gz')[0], 'Distance_{}'.format(suffix)] = deform_numpy_metric['Distance']*1.35
            new_acc_df.loc[case.split('.nii.gz')[0], 'Accuracy_{}'.format(suffix)] = deform_numpy_metric['Accuracy']
        pbar.update()
    pbar.close()
    if write_csv:
        if (os.path.exists(mse_root) and os.path.exists(dice_root) and os.path.exists(dist_root) and os.path.exists(acc_root)):
            mse_df = pd.read_csv(mse_root, index_col=0)
            dice_df = pd.read_csv(dice_root, index_col=0)
            dist_df = pd.read_csv(dist_root, index_col=0)
            acc_df = pd.read_csv(acc_root, index_col=0)
            pd.concat([mse_df, new_mse_df], axis=1).to_csv(mse_root, header=True)
            pd.concat([dice_df, new_dice_df], axis=1).to_csv(dice_root, header=True)
            pd.concat([dist_df, new_dist_df], axis=1).to_csv(dist_root, header=True)
            pd.concat([acc_df, new_acc_df], axis=1).to_csv(acc_root, header=True)
        else:
            new_mse_df.to_csv(mse_root, header=True)
            new_dice_df.to_csv(dice_root, header=True)
            new_dist_df.to_csv(dist_root, header=True)
            new_acc_df.to_csv(acc_root, header=True)

    print(deform_dict)
    for key in deform_dict.keys():
        if key == 'Distance':
            print(key, deform_dict[key] / len_nodule * 1.35)
        elif key == 'Accuracy':
            print(key, deform_dict[key] / len_nodule)
        else:
            if key in ['PSNR', 'SSIM']: continue
            print(key, deform_dict[key] / len(case_list))


def ComputeTransformImage(suffix='0921_SingleDeform', fixed_key='exhale', is_image=False, case_list=[], write_csv=True):
    print(suffix)
    data_root = '/data/data1/zyh/Data/CTLung/CropData'
    if not fixed_key == 'exhale':
        model_suffix = suffix
        suffix = '{}_{}'.format(suffix, fixed_key)
    else:
        model_suffix = suffix

    deform_dict = {}
    moving_distance = ComputePointMovingByImage()
    len_nodule = 0
    if len(case_list) == 0:
        case_list = os.listdir(os.path.join(data_root))
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        deform_numpy_metric = {}
        sleep(0.1)
        fixed_nodule_folder = os.path.join(data_root, case, '{}_nodule.nii.gz'.format(fixed_key))
        deform_nodule_folder = os.path.join(data_root, case, 'nodule_{}.nii.gz'.format(model_suffix))

        ########### fixed nodule ###########
        if os.path.exists(fixed_nodule_folder):
            fixed_nodule_image = sitk.ReadImage(fixed_nodule_folder)
            fixed_nodule_image = moving_distance.KeepLargest(fixed_nodule_image)
            len_nodule += 1
            if moving_distance.GetSum(fixed_nodule_image) == 0: continue

        ########### predict nodule ###########
        if os.path.exists(deform_nodule_folder):
            deform_nodule_image = sitk.ReadImage(deform_nodule_folder)
            deform_nodule_image = moving_distance.CaseImage(deform_nodule_image, dtype=sitk.sitkInt32)
            deform_nodule_image = moving_distance.MedianFilter(deform_nodule_image)
            if moving_distance.GetSum(deform_nodule_image) == 0: continue

        # deform_numpy_metric['Dice'] = moving_distance.GetDice(deform_mask_image, fixed_mask_image)
        if os.path.exists(deform_nodule_folder):
            # Distance of nodule
            nodule_distance, puncture_accuracy = moving_distance.Excute(deform_nodule_image, fixed_nodule_image, return_center=False)
            deform_numpy_metric['Distance'] = nodule_distance
            deform_numpy_metric['Accuracy'] = puncture_accuracy

        else:
            deform_numpy_metric['Distance'] = 0
            deform_numpy_metric['Accuracy'] = 0
        # print(case, deform_numpy_metric)
        deform_dict = dict(Counter(deform_numpy_metric) + Counter(deform_dict))
        pbar.update()
    pbar.close()
    print(deform_dict)
    for key in deform_dict.keys():
        if key == 'Distance':
            print(key, deform_dict[key] / len_nodule * 1.35)
        elif key == 'Accuracy':
            print(key, deform_dict[key] / len_nodule)
        else:
            if key in ['PSNR', 'SSIM']: continue
            print(key, deform_dict[key] / len(case_list))


def ComputeInEx(is_image=False, fixed_key='exhale', case_list=[], write_csv=True):
    data_root = '/data/data1/zyh/Data/CTLung/CropData'

    if write_csv:
        mse_root = r'/data/data1/zyh/Data/CTLung/CropData/Mse.csv'
        dice_root = r'/data/data1/zyh/Data/CTLung/CropData/Dice.csv'
        dist_root = r'/data/data1/zyh/Data/CTLung/CropData/Distance.csv'
        acc_root = r'/data/data1/zyh/Data/CTLung/CropData/Accuracy.csv'
        new_mse_df = pd.DataFrame()
        new_dice_df = pd.DataFrame()
        new_dist_df = pd.DataFrame()
        new_acc_df = pd.DataFrame()
    if is_image:
        if fixed_key == 'fixed':
            figure_root = r'/data/data1/zyh/Data/CTLung/CNNData/AImage/ImageMoving'
        else:
            figure_root = r'/data/data1/zyh/Data/CTLung/CNNData/AImage/ImageMoving_{}'.format(fixed_key)
        if not os.path.exists(figure_root): os.mkdir(figure_root)

    moving_dict = {}
    moving_distance = ComputePointMoving()
    len_nodule = 0
    if len(case_list) == 0:
        case_list = os.listdir(os.path.join(data_root, 'deform_flow'))
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        sleep(0.1)
        fixed_folder = os.path.join(data_root, case, '{}.nii.gz'.format(fixed_key))
        fixed_mask_folder = os.path.join(data_root, case, '{}_mask.nii.gz'.format(fixed_key))
        fixed_nodule_folder = os.path.join(data_root, case, '{}_nodule.nii.gz'.format(fixed_key))
        moving_folder = os.path.join(data_root, case, 'inhale.nii.gz')
        moving_mask_folder = os.path.join(data_root, case, 'inhale_mask.nii.gz')
        moving_nodule_folder = os.path.join(data_root, case, 'inhale_nodule.nii.gz')

        ########### fixed image ###########
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_folder))
        ########### fixed mask ###########
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_folder))
        ########### fixed nodule ###########
        if os.path.exists(fixed_nodule_folder):
            fixed_nodule = sitk.GetArrayFromImage(sitk.ReadImage(fixed_nodule_folder))
            fixed_nodule = KeepLargest(fixed_nodule)
        ########### predict image ###########
        moving = sitk.GetArrayFromImage(sitk.ReadImage(moving_folder))
        ########### predict mask ###########
        moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask_folder))
        ########### predict nodule ###########
        if os.path.exists(fixed_nodule_folder):
            moving_nodule = sitk.GetArrayFromImage(sitk.ReadImage(moving_nodule_folder))
            moving_nodule = KeepLargest(moving_nodule)

        evaluate = ComputeMetric()
        evaluate.ImageMetric(moving, fixed)
        evaluate.MaskMetric(moving_mask, fixed_mask)
        moving_numpy_metric = evaluate.Excute()
        if is_image:
            max_mask = np.argmax(np.sum(fixed_mask, axis=(0, 2)))
            plt.title(case.split('.nii.gz')[0])
            plt.subplot(121)
            plt.imshow(np.flip(fixed[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(moving_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(np.flip(moving[:, max_mask], axis=0), cmap='gray')
            plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r')
            plt.contour(np.flip(moving_mask[:, max_mask], axis=0), colors='g')
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('mask', case.split('.nii.gz')[0])),
                        bbox_inches='tight', dpi=300)
            plt.close()
        if os.path.exists(fixed_nodule_folder):
            # Distance of nodule
            nodule_distance, nodule_accuracy = moving_distance.Excute(moving_nodule, fixed_nodule, return_center=False)
            moving_numpy_metric['Distance'] = nodule_distance
            moving_numpy_metric['Accuracy'] = nodule_accuracy
            len_nodule += 1
            if is_image:
                max_nodule = np.argmax(np.sum(fixed_nodule, axis=(0, 2)))
                plt.title(case.split('.nii.gz')[0])
                plt.subplot(121)
                plt.imshow(np.flip(fixed[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(moving_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(np.flip(moving[:, max_nodule], axis=0), cmap='gray')
                plt.contour(np.flip(fixed_nodule[:, max_nodule], axis=0), colors='r')
                plt.contour(np.flip(moving_nodule[:, max_nodule], axis=0), colors='g')
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(os.path.join(figure_root, '{}_{}.jpg'.format('nodule', case.split('.nii.gz')[0])),
                            bbox_inches='tight', dpi=300)
                plt.close()
        else:
            moving_numpy_metric['Distance'] = 0
            moving_numpy_metric['Accuracy'] = 0
        moving_dict = dict(Counter(moving_numpy_metric) + Counter(moving_dict))
        if write_csv:
            if fixed_key == 'fixed':
                new_mse_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving')] = moving_numpy_metric['MSE']
                new_dice_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving')] = moving_numpy_metric['Dice']
                new_dist_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving')] = moving_numpy_metric['Distance'] * 1.35
                new_acc_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving')] = moving_numpy_metric['Accuracy'] * 1.35

            else:
                new_mse_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving_{}'.format(fixed_key))] = moving_numpy_metric['MSE']
                new_dice_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving_{}'.format(fixed_key))] = moving_numpy_metric['Dice']
                new_dist_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving_{}'.format(fixed_key))] = moving_numpy_metric['Distance'] * 1.35
                new_acc_df.loc[case.split('.nii.gz')[0], '{}'.format('Moving_{}'.format(fixed_key))] = moving_numpy_metric['Accuracy'] * 1.35

        pbar.update()
    pbar.close()
    if write_csv:
        if (os.path.exists(mse_root) and os.path.exists(dice_root) and os.path.exists(dist_root) and os.path.exists(acc_root)):
            mse_df = pd.read_csv(mse_root, index_col=0)
            dice_df = pd.read_csv(dice_root, index_col=0)
            dist_df = pd.read_csv(dist_root, index_col=0)
            acc_df = pd.read_csv(acc_root, index_col=0)
            pd.concat([mse_df, new_mse_df], axis=1).to_csv(mse_root, header=True)
            pd.concat([dice_df, new_dice_df], axis=1).to_csv(dice_root, header=True)
            pd.concat([dist_df, new_dist_df], axis=1).to_csv(dist_root, header=True)
            pd.concat([acc_df, new_acc_df], axis=1).to_csv(acc_root, header=True)
        else:
            new_mse_df.to_csv(mse_root, header=True)
            new_dice_df.to_csv(dice_root, header=True)
            new_dist_df.to_csv(dist_root, header=True)
            new_acc_df.to_csv(acc_root, header=True)
    print(moving_dict)
    for key in moving_dict.keys():
        if key == 'Distance':
            print(key, moving_dict[key] / len_nodule * 1.35)
        elif key == 'Accuracy':
            print(key, moving_dict[key] / len_nodule)
        else:
            if key in ['PSNR', 'SSIM']: continue
            print(key, moving_dict[key] / len(case_list))


def CheckData():
    root = r'/data/data1/zyh/Data/CTLung/CropData'
    image_root = os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/AImage/ImageDeform')
    if not os.path.exists(image_root): os.makedirs(image_root)
    pbar = tqdm(total=len(os.listdir(root)), ncols=80)
    for case in sorted(os.listdir(root)):
        if case == '20201126_gu_kang_xiu': continue
        if case.endswith('.csv'): continue
        moving_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'inhale.nii.gz')))
        moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'inhale_mask.nii.gz')))
        moving_mask[moving_mask > 1] = 1
        if os.path.exists(os.path.join(root, case, 'inhale_nodule.nii.gz')):
            moving_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'inhale_nodule.nii.gz')))
        fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'rigid.nii.gz')))
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'rigid_mask.nii.gz')))
        fixed_mask[fixed_mask > 1] = 1
        if os.path.exists(os.path.join(root, case, 'rigid_nodule.nii.gz')):
            fixed_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'rigid_nodule.nii.gz')))

        # if os.path.exists(os.path.join(root, case, 'rigid_nodule.nii.gz')):
        #     max_mask = np.argmax(np.sum(fixed_nodule, axis=(0, 2)))
        # else:
        #     max_mask = np.argmax(np.sum(fixed_mask, axis=(0, 2)))


        deform_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'deform.nii.gz')))
        deform_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'deform_mask.nii.gz')))
        if os.path.exists(os.path.join(root, case, 'deform_nodule.nii.gz')):
            deform_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'deform_nodule.nii.gz')))

        if os.path.exists(os.path.join(root, 'rigid_nodule', case)):
            max_mask = np.argmax(np.sum(fixed_nodule, axis=(0, 2)))
        else:
            max_mask = np.argmax(np.sum(fixed_mask, axis=(0, 2)))
        plt.subplot(131)
        plt.title('moving')
        plt.imshow(np.flip(moving_image[:, max_mask], axis=0), cmap='gray')
        plt.contour(np.flip(moving_mask[:, max_mask], axis=0), colors='r', linewidths=0.25)
        if os.path.exists(os.path.join(root, 'moving_nodule', case)):
            plt.contour(np.flip(moving_nodule[:, max_mask], axis=0), colors='g', linewidths=0.25)
        plt.axis('off')
        plt.subplot(132)
        plt.title('fixed')
        plt.imshow(np.flip(fixed_image[:, max_mask], axis=0), cmap='gray')
        plt.contour(np.flip(fixed_mask[:, max_mask], axis=0), colors='r', linewidths=0.25)
        if os.path.exists(os.path.join(root, 'rigid_nodule', case)):
            plt.contour(np.flip(fixed_nodule[:, max_mask], axis=0), colors='g', linewidths=0.25)
        plt.axis('off')
        plt.subplot(133)
        plt.title('deform')
        plt.imshow(np.flip(deform_image[:, max_mask], axis=0), cmap='gray')
        plt.contour(np.flip(deform_mask[:, max_mask], axis=0), colors='r', linewidths=0.25)
        if os.path.exists(os.path.join(root, 'deform_nodule', case)):
            plt.contour(np.flip(deform_nodule[:, max_mask], axis=0), colors='g', linewidths=0.25)
        plt.axis('off')
        plt.savefig(os.path.join(image_root, '{}.jpg'.format(case.split('.nii.gz')[0])),
                    bbox_inches='tight', dpi=300)
        plt.close()
        pbar.update()
    pbar.close()


def SplitMetric():
    dice_df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/Dice.csv', index_col=0)
    dist_df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/Distance.csv', index_col=0)
    if os.path.exists(r'/data/data1/zyh/Data/CTLung/CNNData/volume.csv'):
        df = pd.read_csv(r'/data/data1/zyh/Data/CTLung/CNNData/volume.csv')
    else:
        volume_dict = {}
        volume_per_dict = {}
        pbar = tqdm(total=len(os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving')), ncols=80)
        for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving')):
            sleep(0.1)
            inhale_path = os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/moving_mask', case)
            inhale_mask = sitk.ReadImage(inhale_path)
            exhale_path = os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/fixed_mask', case)
            exhale_mask = sitk.ReadImage(exhale_path)
            inhale_volume = np.count_nonzero(sitk.GetArrayFromImage(inhale_mask))
            exhale_volume = np.count_nonzero(sitk.GetArrayFromImage(exhale_mask))
            volume = (inhale_volume - exhale_volume) / inhale_volume
            volume_dict[case.split('.nii.gz')[0]] = [inhale_volume, exhale_volume, volume]
            volume_per_dict[case.split('.nii.gz')[0]] = volume
            pbar.update()
        pbar.close()
        df = pd.DataFrame(volume_dict, index=['inhale_volume', 'exhale_volume', 'volume_percent']).T
        df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/volume.csv')
    dice_df = pd.concat([dice_df, df], axis=1, join="inner")
    dice_df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/Dice_volume.csv')
    dist_df = pd.concat([dist_df, df], axis=1, join="inner")
    dist_df.to_csv(r'/data/data1/zyh/Data/CTLung/CNNData/Distance_volume.csv')


def Dice(pred, label):
    smooth = 1
    intersection = (pred * label).sum()
    return (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)


def ComparePreprocessWay():
    print('RightCrop\tRigid+Crop\tCrop+Rigid\tCrop')
    dist = ComputePointMoving()
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/CropData')):
        if case == '20201126_gu_kang_xiu': continue
        print(case, end=',')
        try:
            case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/CropData', case)
            inhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'inhale_mask.nii.gz')))
            inhale_mask[inhale_mask > 1] = 1
            exhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'exhale_mask.nii.gz')))
            exhale_mask[exhale_mask > 1] = 1
            rigid_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'rigid_mask.nii.gz')))
            rigid_mask[rigid_mask > 1] = 1
            crop_rigid_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'rigid_mask.nii.gz')))
            crop_rigid_mask[crop_rigid_mask > 1] = 1
            crop_inhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'inhale_mask.nii.gz')))
            crop_inhale_mask[crop_inhale_mask > 1] = 1
            crop_exhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'exhale_mask.nii.gz')))
            crop_exhale_mask[crop_exhale_mask > 1] = 1
            print(Dice(inhale_mask, exhale_mask),Dice(inhale_mask, rigid_mask),Dice(crop_inhale_mask, crop_rigid_mask),Dice(crop_inhale_mask, crop_exhale_mask), end=',')
            if os.path.exists(os.path.join(case_folder, 'inhale_nodule.nii.gz')):
                inhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'inhale_nodule.nii.gz')))
                exhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'exhale_nodule.nii.gz')))
                rigid_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, 'rigid_nodule.nii.gz')))
                print(dist.Excute(inhale_nodule, exhale_nodule)[2], dist.Excute(inhale_nodule, rigid_nodule)[2], end=',')
                if os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'inhale_nodule.nii.gz')):
                    crop_rigid_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'rigid_nodule.nii.gz')))
                    crop_inhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'inhale_nodule.nii.gz')))
                    crop_exhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData', case, 'exhale_nodule.nii.gz')))
                    print(dist.Excute(crop_inhale_nodule, crop_rigid_nodule)[2], dist.Excute(crop_inhale_nodule, crop_exhale_nodule)[2], end=',')
        except Exception as e:
            print(e, end=',')
        print()


def TestPointMetric():
    a = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    b = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    Cdist = ComputePointMoving()
    a_center, b_center, cdist, acc = Cdist.Excute(a, b, return_center=True)
    print('a_center:{}, b_center:{}, cdist:{}, acc:{}'.format(a_center, b_center, cdist, acc))

    a = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    b = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    Cdist = ComputePointMoving()
    a_center, b_center, cdist, acc = Cdist.Excute(a, b, return_center=True)
    print('a_center:{}, b_center:{}, cdist:{}, acc:{}'.format(a_center, b_center, cdist, acc))

    a = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    b = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 1]]])
    Cdist = ComputePointMoving()
    a_center, b_center, cdist, acc = Cdist.Excute(a, b, return_center=True)
    print('a_center:{}, b_center:{}, cdist:{}, acc:{}'.format(a_center, b_center, cdist, acc))


def ComputeVolume(suffix, fixed_key, case_list=[]):
    print(suffix)
    data_root = '/data/data1/zyh/Data/CTLung/CropData'
    if not fixed_key == 'exhale':
        model_suffix = suffix
        suffix = '{}_{}'.format(suffix, fixed_key)
    else:
        model_suffix = suffix

    percent_dict = {}
    if len(case_list) == 0:
        case_list = os.listdir(os.path.join(data_root))
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        sleep(0.1)
        moving_mask_folder = os.path.join(data_root, case, '{}_mask.nii.gz'.format('inhale'))
        fixed_mask_folder = os.path.join(data_root, case, '{}_mask.nii.gz'.format(fixed_key))
        deform_mask_folder = os.path.join(data_root, case, 'lung_{}.nii.gz'.format(model_suffix))

        if not os.path.exists(deform_mask_folder): continue

        moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask_folder))
        moving_mask[moving_mask > 1] = 1
        ########### fixed mask ###########
        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_folder))
        fixed_mask[fixed_mask > 1] = 1
        ########### predict mask ###########
        deform_mask = sitk.GetArrayFromImage(sitk.ReadImage(deform_mask_folder))
        deform_mask = np.around(deform_mask)
        deform_mask = median_filter(deform_mask, size=3).astype(np.int32)
        deform_mask[deform_mask > 1] = 1

        ininstal_percent = (np.sum(moving_mask) - np.sum(fixed_mask)) / np.sum(moving_mask)
        predict_percent = (np.sum(moving_mask) - np.sum(deform_mask)) / np.sum(moving_mask)
        percent_dict[case] = [ininstal_percent, predict_percent]

        pbar.update()
    df = pd.DataFrame.from_dict(percent_dict).T
    df.to_csv(r'/data/data1/zyh/Data/CTLung/CropData/test_volume_compare.csv')
    pbar.close()



if __name__ == '__main__':
    csv_path = r'/data/data1/zyh/Data/CTLung/CropData/test.csv'
    case_df = pd.read_csv(str(csv_path), index_col=0).squeeze()
    case_list = case_df.index.tolist()
    # case_list = ['{}.nii.gz'.format(case) for case in case_list]
    is_image = False
    # ComputeElastix(is_image=is_image, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeInEx(is_image=is_image, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='0927_Only3DeformVTNSingle', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0927_OnlyDeformUNet', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0927_Only3DeformUNetSingle', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0929_UNetComplex', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0929_UNetComplexNorm', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0929_UNetNorm', is_image=is_image, case_list=case_list)
    # # ComputeTransform(suffix='1010_CGAN', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1014_CGAN_image', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1014_CGAN_2D', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1014_CGAN_2D_pretrain', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1014_CGAN_2D_pretrain_cdist', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1021_2D_LungMask', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0929_UNet', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='0927_OnlyDeformVTN', is_image=is_image, case_list=case_list)
    # ComputeTransform(suffix='1110_CGAN_2D', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1110_CGAN_2D_mask', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1110_CGAN_2D', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1114_CGAN_2D_pretrained', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1110_CGAN_field', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1031_batch2', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1031_CGAN_2D', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1110_CGAN_2D_5E', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1110_CGAN_image', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransform(suffix='1110_CGAN_2D_mask', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1117_CGAN_2D_mask', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransformImage(suffix='1117_CGAN_2D_mask', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1117', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransformImage(suffix='1117', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1117_box', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransformImage(suffix='1117_box', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)

    # ComputeTransform(suffix='1118_CGAN_2D', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransformImage(suffix='1118_CGAN_2D', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1118_CGAN_2D_pretrained', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransformImage(suffix='1118_CGAN_2D_pretrained', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransform(suffix='1122', is_image=True, case_list=case_list, fixed_key='rigid', write_csv=True)
    # ComputeTransformImage(suffix='1122', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # ComputeTransformImage(suffix='1110_CGAN_2D_mask', is_image=False, case_list=case_list, fixed_key='rigid',
    #                       write_csv=False)
    # ComputeTransformImage(suffix='1110_CGAN_2D_mask', is_image=False, case_list=case_list, fixed_key='rigid',
    #                       write_csv=False)
    ComputeTransformImage(suffix='1110_CGAN_2D_5E', is_image=False, case_list=case_list, fixed_key='rigid',
                          write_csv=False)
    ComputeTransformImage(suffix='1110_CGAN_image', is_image=False, case_list=case_list, fixed_key='rigid', write_csv=False)
    # CheckData()
    # SplitMetric()
    # ComputeVolume(suffix='1110_CGAN_2D_mask', fixed_key='rigid', case_list=case_list)

    # ComputeTransform(suffix='0929_UNetComplex', fixed_key='rigid', is_image=is_image, case_list=case_list, write_csv=False)
    # ComputeTransform(suffix='1014_CGAN_2D_pretrain', fixed_key='rigid', is_image=is_image, case_list=case_list, write_csv=False)

    # moving_distance = ComputePointMovingByImage()
    # deform_mask_image = sitk.GetImageFromArray(np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32))
    # fixed_mask_image = sitk.GetImageFromArray(np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.int32))
    # print(moving_distance.GetDice(deform_mask_image, fixed_mask_image),
    #       Dice(np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32), np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.int32)))


