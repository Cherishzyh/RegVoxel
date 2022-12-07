import math
import os
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import median_filter
import torch

from T4T.Utility.Data import Dataset
from Dataset.Augmentation import DataAugmentor3D
from Utils.Transform import AffineTransform, SpatialTransformer


def ImageTransform(moving_image, field, mask=False, affine=True, theta=None):
    warpper = SpatialTransformer(moving_image.shape[-3:])
    if affine and theta:
        warp_image = AffineTransform(moving_image, theta)
    else:
        warp_image = moving_image
    wrap_image = warpper(warp_image, field, mask=mask)
    if mask:
        wrap_image = torch.round(wrap_image).cpu().numpy()
        wrap_image = median_filter(wrap_image, size=3)
    return wrap_image


class NifitDataSet(Dataset):

    def __init__(self, data_path,
                 csv_path=r'',
                 transforms={},
                 shuffle=True,
                 is_min=True,
                 moving_key='inhale',
                 fixed_key='exhale',
                 ):

        # Init membership variables
        self.data_path = data_path
        self.moving_key = moving_key
        self.fixed_key = fixed_key
        self.augment = DataAugmentor3D()

        if csv_path:
            case_df = pd.read_csv(str(csv_path), index_col=0).squeeze()
            self.case_list = case_df.index.tolist()
            # self.case_list = ['{}.nii.gz'.format(case) for case in self.case_list]
        else:
            self.case_list = os.listdir(self.data_path)
        if shuffle:
            random.shuffle(self.case_list)

        self.transforms = transforms
        self.is_min = is_min
        self.shuffle_labels = shuffle

    def _Normalization(self, image, min, max):
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

    def _LoadImage(self, data_path, is_roi=False):
        image = sitk.ReadImage(data_path)
        if is_roi:
            image_arr = sitk.GetArrayFromImage(image)
            image_arr[image_arr >= 1] = 1
        else:
            image = self._Normalization(image, 0, 1)  # set intensity 0-1
            image_arr = abs(sitk.GetArrayFromImage(image))
        return image_arr

    def _ImageTransform(self, moving_image, field, mask=False):
        warpper = SpatialTransformer(moving_image.shape[-3:])
        wrap_image = warpper(moving_image[np.newaxis], field[np.newaxis], mask=mask)
        return wrap_image[0]

    def __getitem__(self, index):
        case = self.case_list[index]
        moving_path = os.path.join(self.data_path, case, '{}.nii.gz'.format(self.moving_key))
        moving_mask_path = os.path.join(self.data_path, case, '{}_mask.nii.gz'.format(self.moving_key))
        moving_nodule_path = os.path.join(self.data_path, case, '{}_nodule.nii.gz'.format(self.moving_key))
        fixed_mask_path = os.path.join(self.data_path, case, '{}_mask.nii.gz'.format(self.fixed_key))
        field_path = os.path.join(self.data_path, case, 'deform_flow.nii.gz')

        moving_arr = self._LoadImage(moving_path)
        field_arr = sitk.GetArrayFromImage(sitk.ReadImage(field_path))
        if self.is_min: field_arr = field_arr / 128

        # Compute volume

        moving_mask_arr = self._LoadImage(moving_mask_path, is_roi=True)
        fixed_mask_arr = self._LoadImage(fixed_mask_path, is_roi=True)
        volume = (np.sum(moving_mask_arr) - np.sum(fixed_mask_arr)) / np.sum(moving_mask_arr)
        # Load nodules if exist
        moving_nodule_arr = sitk.GetArrayFromImage(sitk.ReadImage(moving_nodule_path)) \
            if os.path.exists(moving_nodule_path) else np.zeros_like(moving_mask_arr)

        if self.transforms:
            moving_arr, field_arr, volume, [moving_mask_arr, moving_nodule_arr] = self.augment.Execute(moving_arr,
                                                                                                       field_arr,
                                                                                                       volume,
                                                                                                       source_mask=[moving_mask_arr, moving_nodule_arr],
                                                                                                       parameter_dict=self.transforms)


        moving_tensor = torch.from_numpy(moving_arr.transpose(2, 1, 0)[np.newaxis]).to(torch.float32)
        moving_mask_tensor = torch.from_numpy(moving_mask_arr.transpose(2, 1, 0)[np.newaxis]).to(torch.float32)
        moving_nodule_tensor = torch.from_numpy(moving_nodule_arr.transpose(2, 1, 0)[np.newaxis]).to(torch.float32)
        field_tensor = torch.from_numpy(field_arr.transpose(3, 2, 1, 0)).to(torch.float32)
        fixed_tensor = self._ImageTransform(moving_tensor, field_tensor)
        fixed_mask_tensor = self._ImageTransform(moving_mask_tensor, field_tensor, True)
        fixed_nodule_tensor = self._ImageTransform(moving_nodule_tensor, field_tensor, True)

        return moving_tensor, \
               fixed_tensor, \
               field_tensor, \
               torch.fill_(torch.zeros_like(moving_tensor), volume), \
               moving_mask_tensor, \
               fixed_mask_tensor, \
               moving_nodule_tensor, \
               fixed_nodule_tensor

    def __len__(self):
        return len(self.case_list)


class NifitDataSetTest():
    def __init__(self, data_path,
                 csv_path=r'',
                 transforms={},
                 shuffle=True,
                 is_min=True,
                 moving_key='inhale',
                 fixed_key='exhale'
                 ):
        self.data_path = data_path
        self.moving_key = moving_key
        self.fixed_key = fixed_key
        self.augment = DataAugmentor3D()
        if csv_path:
            case_df = pd.read_csv(str(csv_path), index_col=0).squeeze()
            self.case_list = sorted(case_df.index.tolist())
        else:
            self.case_list = sorted(os.listdir(self.data_path))
        if shuffle:
            random.shuffle(self.case_list)
        self.is_min = is_min
        self.transforms = transforms
        self.shuffle_labels = shuffle

    def _Normalization(self, image, min, max):
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

    def _ImageTransform(self, moving_image, field, mask=False):
        warpper = SpatialTransformer(moving_image.shape[-3:])
        wrap_image = warpper(moving_image, field, mask=mask)
        return wrap_image

    def getitem(self, index):
        # case = self.case_list[index]

        # moving
        # moving_image = sitk.ReadImage(os.path.join(self.data_path, case, '{}.nii.gz'.format(self.moving_key)))
        # moving_image = self._Normalization(moving_image, 0, 1)  # set intensity 0-1
        # moving_arr = abs(sitk.GetArrayFromImage(moving_image))
        # fixed_image = sitk.ReadImage(os.path.join(self.data_path, case, '{}.nii.gz'.format(self.fixed_key)))
        # fixed_image = self._Normalization(fixed_image, 0, 1)  # set intensity 0-1
        #
        # field_image = sitk.ReadImage(os.path.join(self.data_path, case, 'deform_flow.nii.gz'))
        # field_arr = sitk.GetArrayFromImage(field_image)
        # if self.is_min: field_arr = field_arr / 128
        #
        # # Compute volume
        # moving_mask = sitk.ReadImage(os.path.join(self.data_path, case, '{}_mask.nii.gz'.format(self.moving_key)))
        # moving_mask_arr = sitk.GetArrayFromImage(moving_mask)
        # moving_mask_arr = moving_mask_arr
        # fixed_mask = sitk.ReadImage(os.path.join(self.data_path, case, '{}_mask.nii.gz'.format(self.fixed_key)))
        # fixed_mask_arr = sitk.GetArrayFromImage(fixed_mask)
        # volume = (np.count_nonzero(moving_mask_arr) - np.count_nonzero(fixed_mask_arr)) / np.count_nonzero(moving_mask_arr)
        #
        # # Load nodules if exist
        # moving_nodule_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path, case, '{}_nodule.nii.gz'.format(self.moving_key)))) \
        #     if os.path.exists(os.path.join(self.data_path, case, '{}_nodule.nii.gz'.format(self.moving_key))) else np.zeros_like(moving_mask_arr)

        moving_arr = np.zeros((5, 5, 5))
        moving_arr[2, 2, 2] = 8
        field_arr = np.zeros((5, 5, 5, 3))
        field_arr[2, 2, 2, 0] = 0
        field_arr[2, 2, 2, 1] = 0
        field_arr[2, 2, 2, 2] = 1
        volume = 0.3
        moving_mask_arr = np.zeros_like(moving_arr)
        moving_nodule_arr = np.zeros_like(moving_arr)

        if self.transforms:
            moving_arr, field_arr, volume, [moving_mask_arr, moving_nodule_arr] = self.augment.Execute(moving_arr, field_arr, volume, source_mask=[moving_mask_arr, moving_nodule_arr], parameter_dict=self.transforms)

        moving_tensor = torch.from_numpy(moving_arr.transpose(2, 1, 0)[np.newaxis, np.newaxis]).to(torch.float32)
        moving_mask_tensor = torch.from_numpy(moving_mask_arr.transpose(2, 1, 0)[np.newaxis, np.newaxis]).to(torch.float32)
        moving_nodule_tensor = torch.from_numpy(moving_nodule_arr.transpose(2, 1, 0)[np.newaxis, np.newaxis]).to(torch.float32)
        field_tensor = torch.from_numpy(field_arr.transpose(3, 2, 1, 0)[np.newaxis]).to(torch.float32)

        fixed_tensor = self._ImageTransform(moving_tensor, field_tensor)
        fixed_mask_tensor = self._ImageTransform(moving_mask_tensor, field_tensor, True)
        fixed_nodule_tensor = self._ImageTransform(moving_nodule_tensor, field_tensor, True)

        return moving_tensor, \
               fixed_tensor, \
               field_tensor, \
               torch.fill_(torch.zeros_like(moving_tensor), volume), \
               [field_image, moving_image, fixed_image], \
               moving_nodule_tensor, \
               fixed_nodule_tensor, \
               moving_mask_tensor, \
               fixed_mask_tensor



if __name__ == '__main__':
    # from config import configs

    random_3d_augment = {
        # 'zoom': [1, 1.25],  # 缩放？
        # 'horizontal_flip': True,  # 翻转
        # 'volume_percent': [0.05, 0.5],
        # 'rotate_angle': [-90, 90],
        # 'rotate_axis': [1, 0, 0]
    }

    data_generator = NifitDataSetTest(data_path=r'/data/data1/zyh/Data/CTLung/CropData',
                                      csv_path=r'/data/data1/zyh/Data/CTLung/CropData/train.csv',
                                      transforms=random_3d_augment,
                                      shuffle=False,
                                      is_min=True,
                                      fixed_key='rigid',
                                      moving_key='inhale')

    case = '20191112_luo_hua_shi'
    for epoch in range(0, 1):
        print(case)
        index = data_generator.case_list.index(case)
        moving, fixed, field, _, [ref_field, ref_moving, ref_fixed], \
        moving_nodule, fixed_nodule, moving_mask, fixed_mask = data_generator.getitem(index)

        moving_image = sitk.GetImageFromArray(moving.numpy().squeeze().transpose((2, 1, 0)))
        moving_image.CopyInformation(ref_moving)
        moving_mask_image = sitk.GetImageFromArray(moving_mask.numpy().squeeze().transpose((2, 1, 0)))
        moving_mask_image.CopyInformation(ref_moving)
        moving_nodule_image = sitk.GetImageFromArray(moving_nodule.numpy().squeeze().transpose((2, 1, 0)))
        moving_nodule_image.CopyInformation(ref_moving)

        fixed_image = sitk.GetImageFromArray(fixed.numpy().squeeze().transpose((2, 1, 0)))
        fixed_image.CopyInformation(ref_fixed)
        fixed_mask_image = sitk.GetImageFromArray(fixed_mask.numpy().squeeze().transpose((2, 1, 0)))
        fixed_mask_image.CopyInformation(ref_fixed)
        fixed_nodule_image = sitk.GetImageFromArray(fixed_nodule.numpy().squeeze().transpose((2, 1, 0)))
        fixed_nodule_image.CopyInformation(ref_fixed)
        field_image = sitk.GetImageFromArray(field.numpy().squeeze().transpose((3, 2, 1, 0)), isVector=True)
        field_image.CopyInformation(ref_field)

        # sitk.WriteImage(moving_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'moving_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(moving_mask_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'moving_mask_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(moving_nodule_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'moving_nodule_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(fixed_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'fixed_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(fixed_mask_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'fixed_mask_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(fixed_nodule_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'fixed_nodule_{:.3f}.nii.gz'.format(float(torch.unique(_)))))
        # sitk.WriteImage(field_image,
        #                 os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi', 'deform_flow_{:.3f}.nii.gz'.format(float(torch.unique(_)))))

        sitk.WriteImage(moving_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'moving_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(moving_mask_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'moving_mask_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(moving_nodule_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'moving_nodule_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(fixed_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'fixed_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(fixed_mask_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'fixed_mask_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(fixed_nodule_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'fixed_nodule_rotate_{}.nii.gz'.format(epoch)))
        sitk.WriteImage(field_image,
                        os.path.join(r'/data/data1/zyh/Data/CTLung/Augmentation/20191112_luo_hua_shi',
                                     'deform_flow_rotate_{}.nii.gz'.format(epoch)))

