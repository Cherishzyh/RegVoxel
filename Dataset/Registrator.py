import os
import shutil
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt


def CreateFolder(root):
    if not os.path.exists(root): os.makedirs(root)
    if not os.path.exists(os.path.join(root, 'inhale')): os.makedirs(os.path.join(root, 'inhale'))
    if not os.path.exists(os.path.join(root, 'exhale')): os.makedirs(os.path.join(root, 'exhale'))
    if not os.path.exists(os.path.join(root, 'inhale', 'image')): os.makedirs(os.path.join(root, 'inhale', 'image'))
    if not os.path.exists(os.path.join(root, 'inhale', 'mask')): os.makedirs(os.path.join(root, 'inhale', 'mask'))
    if not os.path.exists(os.path.join(root, 'exhale', 'image')): os.makedirs(os.path.join(root, 'exhale', 'image'))
    if not os.path.exists(os.path.join(root, 'exhale', 'mask')): os.makedirs(os.path.join(root, 'exhale', 'mask'))


class Registrator():
    def __init__(self, fixed_image='', moving_image=''):
        self.__fixed_image = None
        self.__moving_image = None
        self.SetFixedImage(fixed_image)
        self.SetMovingImage(moving_image)

    def SetFixedImage(self, fixed_image):
        if isinstance(fixed_image, Path):
            fixed_image = str(fixed_image)

        if isinstance(fixed_image, str) and fixed_image:
            self.__fixed_image = sitk.ReadImage(fixed_image)
        elif isinstance(fixed_image, sitk.Image):
            self.__fixed_image = fixed_image
    def GetFixedImage(self):
        return self.__fixed_image
    fixed_image = property(GetFixedImage, SetFixedImage)

    def SetMovingImage(self, moving_image):
        if isinstance(moving_image, Path):
            moving_image = str(moving_image)

        if isinstance(moving_image, str) and moving_image:
            self.__moving_image = sitk.ReadImage(moving_image)
        elif isinstance(moving_image, sitk.Image):
            self.__moving_image = moving_image
    def GetMovingImage(self):
        return self.__moving_image
    moving_image = property(GetMovingImage, SetMovingImage)


    def GenerateStorePath(self, moving_image_path):
        moving_image_path = str(moving_image_path)
        if moving_image_path.endswith('.nii.gz'):
            return moving_image_path[:-7] + '_Reg.nii.gz'
        else:
            file_path, ext = os.path.splitext(moving_image_path)
        return file_path + '_Reg' + ext

    def RegistrateBySpacing(self, method=sitk.sitkBSpline, dtype=sitk.sitkFloat32, store_path=''):
        min_filter = sitk.MinimumMaximumImageFilter()
        min_filter.Execute(self.__moving_image)
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(self.__fixed_image.GetSpacing())
        resample_filter.SetOutputOrigin(self.__fixed_image.GetOrigin())
        resample_filter.SetOutputDirection(self.__fixed_image.GetDirection())
        resample_filter.SetSize(self.__fixed_image.GetSize())
        resample_filter.SetTransform(sitk.AffineTransform(3))
        resample_filter.SetInterpolator(method)
        resample_filter.SetDefaultPixelValue(min_filter.GetMinimum())
        resample_filter.SetOutputPixelType(dtype)
        output = resample_filter.Execute(self.__moving_image)

        if store_path:
            sitk.WriteImage(output, store_path)

        return output



if __name__ == '__main__':
    root_path = r'/data/data1/zyh/Data/CTLung/RawData/anotatedNoduleImage'

    save_root = r'/data/data1/zyh/Data/CTLung/Registrator'
    CreateFolder(save_root)
    if not os.path.exists(os.path.join(save_root, 'exhale', 'nodule')): os.makedirs(os.path.join(save_root, 'exhale', 'nodule'))
    if not os.path.exists(os.path.join(save_root, 'inhale', 'nodule')): os.makedirs(os.path.join(save_root, 'inhale', 'nodule'))
    for index, case in enumerate(sorted(os.listdir(root_path))):
        inhale_folder = os.path.join(root_path, case, 'inhale')
        exhale_folder = os.path.join(root_path, case, 'exhale')
        # if os.path.exists(os.path.join(save_root, 'inhale', 'image', case)): continue
        print('**************** {} / {} ****************'.format(index+1, len(os.listdir(root_path))))
        in_image_path = os.path.join(inhale_folder, 'image.nii.gz')
        in_mask_path = os.path.join(inhale_folder, 'lobe.nii.gz')
        ex_image_path = os.path.join(exhale_folder, 'image.nii.gz')
        ex_mask_path = os.path.join(exhale_folder, 'lobe.nii.gz')


        registrator = Registrator(in_image_path, ex_image_path)
        try:
            registrator.RegistrateBySpacing(store_path=os.path.join(save_root, 'exhale', 'image', '{}.nii.gz'.format(case)))
        except: print('Align Exhale Image Failed')

        registrator = Registrator(in_image_path, in_mask_path)
        try:
            registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                            dtype=sitk.sitkInt32,
                                            store_path=os.path.join(save_root, 'inhale', 'mask', '{}.nii.gz'.format(case)))
        except: print('Align Inhale Mask Failed')

        registrator = Registrator(in_image_path, ex_mask_path)
        try:
            registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                            dtype=sitk.sitkInt32,
                                            store_path=os.path.join(save_root, 'exhale', 'mask', '{}.nii.gz'.format(case)))
        except: print('Align Exhale Mask Failed')

        shutil.copyfile(in_image_path, os.path.join(save_root, 'inhale', 'image', '{}.nii.gz'.format(case)))

    # for root, dirs, files in os.walk(root_path):
    #     if len(files) == 2 and ('exhale.nii.gz' in files) and ('inhale.nii.gz' in files):
    #         print(os.path.basename(root))
    #         new_name = os.path.basename(root).replace(' ', '_')
    #         ref_image = os.path.join(r'/data/data1/zyh/Data/CTLung/Registrator/inhale/image', '{}.nii.gz'.format(new_name))
    #         if not os.path.exists(ref_image):
    #             print('there is no {}'.format(os.path.basename(root)))
    #             continue
    #         inhale_folder = os.path.join(root, 'inhale.nii.gz')
    #         exhale_folder = os.path.join(root, 'exhale.nii.gz')
    #
    #         registrator = Registrator(ref_image, inhale_folder)
    #         try:
    #             registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
    #                                             dtype=sitk.sitkInt32,
    #                                             store_path=os.path.join(save_root, 'inhale', 'nodule', '{}.nii.gz'.format(new_name)))
    #         except: print('Align inhale Image Failed')
    #
    #         registrator = Registrator(ref_image, exhale_folder)
    #         try:
    #             registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
    #                                             dtype=sitk.sitkInt32,
    #                                             store_path=os.path.join(save_root, 'exhale', 'nodule', '{}.nii.gz'.format(new_name)))
    #         except: print('Align Exhale Image Failed')









