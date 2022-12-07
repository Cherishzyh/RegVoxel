import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch

from Networks.Mainnet import SimpleRegNet
from Dataset.DataSet import NifitDataSetTest, ImageTransform

from pathlib import Path


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


class DataTools():
    def __init__(self):
        super().__init__()

    def Tensor2Numpy(self, tensor):
        return tensor.cpu().detach().numpy().squeeze()

    def Numpy2Image(self, array, ref=None, isVector=False, save_folder=r''):
        if np.ndim(array) == 3:
            array = np.transpose(array, (2, 1, 0))
        elif np.ndim(array) == 4:
            array = np.transpose(array, (3, 2, 1, 0))
        else:
            raise ValueError
        if isVector:
            image = sitk.GetImageFromArray(array, isVector=isVector)
        else:
            image = sitk.GetImageFromArray(array)
        if ref:
            image.CopyInformation(ref)
        if save_folder:
            sitk.WriteImage(image, save_folder)
        return image




def Test(suffix='0927_OnlyDeformUNet', model='VTN', weights_path=None, n_deform=1, fixed_key='exhale'):
    ##########
    # Prepare
    ##########
    # Get train data
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '6'
    device = 'cuda:6'
    is_min = True
    is_affine = False
    if 'box' in suffix: is_mask = True
    else: is_mask = False
    model_folder = '/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_{}'.format(suffix)
    if not weights_path:
        weights_list = [weight for weight in os.listdir(model_folder) if weight.endswith('.pt')]
        weights_list = sorted(weights_list, key=lambda x: os.path.getctime(os.path.join(model_folder, x)))
        weights_path = weights_list[-1]
    print(weights_path)
    data_tool = DataTools()
    data_folder = r'/data/data1/zyh/Data/CTLung/CropData'
    save_folder = r'/data/data1/zyh/Data/CTLung/CropData'

    data_generator = NifitDataSetTest(data_path=data_folder,
                                      csv_path=r'/data/data1/zyh/Data/CTLung/CropData/test.csv',
                                      transforms=None,
                                      shuffle=False,
                                      is_min=is_min,
                                      fixed_key=fixed_key,
                                      moving_key='inhale')

    # ModelSetting
    simple_regnet = SimpleRegNet(shape=[256, 256, 256],
                                 n_deform=n_deform,
                                 n_recursive=1,
                                 in_channels=2,
                                 affine_loss_dict={},
                                 image_loss_dict={},
                                 flow_loss_dict={},
                                 deform_net=model,
                                 is_affine=is_affine,
                                 is_min=is_min,
                                 is_box=is_mask).to(device)

    simple_regnet.load_state_dict(torch.load(os.path.join(model_folder, weights_path), map_location=device))
    simple_regnet.eval()
    pbar = tqdm(total=len(data_generator.case_list), ncols=80)
    for index, case in enumerate(sorted(data_generator.case_list)):
        moving, fixed, field, volume, [ref_field, ref_moving, ref_fixed], \
        moving_nodule, fixed_nodule, moving_mask, fixed_mask = data_generator.getitem(index)
        with torch.no_grad():
            moving = moving.to(torch.float32).to(device)  # (batch, 1, *)
            fixed = fixed.to(torch.float32).to(device)
            field = field.to(torch.float32).to(device)
            volume = volume.to(torch.float32).to(device)
            moving_mask = moving_mask.to(torch.float32).to(device)

            if is_affine:
                image_pred, field_pred, theta, _ = simple_regnet(moving, fixed, volume, moving_mask, field)
            else:
                image_pred, field_pred, _ = simple_regnet(moving, fixed, volume, moving_mask, field)

            image_pred = data_tool.Tensor2Numpy(image_pred)
            data_tool.Numpy2Image(image_pred, ref_moving, save_folder=os.path.join(save_folder, case, 'exhale_{}.nii.gz'.format(suffix)))

            deform_lung = ImageTransform(moving_mask, field_pred, True, False)
            deform_lung = deform_lung.squeeze()
            data_tool.Numpy2Image(deform_lung, ref_moving, save_folder=os.path.join(save_folder, case, 'lung_{}.nii.gz'.format(suffix)))

            if torch.sum(moving_nodule) > 0:
                deform_nodule = ImageTransform(moving_nodule.to(device), field_pred, True, False)
                deform_nodule = deform_nodule.squeeze()
                data_tool.Numpy2Image(deform_nodule, ref_moving,
                                      save_folder=os.path.join(save_folder, case, 'nodule_{}.nii.gz'.format(suffix)))
            field_pred = data_tool.Tensor2Numpy(field_pred)
            if is_min:
                field_pred = field_pred * 128
            data_tool.Numpy2Image(field_pred, ref_field, isVector=True,
                                  save_folder=os.path.join(save_folder, case, 'field_{}.nii.gz'.format(suffix)))
            #
            # if is_affine:
            #     theta = data_tool.Tensor2Numpy(theta)
            #     np.save(os.path.join(save_folder, 'param_{}'.format(suffix), '{}.npy'.format(case.split('.nii.gz')[0])), theta)
            #     data_tool.Numpy2Image(theta, ref_field, isVector=True,
            #                           save_folder=os.path.join(save_folder, 'param_{}'.format(suffix), case))
        pbar.update()
    pbar.close()


def TestByVolumePercent(suffix='0927_OnlyDeformUNet', model='VTN', weights_path=None, n_deform=1, fixed_key='exhale'):
    ##########
    # Prepare
    ##########
    # Get train data
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '6'
    device = 'cuda:4'
    is_min = True
    is_affine = False
    if 'mask' in suffix: is_mask = True
    else: is_mask = False
    model_folder = '/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_{}'.format(suffix)
    if not weights_path:
        weights_list = [weight for weight in os.listdir(model_folder) if weight.endswith('.pt')]
        weights_list = sorted(weights_list, key=lambda x: os.path.getctime(os.path.join(model_folder, x)))
        weights_path = weights_list[-1]
    print(weights_path)
    data_tool = DataTools()
    data_folder = r'/data/data1/zyh/Data/CTLung/CropData'
    save_folder = r'/data/data1/zyh/Data/CTLung/CropData'

    data_generator = NifitDataSetTest(data_path=data_folder,
                                      csv_path=r'/data/data1/zyh/Data/CTLung/CropData/test.csv',
                                      transforms=None,
                                      shuffle=False,
                                      is_min=is_min,
                                      fixed_key=fixed_key,
                                      moving_key='inhale')

    # ModelSetting
    simple_regnet = SimpleRegNet(shape=[256, 256, 256],
                                 n_deform=n_deform,
                                 n_recursive=1,
                                 in_channels=2,
                                 affine_loss_dict={},
                                 image_loss_dict={},
                                 flow_loss_dict={},
                                 deform_net=model,
                                 is_affine=is_affine,
                                 is_min=is_min,
                                 is_box=is_mask).to(device)

    simple_regnet.load_state_dict(torch.load(os.path.join(model_folder, weights_path), map_location=device))
    simple_regnet.eval()

    for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        case = '20210709_guo_tie'
        index = data_generator.case_list.index(case)
        moving, fixed, field, _, [ref_field, ref_moving, ref_fixed], \
        moving_nodule, fixed_nodule, moving_mask, fixed_mask = data_generator.getitem(index)
        volume = torch.from_numpy(np.resize(i, moving.shape).astype(np.float32))

        with torch.no_grad():
            moving = moving.to(torch.float32).to(device)  # (batch, 1, *)
            volume = volume.to(torch.float32).to(device)
            moving_mask = moving_mask.to(torch.float32).to(device)

            if is_affine:
                image_pred, field_pred, theta, _ = simple_regnet(moving, _, volume, _, _)
            else:
                image_pred, field_pred, _ = simple_regnet(moving, _, volume, _, _)

            image_pred = data_tool.Tensor2Numpy(image_pred)
            data_tool.Numpy2Image(image_pred, ref_moving, save_folder=os.path.join(save_folder, case, 'exhale_{}_{:2f}.nii.gz'.format(suffix, i)))

            deform_lung = ImageTransform(moving_mask, field_pred, True, False)
            deform_lung = deform_lung.squeeze()
            data_tool.Numpy2Image(deform_lung, ref_moving, save_folder=os.path.join(save_folder, case, 'lung_{}_{:2f}.nii.gz'.format(suffix, i)))

            if torch.sum(moving_nodule) > 0:
                deform_nodule = ImageTransform(moving_nodule.to(device), field_pred, True, False)
                deform_nodule = deform_nodule.squeeze()
                data_tool.Numpy2Image(deform_nodule, ref_moving,
                                      save_folder=os.path.join(save_folder, case, 'nodule_{}_{:2f}.nii.gz'.format(suffix, i)))


if __name__ == '__main__':
    # Test(suffix='0927_OnlyDeformUNet', model='UNet', weights_path='27-0.065810.pt')
    # Test(suffix='0927_Only3DeformUNetSingle', model='UNet', weights_path='24-0.194120.pt', n_deform=3)
    # Test(suffix='0927_Only3DeformVTNSingle', model='VTN', weights_path='42-0.201760.pt', n_deform=3)
    # Test(suffix='0927_OnlyDeformVTN', model='VTN', weights_path='35-0.069476.pt')
    # Test(suffix='0929_UNetComplex', model='UNetComplex', weights_path='48-0.065568.pt')
    # Test(suffix='0929_UNetComplexNorm', model='UNetComplexNorm', weights_path='88-0.066116.pt')
    # Test(suffix='0929_UNetNorm', model='UNetNorm', weights_path='52-0.066923.pt')
    # Test(suffix='0929_UNet', model='UNet', weights_path='34-0.066614.pt')
    # Test(suffix='1110', model='UNetComplex', weights_path='73-0.011401.pt', fixed_key='rigid')
    # Test(suffix='1117', model='UNetComplex', weights_path='82-0.032404.pt', fixed_key='rigid')
    # Test(suffix='1117_box', model='UNetComplex', weights_path='58-0.020242.pt', fixed_key='rigid')
    # Test(suffix='1122', model='UNetComplex', weights_path='36-0.076059.pt', fixed_key='rigid')

    TestByVolumePercent(suffix='1117', model='UNetComplex', weights_path='82-0.032404.pt', fixed_key='rigid')
