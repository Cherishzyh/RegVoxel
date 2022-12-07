import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from copy import deepcopy

import sys
sys.path.append('/')

from Dataset.NewResample import Resampler
from MeDIT.SaveAndLoad import LoadImage

from Dataset.Dcm2Nii import Dcm2Nii
from Dataset.Registrator import Registrator


def CreateFolder(root):
    if not os.path.exists(root): os.makedirs(root)
    if not os.path.exists(os.path.join(root, 'inhale')): os.makedirs(os.path.join(root, 'inhale'))
    if not os.path.exists(os.path.join(root, 'exhale')): os.makedirs(os.path.join(root, 'exhale'))
    if not os.path.exists(os.path.join(root, 'inhale', 'image')): os.makedirs(os.path.join(root, 'inhale', 'image'))
    if not os.path.exists(os.path.join(root, 'inhale', 'mask')): os.makedirs(os.path.join(root, 'inhale', 'mask'))
    if not os.path.exists(os.path.join(root, 'exhale', 'image')): os.makedirs(os.path.join(root, 'exhale', 'image'))
    if not os.path.exists(os.path.join(root, 'exhale', 'mask')): os.makedirs(os.path.join(root, 'exhale', 'mask'))

def KeepLargest(mask):
    new_mask = np.zeros(mask.shape)
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    index = np.argmax(max_volume)
    new_mask[label_im == index + 1] = 1
    return new_mask

def KeepLargestTwo(mask):
    new_mask = np.zeros(mask.shape)
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    index = np.argmax(max_volume)
    new_mask[label_im == index + 1] = 1
    max_volume.remove(max_volume[index])
    # max_volume.remove(max_volume[np.argmax(max_volume)])
    second_index = np.argmax(max_volume)
    if second_index >= index: second_index += 1
    new_mask[label_im == second_index + 1] = 1
    return new_mask

def PostPrecess():
    case = '20211231 ruan a mu.nii.gz'
    inhale_mask = sitk.ReadImage(os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/exhale/mask', case))
    inhale_mask_arr = sitk.GetArrayFromImage(inhale_mask)
    new_inhale_arr = np.zeros_like(inhale_mask_arr)
    for c in np.unique(inhale_mask_arr):
        if c == 0: continue
        # if c == 1:
        #     temp_inhale = np.zeros_like(inhale_mask_arr)
        #     temp_inhale[inhale_mask_arr == c] = c
        #     temp_inhale = KeepLargestTwo(temp_inhale)
        #     new_inhale_arr = new_inhale_arr + temp_inhale*c
        else:
            temp_inhale = np.zeros_like(inhale_mask_arr)
            temp_inhale[inhale_mask_arr == c] = c
            temp_inhale = KeepLargest(temp_inhale)
            new_inhale_arr = new_inhale_arr + temp_inhale*c
    new_inhale_arr = sitk.GetImageFromArray(new_inhale_arr)
    new_inhale_arr.CopyInformation(inhale_mask)
    sitk.WriteImage(new_inhale_arr, os.path.join(r'/data/data1/zyh/Data/CTLung/ResampleDataAll1.35/exhale', case))

def StandardFileName():
    for root, dirs, files in os.walk(r'D:\Data\Crop\inhale_nodule'):
        if len(files) > 0:
            error_file_name = [file for file in files if ' ' in file]
            for file in error_file_name:
                if os.path.exists(os.path.join(root, file.replace(' ', '_'))): os.remove(os.path.join(root, file))
                else:
                    shutil.copyfile(os.path.join(root, file),
                                    os.path.join(root, file.replace(' ', '_')))
                    os.remove(os.path.join(root, file))
                print('successful change {} to {}'.format(file, file.replace(' ', '_')))

def CastImage(inputImage, dtype=sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(dtype)
    inputImage = castImageFilter.Execute(inputImage)
    return inputImage

def DataConvert(raw_folder):
    processor = Dcm2Nii(raw_folder=r'',
                        processed_folder=r'',
                        failed_folder=raw_folder, is_overwrite=False)
    for root, dirs, files in os.walk(raw_folder):
        dcm_file = [file for file in files if file.endswith('.dcm')]
        if len(dcm_file) > 0:
            try:
                processor.ConvertDicom2Nii(root)
            except Exception as e:
                print(root)

class HydraNiiPreprocess():
    def __init__(self, dcm_folder, nii_folder, hydra_code):
        super().__init__()
        self.dcm_folder = dcm_folder
        self.nii_folder = nii_folder
        self.hydra_code = hydra_code

    def CopyHydraNii(self):
        if os.path.exists(self.nii_folder):
            shutil.rmtree(self.nii_folder)
        os.makedirs(self.nii_folder)
        for root, dirs, files in os.walk(self.dcm_folder):
            if len(files) > 0:
                nii_files = [file for file in files if file.endswith('.nii')]
                if len(nii_files) > 0:
                    des_folder = os.path.join(self.nii_folder, root.split('/')[-2])
                    if not os.path.exists(des_folder): os.makedirs(des_folder)
                    [shutil.copyfile(os.path.join(root, nii_file),  os.path.join(des_folder, nii_file)) for nii_file in nii_files]

    def HydraRename(self):
        exchange_dict = {}
        for case in os.listdir(self.nii_folder):
            if os.path.isdir(os.path.join(self.nii_folder, case)):
                os.rename(os.path.join(self.nii_folder, case), os.path.join(self.nii_folder, 'Case_{}'.format(self.hydra_code)))
                exchange_dict[case] = 'Case_{}'.format(self.hydra_code)
                self.hydra_code += 1
        df = pd.DataFrame([exchange_dict]).T
        df.to_csv(os.path.join(self.nii_folder, 'exchange.csv'), header=False, encoding='gbk')

    def CleanHydra(self):
        def GetCaseNumber(file_list):
            remain_case = []
            number_list = sorted([file[:3] for file in file_list])
            for num in range(0, 10):
                num_list = sorted([file for file in number_list if file.startswith(str(num))])
                if len(num_list):
                    case_name = [file for file in file_list if file.startswith(num_list[0])][0]
                    remain_case.append(case_name)
            return remain_case
        for case in sorted(os.listdir(self.nii_folder)):
            if not os.path.isdir(os.path.join(self.nii_folder, case)): continue
            case_folder = os.path.join(self.nii_folder, case)
            remain_case = GetCaseNumber(sorted([file for file in os.listdir(case_folder) if ('1.0_x_1.0' in file or '0.6_x_0.6' in file)]))
            [os.remove(os.path.join(case_folder, file)) for file in os.listdir(case_folder) if file not in remain_case]

    def Rename2SegVolume(self):
        for case in sorted(os.listdir(self.nii_folder)):
            case_folder = os.path.join(self.nii_folder, case)
            if not os.path.isdir(case_folder): continue
            if (len(os.listdir(case_folder)) < 2) or (not os.path.isdir(case_folder)): continue
            elif len(os.listdir(case_folder)) == 2:
                if not os.path.exists(os.path.join(self.nii_folder, '{}_1'.format(case))):
                    os.makedirs(os.path.join(self.nii_folder, '{}_1'.format(case)))
                case_1, case_2 = os.listdir(case_folder)[0], os.listdir(case_folder)[1]
                shutil.move(os.path.join(case_folder, case_2),
                            os.path.join(self.nii_folder, '{}_1'.format(case), 'image.nii.gz'))
                os.rename(os.path.join(case_folder, case_1),
                          os.path.join(case_folder, 'image.nii.gz'))
            else:
                print(case)

    def SplitInEx(self):
        case_list = [case for case in os.listdir(self.nii_folder) if not case.endswith('_1')]
        df_dict = {'CaseName': [], 'Inhale': [], 'Exhale': [], 'Percent': []}
        for case in case_list:
            case1_folder = os.path.join(self.nii_folder, case)
            case2_folder = os.path.join(self.nii_folder, '{}_1'.format(case))
            inhale_folder = os.path.join(self.nii_folder, case, 'inhale')
            exhale_folder = os.path.join(self.nii_folder, case, 'exhale')
            if not os.path.exists(inhale_folder): os.makedirs(inhale_folder)
            if not os.path.exists(exhale_folder): os.makedirs(exhale_folder)
            mask1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.nii_folder, case, 'lobe.nii.gz')))
            mask1[mask1 >= 1] = 1
            mask2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.nii_folder, '{}_1'.format(case), 'lobe.nii.gz')))
            mask2[mask2 >= 1] = 1
            volume1 = np.sum(mask1)
            volume2 = np.sum(mask2)
            # 体积大是in， 体积小的是ex
            if volume1 > volume2:
                # mask1 2 inhale
                [shutil.move(os.path.join(case1_folder, file), os.path.join(inhale_folder, file)) for file in os.listdir(case1_folder) if os.path.isfile(os.path.join(case1_folder, file))]
                # mask2 2 exhale
                [shutil.move(os.path.join(case2_folder, file), os.path.join(exhale_folder, file)) for file in os.listdir(case2_folder) if os.path.isfile(os.path.join(case2_folder, file))]
            elif volume1 < volume2:
                # mask1 2 exhale
                [shutil.move(os.path.join(case1_folder, file), os.path.join(exhale_folder, file)) for file in os.listdir(case1_folder) if os.path.isfile(os.path.join(case1_folder, file))]
                # mask2 2 inhale
                [shutil.move(os.path.join(case2_folder, file), os.path.join(inhale_folder, file)) for file in os.listdir(case2_folder) if os.path.isfile(os.path.join(case2_folder, file))]
            else:
                print(case)
            if len(os.listdir(case2_folder)) == 0: shutil.rmtree(case2_folder)
            df_dict['CaseName'].append(case)
            df_dict['Inhale'].append(max(volume1, volume2))
            df_dict['Exhale'].append(min(volume1, volume2))
            df_dict['Percent'].append(abs(volume1 - volume2) / max(volume1, volume2))
        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(self.nii_folder, 'volume.csv'), index=False)

    def Run(self, copy=False):
        if copy: self.CopyHydraNii()
        self.HydraRename()
        self.CleanHydra()
        self.Rename2SegVolume()

def CopyNii(src_root, des_root):
    if os.path.exists(des_root): shutil.rmtree(des_root)
    os.makedirs(des_root)
    for root, dirs, files in os.walk(src_root):
        if len(files) > 0:
            nii_files = [file for file in files if file.endswith('.nii') or file.endswith('.nii.gz')]
            if len(nii_files) == 1:
                if nii_files[0].endswith('.nii.gz'):
                    des_folder = os.path.join(des_root, root.split('/')[-2], root.split('/')[-1])
                    case_name = root.split('/')[-2].replace(' ', '_')
                    case_name = '{}.nii.gz'.format(case_name)
                else:
                    des_folder = os.path.join(des_root, root.split('/')[-3], root.split('/')[-2])
                    case_name = root.split('/')[-3].replace(' ', '_')
                    case_name = '{}.nii.gz'.format(case_name)
                # print(des_folder)
                if not os.path.exists(des_folder): os.makedirs(des_folder)
                shutil.copyfile(os.path.join(root, nii_files[0]),  os.path.join(des_folder, case_name))

def Registrate(root_path, save_root, is_nodule=False):
    nodule_root = r'/data/data1/zyh/Data/CTLung/RawData/NoduleAnnotation1031/Nodule'
    CreateFolder(save_root)
    if is_nodule:
        if not os.path.exists(os.path.join(save_root, 'inhale', 'nodule')): os.makedirs(os.path.join(save_root, 'inhale', 'nodule'))
        if not os.path.exists(os.path.join(save_root, 'exhale', 'nodule')): os.makedirs(os.path.join(save_root, 'exhale', 'nodule'))
        for case in os.listdir(root_path):
            # inhale_folder = os.path.join(root_path, case, 'inhale.nii.gz')
            # exhale_folder = os.path.join(root_path, case, 'exhale.nii.gz')

            inhale_folder = os.path.join(root_path, case, 'inhale')
            inhale_folder = os.path.join(inhale_folder, os.listdir(inhale_folder)[0])
            exhale_folder = os.path.join(root_path, case, 'exhale')
            exhale_folder = os.path.join(exhale_folder, os.listdir(exhale_folder)[0])

            # ref_image = os.path.join(r'/data/data1/zyh/Data/CTLung/CNNData/moving', '{}.nii.gz'.format(case.replace(' ', '_')))
            # if not os.path.exists(ref_image):
            #     print(case)
            #     continue
            # registrator = Registrator(ref_image, inhale_folder)
            # try:
            #     registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
            #                                     dtype=sitk.sitkInt32,
            #                                     store_path=os.path.join(save_root, 'inhale', 'nodule', '{}.nii.gz'.format(case.replace(' ', '_'))))
            # except: print('Align inhale Image Failed')
            #
            # registrator = Registrator(ref_image, exhale_folder)
            # try:
            #     registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
            #                                     dtype=sitk.sitkInt32,
            #                                     store_path=os.path.join(save_root, 'exhale', 'nodule', '{}.nii.gz'.format(case.replace(' ', '_'))))
            # except: print('Align Exhale Image Failed')
            shutil.copyfile(inhale_folder, os.path.join(save_root, 'inhale', 'nodule', '{}.nii.gz'.format(case.replace(' ', '_'))))
            registrator = Registrator(inhale_folder, exhale_folder)
            try:
                registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                                dtype=sitk.sitkInt32,
                                                store_path=os.path.join(save_root, 'exhale', 'nodule', '{}.nii.gz'.format(case.replace(' ', '_'))))
            except: print('Align Exhale Image Failed')
    else:
        for index, case in enumerate(sorted(os.listdir(root_path))):
            # case = 'Case_12'
            if not os.path.isdir(os.path.join(root_path, case)): continue
            print('**************** {} / {} ****************'.format(index + 1, len(os.listdir(root_path))))
            inhale_folder = os.path.join(root_path, case, 'inhale')
            exhale_folder = os.path.join(root_path, case, 'exhale')
            in_image_path = os.path.join(inhale_folder, 'image.nii.gz')
            in_mask_path = os.path.join(inhale_folder, 'lobe.nii.gz')
            in_nodule_path = os.path.join(nodule_root, case, 'nodule.nii.gz')
            ex_image_path = os.path.join(exhale_folder, 'image.nii.gz')
            ex_mask_path = os.path.join(exhale_folder, 'lobe.nii.gz')
            ex_nodule_path = os.path.join(nodule_root, case, 'nodule.nii.gz')

            registrator = Registrator(in_image_path, ex_image_path)
            try:
                registrator.RegistrateBySpacing(
                    store_path=os.path.join(save_root, 'exhale', 'image', '{}.nii.gz'.format(case)))
            except:
                print('Align Exhale Image Failed')

            registrator = Registrator(in_image_path, in_mask_path)
            try:
                registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                                dtype=sitk.sitkInt32,
                                                store_path=os.path.join(save_root, 'inhale', 'mask',
                                                                        '{}.nii.gz'.format(case)))
            except:
                print('Align Inhale Mask Failed')

            registrator = Registrator(in_image_path, ex_mask_path)
            try:
                registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                                dtype=sitk.sitkInt32,
                                                store_path=os.path.join(save_root, 'exhale', 'mask',
                                                                        '{}.nii.gz'.format(case)))
            except:
                print('Align Exhale Mask Failed')

            if os.path.exists(in_nodule_path):
                os.makedirs(os.path.join(save_root, 'inhale', 'nodule'))
                registrator = Registrator(in_image_path, in_nodule_path)
                try:
                    registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                                    dtype=sitk.sitkInt32,
                                                    store_path=os.path.join(save_root, 'inhale', 'nodule',
                                                                            '{}.nii.gz'.format(case)))
                except:
                    print('Align Inhale Nodule Failed')

            if os.path.exists(ex_nodule_path):
                os.makedirs(os.path.join(save_root, 'exhale', 'nodule'))
                registrator = Registrator(in_image_path, ex_nodule_path)
                try:
                    registrator.RegistrateBySpacing(method=sitk.sitkNearestNeighbor,
                                                    dtype=sitk.sitkInt32,
                                                    store_path=os.path.join(save_root, 'exhale', 'nodule',
                                                                            '{}.nii.gz'.format(case)))
                except:
                    print('Align Inhale Nodule Failed')

            shutil.copyfile(in_image_path, os.path.join(save_root, 'inhale', 'image', '{}.nii.gz'.format(case)))
            # break

def FlipAxis(image, save_folder=r''):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    direction = image.GetDirection()
    dim0 = True if direction[0] == -1 else False
    dim1 = True if direction[4] == -1 else False
    dim2 = True if direction[8] == -1 else False
    if dim2 == dim1 == dim0 == False:
        return image
    else:
        filp_filter = sitk.FlipImageFilter()
        filp_filter.SetFlipAxes([dim0, dim1, dim2])
        new_image = filp_filter.Execute(image)
        assert new_image.GetDirection()[0] == new_image.GetDirection()[4] == new_image.GetDirection()[8] == 1
        if save_folder: sitk.WriteImage(new_image, save_folder)
        return new_image

def TestFlipAxis():
    pbar = tqdm(total=len(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample')), ncols=80)
    for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample')):
        try:
            case_folder = os.path.join(r'/data/data1/zyh/Data/CTLung/Resample', case)
            [os.remove(os.path.join(case_folder, data)) for data in os.listdir(case_folder) if 'rigid' in data]
            [FlipAxis(os.path.join(case_folder, data), os.path.join(case_folder, data)) for data in os.listdir(case_folder) if data.endswith('.nii.gz')]
        except Exception as e:
            print(case, e)
        pbar.update()
    pbar.close()

def Resample(root, save_root, only_nodule=False, case_list=[], replace=False):
    resampler = Resampler()
    if len(case_list) == 0: case_list = sorted(os.listdir(os.path.join(root, 'inhale', 'image')))
    for c in ['inhale', 'exhale']:
        image_folder = os.path.join(root, c, 'image')
        mask_folder = os.path.join(root, c, 'mask')
        nodule_folder = os.path.join(root, c, 'nodule')
        pbar = tqdm(total=len(case_list), ncols=80)
        for case in case_list:
            if not case.endswith('.nii.gz'): continue
            case_name = case.replace(' ', '_').split('.nii.gz')[0]
            if (not replace) and (os.path.exists(os.path.join(save_root, case_name))): continue
            if not os.path.exists(os.path.join(save_root, case_name)): os.makedirs(os.path.join(save_root, case_name))
            if only_nodule:
                nodule = LoadImage(os.path.join(nodule_folder, case))[0]
                new_nodule = resampler.ResampleImage(nodule, [1.35, 1.35, 1.35], sitk.sitkNearestNeighbor)
                castImageFilter = sitk.CastImageFilter()
                castImageFilter.SetOutputPixelType(sitk.sitkInt32)
                castImageFilter.Execute(new_nodule)
                FlipAxis(new_nodule, os.path.join(save_root, case_name, '{}_nodule.nii.gz'.format(c)))
            else:
                image = LoadImage(os.path.join(image_folder, case))[0]
                mask = LoadImage(os.path.join(mask_folder, case))[0]
                resample_image = resampler.ResampleImage(image, [1.35, 1.35, 1.35], sitk.sitkBSpline,
                                                         save_path=r'',
                                                         defualt_pixel=-1024.)
                FlipAxis(resample_image,  os.path.join(save_root, case_name, '{}.nii.gz'.format(c)))

                resample_mask = resampler.ResampleToReference(mask, resample_image, sitk.sitkNearestNeighbor,
                                                              save_path=r'', is_roi=True)
                FlipAxis(resample_mask,  os.path.join(save_root, case_name, '{}_mask.nii.gz'.format(c)))

                if os.path.exists(os.path.join(nodule_folder, case)):
                    nodule = LoadImage(os.path.join(nodule_folder, case))[0]
                    resample_nodule = resampler.ResampleToReference(nodule, resample_image, sitk.sitkNearestNeighbor,
                                                                    save_path=r'', is_roi=True)
                    FlipAxis(resample_nodule,  os.path.join(save_root, case_name, '{}_nodule.nii.gz'.format(c)))
            pbar.update()
        pbar.close()

def Crop3DEdge(mask, image, des_shape, nodule=None, center=[]):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param array: the 3D numpy array
    :return: the cropped image.
    '''
    new_image = np.ones(shape=des_shape) * -1024
    new_mask = np.zeros(shape=des_shape)
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
    if not center:
        nonzero_x = np.nonzero(np.sum(mask, axis=(0, 1)))[0]
        center_x = (nonzero_x[0] + nonzero_x[-1]) / 2
        target_x_0 = int(max(center_x - des_shape[2] / 2, 0))
        target_x_1 = int(min(center_x + des_shape[2] / 2, mask.shape[-1]))
    else:
        target_x_0 = center[0]
        target_x_1 = center[1]
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

def CopyInfo(image, ref_image, min_coord=[]):
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)
    if len(min_coord) == 0:
        image.SetOrigin(ref_image.GetOrigin())
    else:
        new_origin = ref_image.GetOrigin() + ref_image.GetSpacing() * np.asarray(min_coord[::-1])
        image.SetOrigin(new_origin)
    image.SetSpacing(ref_image.GetSpacing())
    image.SetDirection(ref_image.GetDirection())
    return image

def Normailzation(data):
    normal_data = np.asarray(data, dtype=np.float32)
    if normal_data.max() - normal_data.min() < 1e-6:
        return np.zeros_like(normal_data)
    normal_data = np.clip(normal_data, a_min=-1024, a_max=600)
    normal_data = normal_data - np.min(normal_data)
    normal_data = normal_data / np.max(normal_data)
    normal_data = normal_data * 255
    return normal_data

def Crop(root, save_root, keys=[], case_list=[]):
    if not os.path.exists(save_root): os.makedirs(save_root)
    des_shape = (256, 256, 256)
    if len(case_list) == 0: case_list = os.listdir(root)
    pbar = tqdm(total=len(case_list), ncols=80)
    for index, case in enumerate(sorted(case_list)):
        center = 0
        if case.endswith('.nii.gz'): case = case.split('.nii.gz')[0]
        if case == '20201126_gu_kang_xiu.nii.gz': continue
        save_folder = os.path.join(save_root, case)
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        for c in keys:
            try:
                image_path = os.path.join(root, case, '{}.nii.gz'.format(c))
                mask_path = os.path.join(root, case, '{}_mask.nii.gz'.format(c))
                nodule_path = os.path.join(root, case, '{}_nodule.nii.gz'.format(c))

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
                if c == 'inhale' or c == 'exhale':
                    crop_image_arr, crop_mask_arr, crop_nodule_arr, min_coord, max_coord = Crop3DEdge(mask_arr,
                                                                                                      image_arr,
                                                                                                      des_shape,
                                                                                                      nodule_arr)
                    if c == 'inhale': center = [min_coord[0], max_coord[0]]

                else:
                    if center == 0:
                        inhale_mask = np.transpose(LoadImage(os.path.join(root, case, 'inhale_mask.nii.gz'))[1], axes=(2, 0, 1))
                        nonzero_x = np.nonzero(np.sum(inhale_mask, axis=(0, 1)))[0]
                        center_x = (nonzero_x[0] + nonzero_x[-1]) / 2
                        center = (int(max(center_x - des_shape[2] / 2, 0)), int(min(center_x + des_shape[2] / 2, inhale_mask.shape[-1])))
                    crop_image_arr, crop_mask_arr, crop_nodule_arr, min_coord, max_coord = Crop3DEdge(mask_arr,
                                                                                                      image_arr,
                                                                                                      des_shape,
                                                                                                      nodule_arr,
                                                                                                      center)

                crop_image_arr = Normailzation(crop_image_arr)

                crop_mask = CopyInfo(crop_mask_arr, image)
                crop_mask = CastImage(crop_mask, dtype=sitk.sitkInt32)
                crop_image = CopyInfo(crop_image_arr, image)
                crop_image = CastImage(crop_image, dtype=sitk.sitkFloat32)
                if root == save_root:
                    sitk.WriteImage(crop_image, os.path.join(save_folder, '{}_crop.nii.gz'.format(c)))
                    sitk.WriteImage(crop_mask, os.path.join(save_folder, '{}_mask_crop.nii.gz'.format(c)))
                else:
                    sitk.WriteImage(crop_image, os.path.join(save_folder, '{}.nii.gz'.format(c)))
                    sitk.WriteImage(crop_mask, os.path.join(save_folder, '{}_mask.nii.gz'.format(c)))

                if os.path.exists(nodule_path):
                    crop_nodule = CopyInfo(crop_nodule_arr, image)
                    crop_nodule = CastImage(crop_nodule, dtype=sitk.sitkInt32)
                    if root == save_root:
                        sitk.WriteImage(crop_nodule, os.path.join(save_folder, '{}_nodule_crop.nii.gz'.format(c)))
                    else:
                        sitk.WriteImage(crop_nodule, os.path.join(save_folder, '{}_nodule.nii.gz'.format(c)))
            except Exception as e:
                print(case.split('.nii.gz')[0], e)
        pbar.update()
    pbar.close()

def CropNodule(root, save_root, case_list=[]):
    # CreateFolder(save_root)
    if not os.path.exists(save_root): os.makedirs(save_root)
    des_shape = (256, 256, 256)
    # pbar = tqdm(total=len(os.listdir(os.path.join(root, 'inhale', 'image'))), ncols=80)
    for index, case in enumerate(sorted(os.listdir(os.path.join(root, 'inhale', 'image')))):
        if case == '20201126_gu_kang_xiu.nii.gz': continue
        save_folder = os.path.join(save_root, case.split('.nii.gz')[0])
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        if not os.path.exists(os.path.join(root, 'inhale', 'nodule', case)): continue
        if os.path.exists(os.path.join(save_folder, 'inhale_nodule.nii.gz')): continue
        print(case)
        for c in ['inhale', 'exhale']:
            mask_path = os.path.join(root, c, 'mask', case)
            nodule_path = os.path.join(root, c, 'nodule', case)
            try:
                mask, mask_arr, _ = LoadImage(mask_path)
                mask_arr = np.transpose(mask_arr, axes=(2, 0, 1))
                nodule, nodule_arr, _ = LoadImage(nodule_path)
                nodule_arr = np.transpose(nodule_arr, axes=(2, 0, 1))
                # crop mask
                crop_image_arr, crop_mask_arr, crop_nodule_arr, min_coord, max_coord = Crop3DEdge(mask_arr, nodule_arr, des_shape, nodule_arr)
                crop_nodule = CopyInfo(crop_nodule_arr, mask, min_coord)
                sitk.WriteImage(crop_nodule, os.path.join(save_folder, '{}_nodule.nii.gz'.format(c)))
            except Exception as e:
                print(case.split('.nii.gz')[0], e)
        # pbar.update()
    # pbar.close()

def ExchangeFile(source, target):
    temp = os.path.join(os.path.dirname(target), 'temp.nii.gz')
    os.rename(source, temp)
    os.rename(target, source)
    os.rename(temp, target)

def CheckData(root, case_list=None):
    # in volume > ex volume
    exchange_case = []
    if not case_list:
        case_list = os.listdir(root)
    for case in sorted(case_list):
        if case == '20201126_gu_kang_xiu': continue
        if case.endswith('.nii.gz'): case = case.split('.nii.gz')[0]
        inhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'inhale_mask.nii.gz')))
        exhale_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'exhale_mask.nii.gz')))
        if np.count_nonzero(exhale_mask) >= np.count_nonzero(inhale_mask):
            ExchangeFile(os.path.join(root, case, 'inhale.nii.gz'), os.path.join(root, case, 'exhale.nii.gz'))
            ExchangeFile(os.path.join(root, case, 'inhale_mask.nii.gz'), os.path.join(root, case, 'exhale_mask.nii.gz'))
            if os.path.exists(os.path.join(os.path.join(root, case, 'inhale_nodule.nii.gz'))):
                ExchangeFile(os.path.join(os.path.join(root, case, 'inhale_nodule.nii.gz')), os.path.join(os.path.join(root, case, 'exhale_nodule.nii.gz')))
            print(case.split('.nii.gz')[0], 'exchange successfully!')
            exchange_case.append(case)
    df = pd.DataFrame(exchange_case)
    df.to_csv(os.path.join(os.path.dirname(root), 'ExchangeFile.csv'), header=False, index=False, mode='a')

def InvertData(root, case_list=None):
    if len(case_list) == 0:
        case_list = sorted(os.listdir(root))
    for case in case_list:
        if case.endswith('.nii.gz'): case = case.split('.nii.gz')[0]
        image = sitk.ReadImage(os.path.join(root, case, 'inhale.nii.gz'))
        image_arr = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, case, 'inhale_mask.nii.gz')))
        mask[mask > 1] = 1
        new_image_arr = 255. - image_arr
        new_image_arr = new_image_arr * mask
        new_image = sitk.GetImageFromArray(new_image_arr)
        new_image.CopyInformation(image)
        sitk.WriteImage(new_image, os.path.join(root, case, 'inhale_invert.nii.gz'))



if __name__ == '__main__':
    print()
    ############################         step 1: Dicom to Nii       ############################
    # DataConvert(raw_folder=r'')
    ############################         hydra preprocessing        ############################
    # HydraPrecess = HydraNiiPreprocess(r'/data/data1/zyh/Data/CTLung/RawData/Hydra202211',
    #                                   r'/data/data1/zyh/Data/CTLung/RawData/Hydra202211',
    #                                   hydra_code=137)
    # HydraPrecess.Run()
    # HydraPrecess.SplitInEx()

    # CopyNii(src_root=r'', des_root=r'')
    ############################ step 2: Exhale registrate to inhale ############################
    # Registrate(root_path=r'/data/data1/zyh/Data/CTLung/RawData/Hydra202211',
    #            save_root=r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211',
    #            is_nodule=False)
    ############################ step 2: Exhale resample to inhale ############################
    # Resample(root=r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211',
    #          save_root=r'/data/data1/zyh/Data/CTLung/Resample',
    #          case_list=[],
    #          replace=True)
    # CheckData(r'/data/data1/zyh/Data/CTLung/Resample',
    #           case_list=os.listdir(r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211/exhale/image'))

    ############################ step 3: Crop & Normalization ############################
    # Crop(root=r'/data/data1/zyh/Data/CTLung/Resample',
    #      save_root=r'/data/data1/zyh/Data/CTLung/CropData',
    #      keys=['rigid'],
    #      case_list=[])
    # Crop(root=r'/data/data1/zyh/Data/CTLung/Resample/Hydra', save_root=r'/data/data1/zyh/Data/CTLung/CropData')
    # processed_nodules = os.listdir(r'/data/data1/zyh/Data/CTLung/CNNData/moving_nodule')
    # all_nodules = os.listdir(r'/data/data1/zyh/Data/CTLung/Resample/SHCH/inhale/nodule')
    # all_nodules = list(set([nodule.replace(' ', '_') for nodule in all_nodules]))
    # CropNodule(root=r'/data/data1/zyh/Data/CTLung/Resample/SHCH',
    #            save_root=r'/data/data1/zyh/Data/CTLung/CNNData')
    ############################ step 5: ExchangeErrorData ############################
    # CheckData(root=r'/data/data1/zyh/Data/CTLung/CropData')

    ###################################################################################
    InvertData(r'/data/data1/zyh/Data/CTLung/CropData',
               case_list=['20191112_luo_hua_shi'])








    


