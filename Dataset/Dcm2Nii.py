import os
import time
from pathlib import Path
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd

from MIP4AIM.Utility.DicomInfo import DicomShareInfo
from MIP4AIM.Dicom2Nii.DataReader import DataReader
from MIP4AIM.Dicom2Nii.Dicom2Nii import ConvertDicom2Nii
from MIP4AIM.Application2Series.ManufactureMatcher import SeriesCopyer
from MIP4AIM.NiiProcess.DwiProcessor import DwiProcessor
from MIP4AIM.NiiProcess.Registrator import Registrator

from MeDIT.Log import CustomerCheck, Eclog


class Dcm2Nii:
    def __init__(self, raw_folder, processed_folder, failed_folder, is_overwrite=False):
        self.raw_folder = raw_folder
        self.process_folder = processed_folder
        self.failed_folder = failed_folder
        self.is_overwrite = is_overwrite
        self.dcm2niix_path = r'/data/data1/zyh/Project/RespiratoryCompensation/dcm2niix'

        self.dicom_info = DicomShareInfo()
        self.data_reader = DataReader()

        self.series_copyer = SeriesCopyer()
        self.dwi_processor = DwiProcessor()
        self.registrator = Registrator()

    def GetPath(self, case_folder):
        for root, dirs, files in os.walk(case_folder):
            if len(files) != 0:
                yield root, dirs, files

    def SeperateDWI(self, case_folder):
        self.dwi_processor.Seperate4DDwiInCaseFolder(case_folder)

    def ConvertDicom2Nii(self, case_folder):
        for root, dirs, files in os.walk(case_folder, topdown=False):
            # it is possible to one series that storing the DICOM
            if len(files) > 3 and len(dirs) == 0:
                # if self.dicom_info.IsDICOMFolder(root):
                ConvertDicom2Nii(Path(root), Path(root), dcm2niix_path=self.dcm2niix_path)

    def MoveFilaedCase(self, case):
        if not os.path.exists(os.path.join(self.failed_folder, case)):
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, case))
        else:
            add_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, '_{}'.format(add_time)))
        if os.path.exists(os.path.join(self.process_folder, case)):
            shutil.rmtree(os.path.join(self.process_folder, case))

    def InerativeCase(self):
        self.log = CustomerCheck(os.path.join(self.failed_folder, 'failed_log.csv'), patient=1,
                                 data={'State': [], 'Info': []})
        self.eclog = Eclog(os.path.join(self.failed_folder, 'failed_log_details.log')).GetLogger()
        for case in os.listdir(self.raw_folder):
            case_folder = os.path.join(self.raw_folder, case)

            print('Convert Dicom to Nii:\n {}'.format(case_folder))
            try:
                self.ConvertDicom2Nii(case_folder)
            except Exception as e:
                self.log.AddOne(case_folder, {'State': 'Dicom to Nii failed.', 'Info': e.__str__()})
                self.eclog.error(e)
                # self.MoveFilaedCase(case_folder)
                continue


def Test():
    processor = Dcm2Nii(raw_folder=r'',
                        processed_folder=r'',
                        failed_folder=r'/data/data2/datasets/ctLung/respiratory_compensation/Hydra202211/202211',
                        is_overwrite=False)
    for root, dirs, files in os.walk(r'/data/data2/datasets/ctLung/respiratory_compensation/Hydra202211/202211'):
        dcm_file = [file for file in files if file.endswith('.dcm')]
        if len(dcm_file) > 0:
            try:
                processor.ConvertDicom2Nii(root)
            except Exception as e:
                # processor.log.AddOne(root, {'State': 'Dicom to Nii failed.', 'Info': e.__str__()})
                # processor.eclog.error(e)
                continue


def CopyHydraNii():
    des_root = r'/data/data1/zyh/Data/CTLung/Hydra202211/'
    if os.path.exists(des_root):
        shutil.rmtree(des_root)
    os.makedirs(des_root)
    for root, dirs, files in os.walk(r'/data/data2/datasets/ctLung/respiratory_compensation/Hydra202211/202211'):
        if len(files) > 0:
            nii_files = [file for file in files if file.endswith('.nii')]
            if len(nii_files) > 0:
                # print(root)
                des_folder = os.path.join(des_root, root.split('/')[-3], root.split('/')[-2])
                des_folder = os.path.join(des_root, root.split('/')[-2])
                if not os.path.exists(des_folder): os.makedirs(des_folder)
                [shutil.copyfile(os.path.join(root, nii_file),  os.path.join(des_folder, nii_file))
                 for nii_file in nii_files]


def CopyNii():
    des_root = r'/data/data1/zyh/Data/CTLung/RawData/Batch5_nii'
    if os.path.exists(des_root): shutil.rmtree(des_root)
    os.makedirs(des_root)
    for root, dirs, files in os.walk(r'/data/data1/zyh/Data/CTLung/RawData/Batch5'):
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


def Del5x5():
    des_root = r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii'
    for case in os.listdir(des_root):
        case_folder = os.path.join(des_root, case)
        nii_list = [file for file in os.listdir(case_folder) if ('3.0_x_3.0' in file or 'Batch' in file or '5.0_x_5.0' in file)]
        remain_list = [file for file in os.listdir(case_folder) if file not in nii_list]
        if len(remain_list) > 2:
            [os.remove(os.path.join(case_folder, file)) for file in nii_list]


def PltShow():
    from MeDIT.SaveAndLoad import LoadImage
    des_root = r'/data/data1/zyh/Data/CTLung/RawData/FromHydra'
    image_root = r'/data/data1/zyh/Data/CTLung/RawData/Image'
    if not os.path.exists(image_root): os.mkdir(image_root)
    for case in os.listdir(des_root):
        case_folder = os.path.join(des_root, case)
        for data in os.listdir(case_folder):
            arr = LoadImage(os.path.join(case_folder, data))[1]
            plt.imshow(arr[..., arr.shape[-1] //2], cmap='gray')
            plt.axis('off')
            plt.title(data)
            plt.savefig(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/Image', '{}_{}.jpg'.format(case, data.split('.nii')[0])),
                        dpi=200, bbox_inches='tight')
            plt.close()
        print(case)


def ModelTypeData():
    root = r'/data/data1/zyh/Data/CTLung/RawData/FromHydra'
    for roots, dirs, files in os.walk(root):
        if len(files) > 0 and len(dirs) == 0:
            # if len(files) > 1: file = sorted(files)[0]
            # else: file = files[0]
            # shutil.copyfile(os.path.join(roots, file),
            #                 os.path.join(roots, 'image.nii.gz'))
            [os.remove(os.path.join(roots, file)) for file in files if not (file == 'image.nii.gz')]
            print('Successful to rename {}_{}'.format(os.path.basename(os.path.dirname(roots)), os.path.basename(roots)))


if __name__ == '__main__':
    # Test()
    CopyHydraNii()
    # CopyNii()
    # Del5x5()
    # PltShow()

    # des_root = r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii'
    # for index, case in enumerate(sorted(os.listdir(des_root))):
    #     # print(sorted(os.listdir(os.path.join(des_root, case))))
    #     # new_case = 'Case_{}'.format(index + 62)
    #     # if not os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii', new_case)):
    #     #     os.makedirs(os.path.join(r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii', new_case))
    #     case_folder = os.path.join(des_root, case)
    #     for file in os.listdir(case_folder):
    #         if file == 'image.nii.gz': continue
    #         else:
    #             os.remove(os.path.join(case_folder, file))
        # nii_list = sorted(os.listdir(case_folder))
        # if len(nii_list) == 3:
        #     new_list = [nii_list[0], nii_list[1]]
        #     shutil.copyfile(os.path.join(case_folder, new_list[0]),
        #                     os.path.join(case_folder, 'image.nii.gz'))
        #     os.makedirs(os.path.join(des_root, '{}_1'.format(case)))
        #     shutil.copyfile(os.path.join(case_folder, new_list[1]),
        #                     os.path.join(os.path.join(des_root, '{}_1'.format(case)), 'image.nii.gz'))
        # elif len(nii_list) == 4:
        #     new_list = [nii_list[0], nii_list[2]]
        #     shutil.copyfile(os.path.join(case_folder, new_list[0]),
        #                     os.path.join(case_folder, 'image.nii.gz'))
        #     os.makedirs(os.path.join(des_root, '{}_1'.format(case)))
        #     shutil.copyfile(os.path.join(case_folder, new_list[1]),
        #                     os.path.join(os.path.join(des_root, '{}_1'.format(case)), 'image.nii.gz'))
        # else:
        #     new_list = []
        #     print(case, '\t', new_list)

    # ModelTypeData()

    # import numpy as np
    # data_root = r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii'
    # case_list = [case for case in os.listdir(r'/data/data1/zyh/Data/CTLung/RawData/NewHydra_nii') if not case.endswith('_1')]
    # for case in case_list:
    #     if not os.path.exists(os.path.join(data_root, case, 'inhale')): os.makedirs(os.path.join(data_root, case, 'inhale'))
    #     if not os.path.exists(os.path.join(data_root, case, 'exhale')): os.makedirs(os.path.join(data_root, case, 'exhale'))
    #     case_1_folder = os.path.join(data_root, case)
    #     case_2_folder = os.path.join(data_root, '{}_1'.format(case))
    #     mask_1_path = os.path.join(case_1_folder, 'lobe.nii.gz')
    #     mask_2_path = os.path.join(case_2_folder, 'lobe.nii.gz')
    #     mask_1 = sitk.GetArrayFromImage(sitk.ReadImage(mask_1_path))
    #     mask_2 = sitk.GetArrayFromImage(sitk.ReadImage(mask_2_path))
    #     # mask1 inhale, mask2 exhale
    #     if np.sum(mask_1) > np.sum(mask_2):
    #         # case to inhale
    #         shutil.move(os.path.join(data_root, case, 'image.nii.gz'), os.path.join(data_root, case, 'inhale', 'image.nii.gz'))
    #         shutil.move(os.path.join(data_root, case, 'lobe.nii.gz'), os.path.join(data_root, case, 'inhale', 'lobe.nii.gz'))
    #         shutil.move(os.path.join(data_root, case, 'lung.nii.gz'), os.path.join(data_root, case, 'inhale', 'lung.nii.gz'))
    #         # case1 to exhale
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'image.nii.gz'), os.path.join(data_root, case, 'exhale', 'image.nii.gz'))
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'lobe.nii.gz'), os.path.join(data_root, case, 'exhale', 'lobe.nii.gz'))
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'lung.nii.gz'), os.path.join(data_root, case, 'exhale', 'lung.nii.gz'))
    #
    #     # mask2 inhale, mask1 exhale
    #     elif np.sum(mask_1) < np.sum(mask_2):
    #         # case1 to inhale
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'image.nii.gz'), os.path.join(data_root, case, 'inhale', 'image.nii.gz'))
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'lobe.nii.gz'), os.path.join(data_root, case, 'inhale', 'lobe.nii.gz'))
    #         shutil.move(os.path.join(data_root, '{}_1'.format(case), 'lung.nii.gz'), os.path.join(data_root, case, 'inhale', 'lung.nii.gz'))
    #         # case to exhale
    #         shutil.move(os.path.join(data_root, case, 'image.nii.gz'), os.path.join(data_root, case, 'exhale', 'image.nii.gz'))
    #         shutil.move(os.path.join(data_root, case, 'lobe.nii.gz'), os.path.join(data_root, case, 'exhale', 'lobe.nii.gz'))
    #         shutil.move(os.path.join(data_root, case, 'lung.nii.gz'), os.path.join(data_root, case, 'exhale', 'lung.nii.gz'))
    #
    #     else:
    #         print(case)
    #     shutil.rmtree(os.path.join(data_root, '{}_1'.format(case)))








