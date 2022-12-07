import os
import shutil
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
import numpy as np

from AllPreprocee import Crop
# from AllPreprocee import CastImage


def ElastixBatch(fixed_image, moving_image, out_image, fixed_mask=None, moving_mask=None, parameters=[]):
    '''
    elastix registers a moving image to a fixed image.
    The registration-process is specified in the parameter file.
      --help, -h displays this message and exit
      --version  output version information and exit
      --extended-version  output extended version information and exit

    Call elastix from the command line with mandatory arguments:
      -f        fixed image
      -m        moving image
      -out      output directory
      -p        parameter file, elastix handles 1 or more "-p"

    Optional extra commands:
      -fMask    mask for fixed image
      -mMask    mask for moving image
      -t0       parameter file for initial transform
      -priority set the process priority to high, abovenormal, normal (default),
                belownormal, or idle (Windows only option)
      -threads  set the maximum number of threads of elastix

    The parameter-file must contain all the information necessary for elastix to run properly.
    That includes which metric to use, which optimizer, which transform, etc.
    It must also contain information specific for the metric, optimizer, transform, etc.
    For a usable parameter-file, see the website http://elastix.isi.uu.nl.
    '''

    if parameters == []:
        raise IndexError('must have registration parameters!')
    if moving_mask and fixed_mask:
        input_info = ' '.join(['-f', fixed_image, '-m', moving_image, '-out', out_image, '-fMask', fixed_mask, '-mMask', moving_mask])
    else:
        input_info = ' '.join(['-f', fixed_image, '-m', moving_image, '-out', out_image])
    parameters_info = ' '.join([' '.join(['-p', p]) for p in parameters])
    cmd = ' '.join(['elastix', input_info, parameters_info, '-threads', '4'])
    os.system(cmd)


def Transformix(transformix_parameter, out_image, in_image):
    '''
    transformix applies a transform on an input image and/or generates a deformation field.
    The transform is specified in the transform-parameter file.
        --help, -h displays this message and exit
        --version  output version information and exit

    Call transformix from the command line with mandatory arguments:
      -out      output directory
      -tp       transform-parameter file, only 1

    Optional extra commands:
      -in       input image to deform
      -def      file containing input-image points; the point are transformed
                according to the specified transform-parameter file
                use "-def all" to transform all points from the input-image, which
                effectively generates a deformation field.
      -jac      use "-jac all" to generate an image with the determinant of the
                spatial Jacobian
      -jacmat   use "-jacmat all" to generate an image with the spatial Jacobian
                matrix at each voxel
      -priority set the process priority to high, abovenormal, normal (default),
                belownormal, or idle (Windows only option)
      -threads  set the maximum number of threads of transformix

    At least one of the options "-in", "-def", "-jac", or "-jacmat" should be given.
    '''
    input_info = ' '.join(['-out', out_image, '-tp', transformix_parameter, '-in', in_image, '-def all', '-threads', '4'])
    cmd = ' '.join(['transformix', input_info])
    os.system(cmd)


def mhd2nii(mhd_path, new_name=None, dtype=sitk.sitkFloat32, is_mask=False):
    if new_name:
        nii_path = os.path.join(os.path.dirname(mhd_path),
                                '{}.nii.gz'.format(new_name))
    else:
        nii_path = os.path.join(os.path.dirname(mhd_path),
                                '{}.nii.gz'.format(os.path.basename(mhd_path).split('.mhd')[0]))
    image = sitk.ReadImage(mhd_path)
    if is_mask == True:
        new_image = CastImage(image, dtype)
        sitk.WriteImage(new_image, nii_path)
    else:
        sitk.WriteImage(image, nii_path)


def CastImage(inputImage, dtype=sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(dtype)
    inputImage = castImageFilter.Execute(inputImage)
    return inputImage


def ClosingBinary(mask_path):
    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr[mask_arr > 0] = 1
    new_mask_arr = ndimage.binary_closing(mask_arr, structure=np.ones((3, 3, 3))).astype(np.int32)
    new_mask = sitk.GetImageFromArray(new_mask_arr)
    new_mask.CopyInformation(mask)
    sitk.WriteImage(new_mask, mask_path)
    # sitk.BinaryMorphologicalClosing(sitk.ReadImage(mask_path) != 0, kernelsize)


def MaskTransformixParam(param_path, replace=False):
    mask_param_path = os.path.join(os.path.dirname(param_path), '{}MaskTransformParameters.0.txt'.format(param_path.split('TransformParameters.0.txt')[0]))
    if not os.path.exists(mask_param_path) or replace:
        with open(param_path, "r") as f:
            param = f.read()
            new_param = param.replace('(FinalBSplineInterpolationOrder 3)', '(FinalBSplineInterpolationOrder 0)')
        with open(mask_param_path, "w") as f:
            f.write(new_param)
    return mask_param_path


def ImageTransformixParam(param_path, default=0.000000, replace=False):
    image_param_path = os.path.join(os.path.dirname(param_path), '{}ImageTransformParameters.0.txt'.format(
        param_path.split('TransformParameters.0.txt')[0]))
    if not os.path.exists(image_param_path) or replace:
        with open(param_path, "r") as f:
            param = f.read()
            new_param = param.replace('(DefaultPixelValue 0.000000)', '(DefaultPixelValue {})'.format(default))
        with open(image_param_path, "w") as f:
            f.write(new_param)
    return image_param_path


def ElastixIn2Ex(image_root, save_root, transform=[], moving_key='moving', fixed_key='fixed', is_mask=False, case_list=[], replace=False):
    '''
    fixed image: inhale
    moving image: exhale
    '''
    result_key = '_'.join(transform)
    parameters = []
    statistics_filter = sitk.StatisticsImageFilter()
    for trans in transform:
        if trans == 'rigid':
            parameters.append(r'/data/data1/zyh/Data/CTLung/Parameters_Rigid.txt')
        elif trans == 'deform':
            parameters.append(r'/data/data1/zyh/Data/CTLung/Parameters.Par0015.expA.patient.NC.bspline.txt')
        elif trans == 'affine':
            parameters.append(r'')
        else:
            raise ValueError('only support rigid or affine or deform!')

    if len(case_list) == 0:
        case_list = sorted(os.listdir(image_root))

    for case in case_list:
        if '20201126_gu_kang_xiu' in case: continue
        case_name = case.split('.nii.gz')[0]
        if not os.path.isdir(os.path.join(image_root, case_name)): continue

        # Elastix image
        out_folder = os.path.join(save_root, case_name)
        if not os.path.exists(out_folder): os.makedirs(out_folder)  # 如果保存路径不存在，创建路径
        fixed_image = os.path.join(image_root, case_name, '{}.nii.gz'.format(fixed_key))
        moving_image = os.path.join(image_root, case_name, '{}.nii.gz'.format(moving_key))
        fixed_mask = os.path.join(image_root, case_name, '{}_mask.nii.gz'.format(fixed_key))
        moving_mask = os.path.join(image_root, case_name, '{}_mask.nii.gz'.format(moving_key))
        try:
            if (not os.path.exists(os.path.join(out_folder, '{}_TransformParameters.0.txt'.format(result_key)))) or replace:  # 如果没有配准参数，则elastix
                if is_mask:
                    ElastixBatch(fixed_image, moving_image, out_folder, fixed_mask, moving_mask, parameters=parameters)
                else:
                    ElastixBatch(fixed_image, moving_image, out_folder, parameters=parameters)
                os.rename(os.path.join(out_folder, 'TransformParameters.0.txt'),
                          os.path.join(out_folder, '{}_TransformParameters.0.txt'.format(result_key)))

            if (not os.path.exists(os.path.join(out_folder, '{}_mask.nii.gz').format(result_key))) or replace:
                if os.path.exists(moving_mask):
                    Transformix(MaskTransformixParam(os.path.join(out_folder, '{}_TransformParameters.0.txt'.format(result_key)), replace),
                                out_folder,
                                moving_mask)
                    mhd2nii(os.path.join(out_folder, 'result.mhd'), '{}_mask'.format(result_key), dtype=sitk.sitkInt32, is_mask=True)

            if (not os.path.exists(os.path.join(out_folder, '{}_nodule.nii.gz').format(result_key))) or replace:
                if os.path.exists(os.path.join(image_root, case_name, '{}_nodule.nii.gz'.format(moving_key))):
                    moving_nodule = os.path.join(image_root, case_name, '{}_nodule.nii.gz'.format(moving_key))
                    Transformix(MaskTransformixParam(os.path.join(out_folder, '{}_TransformParameters.0.txt'.format(result_key)), replace),
                                out_folder,
                                moving_nodule)
                    mhd2nii(os.path.join(out_folder, 'result.mhd'), '{}_nodule'.format(result_key), dtype=sitk.sitkInt32, is_mask=True)

            if (not os.path.exists(os.path.join(out_folder, '{}.nii.gz'.format(result_key)))) or replace:
                statistics_filter.Execute(sitk.ReadImage(moving_image))
                Transformix(ImageTransformixParam(os.path.join(out_folder, '{}_TransformParameters.0.txt'.format(result_key)),
                                                  default=statistics_filter.GetMinimum(), replace=replace),
                            out_folder,
                            moving_image)
                mhd2nii(os.path.join(out_folder, 'result.mhd'), result_key)
                mhd2nii(os.path.join(out_folder, 'deformationField.mhd'), '{}_flow'.format(result_key))

        except Exception as e: print(case_name, e)

        try:
            txt_list = [txt for txt in os.listdir(out_folder) if (txt.endswith('.txt') and 'TransformParameters.0.txt' not in txt)]
            [os.remove(os.path.join(out_folder, remove)) for remove in txt_list]
            nii_list = [nii for nii in os.listdir(out_folder) if ('result' in nii or 'deformationField' in nii)]
            [os.remove(os.path.join(out_folder, remove)) for remove in nii_list]
            [os.remove(os.path.join(out_folder, remove)) for remove in os.listdir(out_folder) if remove.endswith('.log')]
        except Exception as e:
            print(case_name, e)


if __name__ == '__main__':
    # StandardFileName()
    # case_df = pd.read_csv(r'D:\Data\test.csv', index_col=0).squeeze()

    # if affine → deform
    # ElastixIn2Ex(image_root=r'D:\Data\Crop',
    #              save_root=r'D:\Data\Affine',
    #              parameters=['D:\elastix\Parameters\Parameters.Par0015.expA.patient.NC.affine.txt',
    #                          'D:\elastix\Parameters\Parameters.Par0015.expA.patient.NC.bspline.txt'],
    #              moving_key='moving',
    #              fixed_key='fixed',
    #              case_list=[])

    # if rigid
    #
    # ElastixIn2Ex(image_root=r'/data/data1/zyh/Data/CTLung/Resample',
    #              save_root=r'/data/data1/zyh/Data/CTLung/Resample',
    #              transform=['rigid'],
    #              moving_key='exhale',
    #              fixed_key='inhale',
    #              is_mask=True,
    #              case_list=os.listdir(r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211/exhale/image'))

    # Crop(root=r'/data/data1/zyh/Data/CTLung/Resample',
    #      save_root=r'/data/data1/zyh/Data/CTLung/CropData',
    #      keys=['inhale', 'exhale', 'rigid'],
    #      case_list=os.listdir(r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211/exhale/image'))

    # ElastixIn2Ex(image_root=r'/data/data1/zyh/Data/CTLung/CropData',
    #              save_root=r'/data/data1/zyh/Data/CTLung/CropData',
    #              transform=['deform'],
    #              moving_key='inhale',
    #              fixed_key='rigid',
    #              is_mask=False,
    #              case_list=os.listdir(r'/data/data1/zyh/Data/CTLung/Registrator/Hydra202211/exhale/image'),
    #              replace=True)

    # for case in sorted(os.listdir(r'/data/data1/zyh/Data/CTLung/Resample')):
    #     if not os.path.exists(os.path.join(r'/data/data1/zyh/Data/CTLung/Resample', case, 'rigid.nii.gz')):
    #         print(case)
    # if deform
    Transformix(r'/data/data1/zyh/Data/CTLung/20191112_luo_hua_shi/deform_ImageTransformParameters.0.txt',
                r'/data/data1/zyh/Data/CTLung/20191112_luo_hua_shi',
                r'/data/data1/zyh/Data/CTLung/20191112_luo_hua_shi/inhale_invert.nii.gz')
    mhd2nii(os.path.join(r'/data/data1/zyh/Data/CTLung/20191112_luo_hua_shi', 'result.mhd'), 'deform_invert')
    mhd2nii(os.path.join(r'/data/data1/zyh/Data/CTLung/20191112_luo_hua_shi', 'deformationField.mhd'), '{}_flow'.format('deform_invert'))









