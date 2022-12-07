import SimpleITK as sitk
import numpy as np
import random
from scipy.ndimage.interpolation import rotate, zoom
import scipy.linalg as linalg


class DataAugmentor3D():
    '''
    To process 3D numpy array transform. The transform contains: stretch in 3 dimensions, shear along x direction,
    rotation around z and x axis, shift along x, y, z direction, and flip along x, y, z direction.
    '''
    def __init__(self):
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.stretch_z = 1.0
        self.shear = 0.0
        self.rotate_x_angle = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.slice_flip = False

        self.bias_center = [0.5, 0.5]   # 默认bias的中心在图像中央
        self.bias_drop_ratio = 0.0

        self.noise_sigma = 0.0
        self.factor = 1.0
        self.gamma = 1.0

        self.elastic = False
        self.is_debug = False

    def RandomParameter(self, param):
        if isinstance(param, list):
            if isinstance(param[0], list):
                return [random.uniform(param[0][0], param[0][1]), random.uniform(param[1][0], param[1][1])]
            else:
                return random.uniform(param[0], param[1])
        else:
            return param

    def RandomAxis(self, param):
        axis = [i for i in range(len(param)) if param[i] == 1]
        rotate_axis = [0, 0, 0]
        rotate_axis[random.choice(axis)] = 1
        return rotate_axis

    def ClearParameters(self):
        self.__init__()

    def SetParameter(self, parameter_dict):
        if 'zoom' in parameter_dict: self.zoom = self.RandomParameter(parameter_dict['zoom'])
        if 'rotate_angle' in parameter_dict: self.rotate_angle = self.RandomParameter(parameter_dict['rotate_angle'])
        if 'rotate_axis' in parameter_dict: self.rotate_axis = self.RandomAxis(parameter_dict['rotate_axis'])
        if 'horizontal_flip' in parameter_dict: self.horizontal_flip = self.RandomParameter(parameter_dict['horizontal_flip'])
        if 'vertical_flip' in parameter_dict: self.vertical_flip = self.RandomParameter(parameter_dict['vertical_flip'])
        if 'volume_percent' in parameter_dict: self.new_volume_percent = self.RandomParameter(parameter_dict['volume_percent'])

    def _Flip3DImage(self, data):
        if self.horizontal_flip: data = np.ascontiguousarray(np.flip(data, axis=2))
        # if self.vertical_flip: data = np.ascontiguousarray(np.flip(data, axis=0))
        # if self.slice_flip: data = np.ascontiguousarray(np.flip(data, axis=2))
        # if self.horizontal_flip: data = np.copy(np.flipud(data))
        return data

    def _ZoomImage(self, data, ratio, is_roi=False):
        order = 0 if is_roi else 3
        zoom_data = zoom(data, ratio, order=order)
        return zoom_data

    def _ZoomFlow(self, flow, volume_percent):
        flow = flow / volume_percent * self.new_volume_percent
        return flow

    def _RotateCoords(self, axis: list, angle, coords):
        radian = angle * np.pi / 180
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        new_coords = np.zeros_like(coords)
        new_coords[..., 0] = rot_matrix[0, 0] * coords[..., 0] + rot_matrix[0, 1] * coords[..., 1] + rot_matrix[0, 2] * coords[..., 2]
        new_coords[..., 1] = rot_matrix[1, 0] * coords[..., 0] + rot_matrix[1, 1] * coords[..., 1] + rot_matrix[1, 2] * coords[..., 2]
        new_coords[..., 2] = rot_matrix[2, 0] * coords[..., 0] + rot_matrix[2, 1] * coords[..., 1] + rot_matrix[2, 2] * coords[..., 2]
        return new_coords

    def _Rotate(self, moving_image, deform_flow, moving_mask_list=[]):
        applied_angle = self.rotate_angle
        target_input = rotate(moving_image, applied_angle, reshape=False, axes=np.nonzero(1 - np.array(self.rotate_axis))[0])
        target_flow = rotate(deform_flow, applied_angle, reshape=False, axes=np.nonzero(1 - np.array(self.rotate_axis))[0])
        target_mask_list = [rotate(mask, applied_angle, reshape=False, axes=np.nonzero(1 - np.array(self.rotate_axis))[0], order=0) for mask in moving_mask_list]
        target_flow = self._RotateCoords(self.rotate_axis, applied_angle, target_flow)
        return target_input, target_flow, target_mask_list

    def Execute(self, source_data, source_flow, volume_percent, source_mask=[], parameter_dict={}, is_clear=False):
        if not (source_data.ndim == 3 and source_flow.ndim == 4):
            print('Check Shape of Data or Flow!')
            return source_data, source_flow, volume_percent
        if parameter_dict != {}:
            self.SetParameter(parameter_dict)

        target_data, target_flow, target_mask = source_data, source_flow, source_mask
        if 'rotate_axis' in parameter_dict.keys() and 'rotate_angle' in parameter_dict.keys():
            print(self.rotate_angle)
            target_data, target_flow, target_mask = self._Rotate(source_data, source_flow, source_mask)
        if 'zoom' in parameter_dict.keys():
            target_data, target_flow, target_mask = self._ZoomImage(target_data, self.zoom), \
                                                    self._ZoomImage(target_flow, self.zoom),\
                                                    [self._ZoomImage(mask, self.zoom, is_roi=True) for mask in target_mask],
        if 'horizontal_flip' in parameter_dict.keys():
            target_data = self._Flip3DImage(target_data)
            target_flow = self._Flip3DImage(target_flow)
            # target_flow_x = np.ascontiguousarray(np.flip(target_flow[..., 0], axis=2))
            # target_flow_y = np.ascontiguousarray(np.flip(target_flow[..., 1], axis=2))
            # target_flow_z = np.ascontiguousarray(np.flip(target_flow[..., 2], axis=2))
            # target_flow = np.stack([target_flow_x, target_flow_y, target_flow_z], axis=-1)
            target_flow[:, :, :, 2] *= -1
            target_mask = [self._Flip3DImage(mask) for mask in target_mask]

        if 'volume_percent' in parameter_dict.keys():
            target_flow = self._ZoomFlow(target_flow, volume_percent)
        else:
            self.new_volume_percent = volume_percent

        if is_clear:
            self.ClearParameters()
        else:
            return target_data, target_flow, self.new_volume_percent, target_mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    inhale_image = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/inhale.nii.gz'))
    inhale_lung = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/inhale_mask.nii.gz'))
    inhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/inhale_nodule.nii.gz'))
    exhale_image = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/rigid.nii.gz'))
    exhale_lung = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/rigid_mask.nii.gz'))
    exhale_nodule = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/rigid_nodule.nii.gz'))
    deform_field = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/deform_flow.nii.gz'))
    volume_percent = (np.count_nonzero(inhale_lung) - np.count_nonzero(exhale_lung)) / np.count_nonzero(inhale_lung)

    random_3d_augment = {
        # 'zoom': [1, 1.25],  # 缩放？
        # 'horizontal_flip': True,  # 翻转
        # 'volume_percent': [0.05, 0.5],
        # 'rotate_angle': [0, 90],
        # 'rotate_axis': [1, 0, 0]
    }




