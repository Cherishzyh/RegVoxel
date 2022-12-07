import types
import random
import numbers
import numpy as np
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad(x, shape):
    diffh = shape[4] - x.size()[4]
    diffw = shape[3] - x.size()[3]
    diffd = shape[2] - x.size()[2]
    pad_size = [diffh//2, diffh-diffh//2,
                diffw//2, diffw-diffw//2,
                diffd//2, diffd-diffd//2]

    x = F.pad(x, pad_size)
    return x


class SpatialTransformer(nn.Module):
    """
    from voxelmorph
    """
    def __init__(self, size):
        super().__init__()
        
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor)

    def forward(self, img, flow, mask=False, is_min=False):
        if is_min: flow = flow * 128
        new_locs = self.grid.to(flow.device) + flow
        shape = flow.shape[2:]

        # normalize grid values to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mask:
            img = F.grid_sample(img, new_locs, mode='nearest', align_corners=True)
        else:
            img = F.grid_sample(img, new_locs, mode='bilinear', align_corners=True, padding_mode='reflection')
        return img


def AffineTransform(img, theta, mask=False):
    grid = F.affine_grid(theta, size=img.shape, align_corners=True)  # (batch,h,w,d,3)
    if mask:
        img = F.grid_sample(img, grid, mode='nearest', padding_mode='border', align_corners=True)
    else:
        img = F.grid_sample(img, grid, padding_mode='border', align_corners=True)
    return img


'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    co_transforms.Compose([
         co_transforms.CenterCrop(10),
         co_transforms.ToTensor(),
      ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input,target)
        return input, target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x D x C) to a torch.FloatTensor of shape (C x H x W x D )."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (3, 0, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input, target):
        return self.lambd(input, target)


# class CenterCrop(object):
#     """Crops the given inputs and target arrays at the center to have a region of
#     the given size. size can be a tuple (target_height, target_width)
#     or an integer, in which case the target will be of a square shape (size, size)
#     Careful, img1 and img2 may not be the same size
#     """
#     # TODO: crop by center
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, inputs, target):
#         h1, w1, _ = inputs[0].shape
#         h2, w2, _ = inputs[1].shape
#         h3, w3, _ = inputs[2].shape
#         th, tw, td = self.size
#         x1 = int(round((w1 - tw) / 2.))
#         y1 = int(round((h1 - th) / 2.))
#         z1 = int(round((h1 - th) / 2.))
#         x2 = int(round((w2 - tw) / 2.))
#         y2 = int(round((h2 - th) / 2.))
#         z2 = int(round((h1 - th) / 2.))
#
#         inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
#         inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
#         target = target[y1: y1 + th, x1: x1 + tw]
#         return inputs, target


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)

        target = ndimage.interpolation.zoom(target, ratio, order=self.order)
        target *= ratio
        return inputs, target

#
# class RandomCrop(object):
#     """Crops the given PIL.Image at a random location to have a region of
#     the given size. size can be a tuple (target_height, target_width)
#     or an integer, in which case the target will be of a square shape (size, size)
#     """
#
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, inputs,target):
#         h, w, _ = inputs[0].shape
#         th, tw = self.size
#         if w == tw and h == th:
#             return inputs,target
#
#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
#         inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
#         return inputs, target[y1: y1 + th,x1: x1 + tw]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
            target[:,:,0] *= -1
        return inputs,target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
            target[:,:,1] *= -1
        return inputs,target


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs,target):
        applied_angle = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180
        diff_rad = diff*np.pi/180

        h, w, _ = target.shape

        warped_coords = np.mgrid[:w, :h].T + target
        warped_coords -= np.array([w / 2, h / 2])

        warped_coords_rot = np.zeros_like(target)

        warped_coords_rot[..., 0] = \
            (np.cos(diff_rad) - 1) * warped_coords[..., 0] + np.sin(diff_rad) * warped_coords[..., 1]

        warped_coords_rot[..., 1] = \
            -np.sin(diff_rad) * warped_coords[..., 0] + (np.cos(diff_rad) - 1) * warped_coords[..., 1]

        target += warped_coords_rot

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:, :, 0] = np.cos(angle1_rad)*target_[:, :, 0] + np.sin(angle1_rad)*target_[:, :, 1]
        target[:, :, 1] = -np.sin(angle1_rad)*target_[:, :, 0] + np.cos(angle1_rad)*target_[:, :, 1]
        return inputs, target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw),  min(w+tw, w),  max(0, -tw),  min(w-tw, w)
        y1, y2, y3, y4 = max(0, th),  min(h+th, h),  max(0, -th),  min(h-th, h)

        inputs[0] = inputs[0][y1:y2, x1:x2]
        inputs[1] = inputs[1][y3:y4, x3:x4]
        target = target[y1:y2, x1:x2]
        target[:, :, 0] += tw
        target[:, :, 1] += th

        return inputs, target


# class RandomColorWarp(object):
#     def __init__(self, mean_range=0, std_range=0):
#         self.mean_range = mean_range
#         self.std_range = std_range
#
#     def __call__(self, inputs, target):
#         random_std = np.random.uniform(-self.std_range, self.std_range, 3)
#         random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
#         random_order = np.random.permutation(3)
#
#         inputs[0] *= (1 + random_std)
#         inputs[0] += random_mean
#
#         inputs[1] *= (1 + random_std)
#         inputs[1] += random_mean
#
#         inputs[0] = inputs[0][:,:,random_order]
#         inputs[1] = inputs[1][:,:,random_order]
#
#         return inputs, target


if __name__ == '__main__':
    import SimpleITK as sitk
    # img = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix/Test/moving/20191112_luo_hua_shi.nii.gz')
    img = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix/Test/moving/20191112_luo_hua_shi.nii.gz')
    img_arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    img_arr = img_arr[np.newaxis, np.newaxis]
    flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/deformationField.mhd')
    # flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/predict_field_0825_1.5smooth.nii.gz')
    # flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/predict_field_0824.nii.gz')
    flow_arr = sitk.GetArrayFromImage(flow).transpose(3, 2, 1, 0)
    # print(flow_arr.max(), flow_arr.min())
    flow_arr = flow_arr[np.newaxis] / 128
    # a = torch.tensor(
    #     [1.004014, -0.000550, -0.000550, -0.009997, 1.106514, 0.015636, 0.000863, -0.005377, 1.029901, -5.548407,
    #      -13.871741, 3.536092])
    theta = np.load(r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix/Test/affine_param/20191112_luo_hua_shi.nii.gz')
    theta = torch.from_numpy(theta).to(torch.float32)
    # theta = torch.tensor([[[1.017788, 0.004833, -0.012667, 5.843587/256],
    #                        [0.005364, 1.076822, 0.000831, -14.079109/256],
    #                        [0.018305, -0.223271, 1.137476, -4.034460/256]]])
    # theta = theta[:, [2, 1, 0]]
    # theta = theta[:, :, [2, 1, 0, 3]]

    # theta = torch.tensor([[[1.137476, -0.223271, 0.018305, -4.034460 / 256],
    #                        [0.000831, 1.076822, 0.005364, -14.079109 / 256],
    #                        [-0.012667, 0.004833, 1.017788, 5.843587 / 256]
    #                        ]])

    # theta = torch.tensor([[[1.017788, 0.004833, -0.012667, 1],
    #                        [0.005364, 1.076822, 0.000831, -1],
    #                        [0.018305, -0.223271, 1.137476, -1]]])

    # seg_m = affine_transform(seg, theta, mask=True)
    # print(torch.unique(seg_m))

    spatial_transformer = SpatialTransformer(size=[256, 256, 256])
    pred = spatial_transformer(torch.from_numpy(img_arr).to(torch.float32), torch.from_numpy(flow_arr).to(torch.float32))
    # flow_arr = flow_arr[np.newaxis]
    # pred = affine_transform(torch.from_numpy(img_arr).to(torch.float32), theta)
    pred_image = sitk.GetImageFromArray(pred.numpy().squeeze().transpose(2, 1, 0))
    pred_image.CopyInformation(img)
    sitk.WriteImage(pred_image, r'/data/data1/zyh/Data/CTLung/deformationField_affine.nii.gz')