import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from monai.losses.image_dissimilarity import LocalNormalizedCrossCorrelationLoss


sys.path.append('./')

### Main Loss###
class ThetaLoss(nn.Module):
    def __init__(self, weight) -> None:
        """
        Args:
            weight (dict): {'loss_name': weight}
        """
        super().__init__()

        self.weight = weight
        self.mse = nn.MSELoss()
        self.ortho = OrthoLoss()

    def forward(self, theta, affine_param):
        loss_dict = {}
        loss_names = self.weight.keys()
        if len(loss_names) > 0:
            if 'ortho' in loss_names:
                loss_dict['ortho'] = self.ortho(theta) * self.weight['ortho']
            if 'mse_theta' in loss_names:
                loss_dict['mse_theta'] = self.mse(theta, affine_param) * self.weight['mse_theta']
        return loss_dict


class ImageLoss(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()

        self.weight = weight
        self.corr = CorrLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.smoothl1 = nn.SmoothL1Loss()
        self.ncc = LocalNormalizedCrossCorrelationLoss()

    def _GetBoundingBox(self, mask, margin=[5, 5, 5]):
        """
        get the bounding box of nonzero region in shape(b, c, x, y, z)
        """
        assert len(mask.shape) == 5, print('Check shape of mask! Only accept (b, c, x, y, z)')
        if (margin is None):
            margin = [0] * 3

        idx_min = []
        idx_min.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 3, 4)))[0]))
        idx_min.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 2, 4)))[0]))
        idx_min.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 2, 3)))[0]))
        idx_max = []
        idx_max.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 3, 4)))[-1]))
        idx_max.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 2, 4)))[-1]))
        idx_max.append(int(torch.nonzero(torch.sum(mask, dim=(0, 1, 2, 3)))[-1]))

        for i in range(len(idx_min)):
            idx_min[i] = max(idx_min[i] - margin[i], 0)
            idx_max[i] = min(idx_max[i] + margin[i], mask.shape[i + 2])
        self.x_idx, self.y_idx, self.z_idx = torch.meshgrid(torch.arange(idx_min[0], idx_max[0]),
                                                            torch.arange(idx_min[1], idx_max[1]),
                                                            torch.arange(idx_min[2], idx_max[2]))
        return self.x_idx, self.y_idx, self.z_idx

    def _Padding(self, image):
        zero_image = torch.zeros_like(image)
        zero_image[:, :, self.x_idx, self.y_idx, self.z_idx] = 1
        image = image * zero_image
        return image

    def forward(self, img_f, img_m, seg_m=None, is_box=False, weight=None):
        if weight == None: weight = 1
        if not isinstance(img_m, list):
            assert isinstance(img_m, torch.Tensor)
            img_m = [img_m]

        if is_box and seg_m != None:
            self._GetBoundingBox(seg_m)
            img_m = [self._Padding(img) for img in img_m]
            img_f = self._Padding(img_f)

        # loss_dict = dict(zip(self.weight.keys(), len(self.weight.keys())*[0]))
        loss_dict = {}
        loss_names = self.weight.keys()
        if len(loss_names) > 0:
            if 'corr' in loss_names:
                loss_dict['corr'] = sum([self.corr(img_f, img) * self.weight['corr'] for img in img_m]) * weight
            if 'mse' in loss_names:
                loss_dict['mse'] = sum([self.mse(img_f, img) * self.weight['mse'] for img in img_m]) * weight
            if 'l1' in loss_names:
                loss_dict['l1'] = sum([self.l1(img_f, img) * self.weight['l1'] for img in img_m]) * weight
            if 'smoothl1' in loss_names:
                loss_dict['smoothl1'] = sum([self.smoothl1(img_f, img) * self.weight['smoothl1'] for img in img_m]) * weight
            if 'ncc' in loss_names:
                loss_dict['ncc'] = sum([self.ncc(img_f, img) * self.weight['ncc'] for img in img_m]) * weight
            loss_dict['all'] = sum([loss_dict[x] for x in loss_names])
        return loss_dict


class FlowLoss(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()

        self.weight = weight

        self.corr = CorrLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.epe = EPE()
        self.smooth = SmoothLoss(penalty='l2')

    def forward(self, flow, label_flow, weight=None):
        loss_dict = {}
        loss_names = self.weight.keys()
        if weight == None: weight = 1
        if len(loss_names) > 0:
            if 'mse_flow' in loss_names:
                loss_dict['mse_flow'] = self.mse(flow, label_flow) * self.weight['mse_flow'] * weight
            if 'smooth' in loss_names:
                loss_dict['smooth'] = self.smooth(flow) * self.weight['smooth'] * weight
            if 'l1_flow' in loss_names:
                loss_dict['l1_flow'] = self.l1(flow, label_flow) * self.weight['l1_flow'] * weight
            if 'epe' in loss_names:
                loss_dict['epe'] = self.epe(flow, label_flow) * self.weight['epe'] * weight

            loss_dict['all'] = sum([loss_dict[x] for x in loss_names])
        return loss_dict


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class MaskLoss(nn.Module):
    def __init__(self, weight):
        super(MaskLoss, self).__init__()
        self.weight = weight

    def _GetCenter(self, mask):
        mask = torch.squeeze(mask)
        assert len(mask.shape) == 3
        center_0 = (torch.nonzero(torch.sum(mask, axis=(1, 2)))[0] + torch.nonzero(torch.sum(mask, axis=(1, 2)))[-1]) / 2
        center_1 = (torch.nonzero(torch.sum(mask, axis=(0, 2)))[0] + torch.nonzero(torch.sum(mask, axis=(0, 2)))[-1]) / 2
        center_2 = (torch.nonzero(torch.sum(mask, axis=(0, 1)))[0] + torch.nonzero(torch.sum(mask, axis=(0, 1)))[-1]) / 2
        return torch.tensor([[center_0, center_1, center_2]]).to(torch.float32)

    def DiceLoss(self, input, target):
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    def forward(self, input_mask, target_mask, input_nodule, target_nodule):
        loss_dict = {}
        loss_names = self.weight.keys()
        if len(loss_names) > 0:
            if 'mask_dice' in loss_names:
                loss_dict['mask_dice'] = self.DiceLoss(input_mask, target_mask) * self.weight['mask_dice']
            if torch.sum(input_nodule) > 0:
                if 'dice' in loss_names:
                    loss_dict['dice'] = self.DiceLoss(input_nodule, target_nodule) * self.weight['dice']
                if 'cdist' in loss_names:
                    center_input = self._GetCenter(input_nodule)
                    center_target = self._GetCenter(target_nodule)
                    loss = torch.cdist(center_input, center_target, p=2) * self.weight['cdist']
                    loss_dict['cdist'] = torch.squeeze(loss.to(input_nodule.device))

            loss_dict['all'] = sum([loss_dict[x] for x in loss_names])
        return loss_dict

### Sub Loss###
# Registration
class CorrLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
            
    def forward(self, img_f, img_m):
        flatten1 = img_f.view(img_f.shape[0], -1) # (b, h*w*d)
        flatten2 = img_m.view(img_m.shape[0], -1)

        mean1 = flatten1.mean(dim=1, keepdim=True)
        mean2 = flatten2.mean(dim=1, keepdim=True)
        var1 = flatten1.var(dim=1)
        var2 = flatten2.var(dim=1)

        cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        raw_loss = 1 - pearson_r
        raw_loss = torch.mean(raw_loss)
        
        return raw_loss
        

class OrthoLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, theta):
        """_summary_
        Args:
            theta (_type_): (b,3,4)
        """
        eps = 1e-6
        W = theta[:,:,:3]  # (b,3,3)
        
        sv = torch.linalg.svdvals(W) + eps  # (b,3)
        loss = -6 + torch.sum(sv.pow(2) +sv.pow(-2), dim=1)  # (b, )
        loss = loss.mean()
        return loss


class SmoothLoss(nn.Module):
    def __init__(self, penalty='l1') -> None:
        super().__init__()
        assert penalty in ['l1', 'l2']
        self.penalty = penalty
    
    def forward(self, flow):
        """_summary_
        Args:
            flow (_type_): (b, 3, h, w, d)
        """
        dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])  # (b, 3, h-1, w, d)
        dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])  # (b, 3, h, w-1, d)
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])  # (b, 3, h, w, d-1)

        if self.penalty == 'l2':
            dx = dx * dx
            dy = dy * dy
            dz = dz * dz
        
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        loss = d / 3.0
        return loss


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, input_flow, target_flow, sparse=False, mean=True):
        EPE_map = torch.norm(target_flow-input_flow, p=2, dim=1)
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with all flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0) & (target_flow[:, 2] == 0)
            EPE_map = EPE_map[~mask]
        if mean:
            return EPE_map.mean()
        else:
            return EPE_map.sum()/batch_size

### TEST ###
def TestLoss():
    import SimpleITK as sitk
    flow = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/predict_field_0825.nii.gz')
    flow_arr = sitk.GetArrayFromImage(flow).transpose(3, 2, 1, 0)
    flow_arr = flow_arr[np.newaxis]
    flow_target = sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/Test/Train/field/20191118_wu_jia_na.nii.gz')
    flow_target_arr = sitk.GetArrayFromImage(flow_target).transpose(3, 2, 1, 0)
    flow_target_arr = flow_target_arr[np.newaxis]
    loss1 = nn.L1Loss()
    loss2 = nn.MSELoss()
    loss3 = nn.MSELoss()
    smoothl1 = nn.SmoothL1Loss()
    smooth = SmoothLoss(penalty='l2')
    print('L1 loss', end=': ')
    print(loss1(torch.from_numpy(flow_target_arr), torch.from_numpy(flow_arr)),
          loss1(torch.from_numpy(flow_target_arr)/128, torch.from_numpy(flow_arr)/128))
    print('L2 loss', end=': ')
    print(loss2(torch.from_numpy(flow_target_arr), torch.from_numpy(flow_arr)),
          loss2(torch.from_numpy(flow_target_arr)/128, torch.from_numpy(flow_arr)/128))
    print('mse loss', end=': ')
    print(loss3(torch.from_numpy(flow_target_arr), torch.from_numpy(flow_arr)),
          loss3(torch.from_numpy(flow_target_arr)/128, torch.from_numpy(flow_arr)/128))
    print('Smooth L1 loss', end=': ')
    print(smoothl1(torch.from_numpy(flow_target_arr), torch.from_numpy(flow_arr)),
          smoothl1(torch.from_numpy(flow_target_arr)/128, torch.from_numpy(flow_arr)/128))
    # flow_arr = flow_arr[np.newaxis] / 128
    print('Smooth', end=': ')
    print(smooth(torch.from_numpy(flow_arr)), smooth(torch.from_numpy(flow_arr)/128))


def TestMaskLoss():
    import SimpleITK as sitk
    from MeDIT.Normalize import Normalize01
    data_root = r'/data/data1/zyh/Data/CTLung/OrganizeDataElastix/Test'
    pred_folder = os.path.join(data_root, 'moving')
    target_folder = os.path.join(data_root, 'fixed')
    mask_folder = os.path.join(data_root, 'moving_mask')
    loss1 = torch.nn.MSELoss()
    deform_loss = ['mse', 'mse_flow']
    deform_loss_dict = dict(zip(deform_loss, [1, 1]))
    for case in sorted(os.listdir(pred_folder)):
        pred_path = os.path.join(pred_folder, case)
        target_path = os.path.join(target_folder, case)
        mask_path = os.path.join(mask_folder, case)

        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).transpose(2, 1, 0)
        pred = pred[np.newaxis, np.newaxis]
        pred = Normalize01(pred)
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path)).transpose(2, 1, 0)
        target = target[np.newaxis, np.newaxis]
        target = Normalize01(target)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).transpose(2, 1, 0)
        mask = mask[np.newaxis, np.newaxis]

        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)

        break


if __name__ == '__main__':
    # label = torch.tensor([[5, 8, 3], [4, 2, 6], [2, 6, 9]]).to(torch.float32)
    # input = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).to(torch.float32)
    # # pred1 = torch.tensor([[1, 3, 2], [2, 3, 1], [1, 2, 5]]).to(torch.float32)
    # # pred2 = torch.tensor([[2, 5, 1], [1, -1, 3], [1, 2, 3]]).to(torch.float32)
    # # pred3 = torch.tensor([[2, 0, 0], [1, 0, 2], [0, 2, 1]]).to(torch.float32)
    # # pred = pred1 + pred2 + pred3
    # # loss = torch.nn.MSELoss()
    # # print(loss(label, pred1))
    # # print(loss(label, pred2))
    # # print(loss(label, pred3))
    # # print(loss(label, pred))
    # # TestMaskLoss()
    # loss = CorrLoss()
    # print(loss(input, label))
    # label = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]).to(torch.float32)
    # input = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    #                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).to(torch.float32)
    # loss = MaskLoss(weight={'dice': 1, 'cdist': 1})
    # torch.cdist(label, input, p=2)
    # print(torch.cdist(label, input, p=2))
    # print(loss(input, label))
    import SimpleITK as sitk
    label = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/deform_flow.nii.gz'))
    label = torch.from_numpy(label[np.newaxis].transpose(0, 4, 3, 2, 1))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(r'/data/data1/zyh/Data/CTLung/CropData/20191115_tan_fen/field_1117.nii.gz'))
    pred = torch.from_numpy(pred[np.newaxis].transpose(0, 4, 3, 2, 1))
    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    print(EPE(label, pred), l1(label, pred), l2(label, pred))
    import torch.nn.functional as F
    F.l1_loss()


