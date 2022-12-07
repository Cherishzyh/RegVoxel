import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys 
import os

sys.path.append('./')
from Networks.Subnets import VTN, VTNAffine, UNet, UNetNorm, UNetComplex, UNetComplexNorm
# from Networks.UNet import UNet
from Utils.Transform import SpatialTransformer, AffineTransform
from Utils.Loss import ImageLoss, FlowLoss

#
# class RegNet(nn.Module):
#     def __init__(self,
#                  shape,
#                  n_deform=1,
#                  n_recursive=1,
#                  n_classes=-1,
#                  affine_loss_dict={'corr':1},
#                  deform_loss_dict={'corr':1},
#                  deform_net='VTN',
#                  seg_input=False,
#                  show=False) -> None:
#         super().__init__()
#         self.show = show
#         self.imgs = []
#         self.segs = []
#
#         self.seg_input = seg_input
#         if seg_input:
#             in_channels = 2+n_classes
#         else:
#             in_channels = 2
#
#         self.affine = VTNAffine(shape, in_channels)
#
#         self.deforms = nn.ModuleList()
#         for _ in range(n_deform):
#             if deform_net == 'VTN':
#                 self.deforms.append(VTN(in_channels))
#             elif deform_net == 'UNet':
#                 self.deforms.append(UNet(in_channels))
#             else:
#                 raise ValueError('unfound deform network')
#
#         self.n_recursive = n_recursive
#
#         # loss
#         self.affine_criterion = AffineLoss(affine_loss_dict)
#         self.deform_criterion = DeformLoss(deform_loss_dict)
#         self.warpper = SpatialTransformer(shape)
#
#     def forward(self, img_m, img_f, seg_m, seg_f):
#         """_summary_
#
#         Args:
#             img_m (torch.float32): (b, 1, *)
#             img_f (torch.float32): (b, 1, *)
#             seg_m (torch.long): (b, n_classes, *)
#             seg_f (torch.long): (b, n_classes, *)
#
#         Returns:
#             _type_: _description_
#         """
#         print(self.affine)
#         print(self.deforms)
#         loss_dict = None
#         if self.show:
#             self.record(img_m, seg_m, init=True)
#
#         # affine
#         if self.seg_input:
#             theta = self.affine(torch.cat([img_m, seg_m], dim=1), img_f)
#         else:
#             theta = self.affine(img_m, img_f)
#         img_m = AffineTransform(img_m, theta)
#         seg_m = AffineTransform(seg_m, theta, mask=True)
#
#         loss_dict = self.add_loss(loss_dict,
#                                 self.affine_criterion(img_f, img_m, theta, seg_f, seg_m))
#         if self.show:
#             self.record(img_m, seg_m)
#
#         for _ in range(self.n_recursive):
#             # deforms
#             for deform_net in self.deforms:
#                 if self.seg_input:
#                     flow = deform_net(torch.cat([img_m, seg_m], dim=1), img_f)
#                 else:
#                     flow = deform_net(img_m, img_f)
#                 img_m = self.warpper(img_m, flow)
#                 seg_m = self.warpper(seg_m, flow, mask=True)
#                 loss_dict = self.add_loss(loss_dict,
#                                     self.deform_criterion(img_f, img_m, flow, seg_f, seg_m))
#                 if self.show:
#                     self.record(img_m, seg_m)
#
#         return img_m, loss_dict, seg_m
#
#     def record(self, img_m, seg_m, init=False):
#         if init:
#             self.imgs = []
#             self.segs = []
#
#         seg_m = torch.argmax(seg_m, dim=1, keepdim=True)
#         self.imgs.append(img_m)
#         self.segs.append(seg_m)
#
#     def add_loss(self, loss_dict, new_loss_dict):
#         if loss_dict is None:
#             return new_loss_dict
#         else:
#             for loss_name in new_loss_dict.keys():
#                 if loss_name in loss_dict.keys():
#                     loss_dict[loss_name] += new_loss_dict[loss_name]
#                 else:
#                     loss_dict[loss_name] = new_loss_dict[loss_name]
#             return loss_dict
#

class SimpleRegNet(nn.Module):
    def __init__(self,
                 shape,
                 n_deform=1,
                 n_recursive=1,
                 in_channels=2,
                 affine_loss_dict={'corr': 1},
                 image_loss_dict={'corr': 1},
                 flow_loss_dict={'mse': 1},
                 deform_net='VTN',
                 is_affine=False,
                 is_min=True,
                 is_box=False) -> None:
        super().__init__()
        self.imgs = []
        self.segs = []
        self.is_affine = is_affine
        self.is_min = is_min
        self.is_box = is_box

        if self.is_affine:
            self.affine = VTNAffine(shape, in_channels)

        self.deforms = nn.ModuleList()
        for _ in range(n_deform):
            if deform_net == 'VTN':
                self.deforms.append(VTN(in_channels))
            elif deform_net == 'UNet':
                self.deforms.append(UNet(in_channels))
            elif deform_net == 'UNetComplex':
                self.deforms.append(UNetComplex(in_channels))
            elif deform_net == 'UNetNorm':
                self.deforms.append(UNetNorm(in_channels))
            elif deform_net == 'UNetComplexNorm':
                self.deforms.append(UNetComplexNorm(in_channels))
            else:
                raise ValueError('unfound deform network')

        self.n_recursive = n_recursive

        # loss
        # self.affine_criterion = AffineLoss(affine_loss_dict)
        self.image_criterion = ImageLoss(image_loss_dict)
        self.flow_criterion = FlowLoss(flow_loss_dict)
        self.warpper = SpatialTransformer(shape)

    def forward(self, moving_image, fixed_image, volume_per, moving_mask, label_flow, affine_param=0, is_show_model=False, flow_compute='add'):
        """_summary_

        Args:
            moving_image (torch.float32): (b, 1, *)
            volume_per (torch.float32): (b, 1, *)
        Returns:
            _type_: _description_
        """
        if is_show_model:
            print(self.affine)
            print(self.deforms)
        loss_dict = None
        flow_list = []
        if self.is_affine:
            theta = self.affine(moving_image, volume_per)
            moving_image = AffineTransform(moving_image, theta)
            loss_dict = self.add_loss(loss_dict, self.affine_criterion(fixed_image, moving_image, theta, affine_param))

        for _ in range(self.n_recursive):
            for deform_num, deform_net in enumerate(self.deforms):
                flow = deform_net(moving_image, volume_per)
                flow_list.append(flow)
                moving_image = self.warpper(moving_image, flow, is_min=self.is_min)
                # mask 做变换
                loss_dict = self.add_loss(loss_dict, self.image_criterion(fixed_image, moving_image, seg_m=moving_mask, is_box=self.is_box))

        final_flow = torch.zeros_like(flow_list[0])
        if flow_compute == 'add':
            for idx, flow in enumerate(flow_list):
                final_flow = final_flow + flow
            loss_dict = self.add_loss(loss_dict, self.flow_criterion(final_flow, label_flow))
        elif flow_compute == 'single':
            for idx, flow in enumerate(flow_list):
                final_flow = final_flow + flow
                loss_dict = self.add_loss(loss_dict, self.flow_criterion(flow, label_flow))
        elif flow_compute == 'add_step_by_step':
            for idx, flow in enumerate(flow_list):
                weight = 0.25*(idx+1)
                final_flow = final_flow + flow
                loss_dict = self.add_loss(loss_dict, self.flow_criterion(final_flow, label_flow, weight=weight))
        if self.is_affine:
            return moving_image, final_flow, theta, loss_dict
        else:
            return moving_image, final_flow, loss_dict

    def add_loss(self, loss_dict, new_loss_dict):
        if loss_dict is None:
            return new_loss_dict
        else:
            for loss_name in new_loss_dict.keys():
                if loss_name in loss_dict.keys():
                    loss_dict[loss_name] += new_loss_dict[loss_name]
                else:
                    loss_dict[loss_name] = new_loss_dict[loss_name]
            return loss_dict


if __name__ == '__main__':
    device = 'cuda:5'
    moving_image = torch.randn([1, 1, 256, 256, 256]).to(device)
    volume_per = torch.randn([1, 1, 256, 256, 256]).to(device)
    label_flow = torch.randn([1, 3, 256, 256, 256]).to(device)
    # img_m = torch.randn([1, 1, 128, 128, 128]).to(device)
    # img_f = torch.randn([1, 1, 128, 128, 128]).to(device)
    # seg_m = torch.zeros([1, 1, 128, 128, 128]).to(device)
    # seg_f = torch.ones([1, 1, 128, 128, 128]).to(device)
    
    simple_regnet = SimpleRegNet([256, 256, 256], 3).to(device)
    # print(RegNet)
    img, loss = simple_regnet(moving_image, volume_per, label_flow)
    print(img.shape, loss)
