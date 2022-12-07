""" https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py """
import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import median_filter

import torch

from Dataset.DataSet import NifitDataSetTest
from models.CGAN import CGANModel
from Utils.Transform import AffineTransform, SpatialTransformer
from Networks.Subnets import VTN, NLayerDiscriminator, UNet, UNetNorm, UNetComplex, UNetComplexNorm
from Networks.Setting import get_norm_layer

sys.path.append('./')


class Options():
    def __init__(self):
        super(Options, self).__init__()

    def OptionalParameter(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--test_csv', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData/test.csv',
                            help='Test images path')
        ################################################################################################################################################################
        parser.add_argument('--moving_key', type=str, default='inhale')
        parser.add_argument('--fixed_key', type=str, default='rigid')
        parser.add_argument('--gpu_ids', default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--suffix', default='1118_CGAN_2D_pretrained', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--weights_path', type=str, default=r'', help='Pre-trained models')

        parser.add_argument('--G_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1118_CGAN_2D_pretrained/netG_1499.pt')
        parser.add_argument('--D_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1118_CGAN_2D_pretrained/netD_1499.pt')
        parser.add_argument('--D2_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1118_CGAN_2D_pretrained/netD2_1499.pt')
        parser.add_argument('--test_of_D', default=True, help='if True, get discriminator pred')
        parser.add_argument('--target_of_D', type=list, default=['image', 'field'], help='the target of discriminator: image or field')
        return parser

    def Initialize(self):
        parser = self.OptionalParameter()
        parser.add_argument('--data_path', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData', help='images path')
        parser.add_argument('--save_path', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData', help='pred path')
        ################################################################################################################################################################
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--patch_size', default=[256, 256, 256],
                            help='size of the patches extracted from the image')
        parser.add_argument('--is_affine', default=False, help='whether the model adds an affine change module')
        parser.add_argument('--is_min', default=True, help='whether to normalize the deformation field'
                                                           'true represents the deformation field divided by half Shape')
        parser.add_argument('--n_deform', type=int, default=1, help='elastic deformations times')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--deform_net', type=str, default='UNetComplex', help='deform network')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')

        parser.add_argument('--name', type=str, default='RespiratoryCompensation',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='cgan', help='chooses which model to use. cycle_gan')

        parser.add_argument('--checkpoints_dir', type=str, default='/data/data1/zyh/Model/RespiratoryCompensation',
                            help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--is_box', default=False, help='Select loss calculation method, '
                                                            'true means to calculate only the loss function inside mask, '
                                                            'false means to calculate the loss of the whole picture')

        self._PrintOptions(parser)
        opt, _ = parser.parse_known_args()
        self._SetGPUID(opt)
        return opt

    def _PrintOptions(self, parser):
        opt, _ = parser.parse_known_args()
        message = ''
        message += '----------------- ModelSetting ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        parser.add_argument('--expr_dir', type=str, default=expr_dir, help='save folder')
        if not os.path.exists(expr_dir): os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_test.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        opt_file.close()

        return message

    def _SetGPUID(self, opt):
        # set gpu ids
        str_ids = list(opt.gpu_ids)
        str_ids = [id for id in str_ids if id != ',']
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])


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


if __name__ == '__main__':
    from NewTest import DataTools

    opt = Options().Initialize()
    data_tool = DataTools()
    test_dataset = NifitDataSetTest(data_path=opt.data_path, csv_path=opt.test_csv, transforms=None, shuffle=False,
                                    is_min=opt.is_min)
    # define deform model
    if opt.deform_net == 'VTN':
        deform_net = VTN(opt.input_nc)
    elif opt.deform_net == 'UNet':
        deform_net = UNet(opt.input_nc)
    elif opt.deform_net == 'UNetComplex':
        deform_net = UNetComplex(opt.input_nc)
    elif opt.deform_net == 'UNetNorm':
        deform_net = UNetNorm(opt.input_nc)
    elif opt.deform_net == 'UNetComplexNorm':
        deform_net = UNetComplexNorm(opt.input_nc)
    else:
        raise ValueError('unfound deform network')

    D_net, D2_net = None, None
    if len(opt.target_of_D) > 0:
        if opt.target_of_D[0] == 'image':
            D_net = NLayerDiscriminator(opt.input_nc + 1, opt.ndf, n_layers=opt.n_layers_D,
                                        norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
        elif opt.target_of_D[0] == 'field':
            D_net = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                        norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
        else:
            raise ValueError
    if len(opt.target_of_D) > 1:
        if opt.target_of_D[1] == 'image':
            D2_net = NLayerDiscriminator(opt.input_nc + 1, opt.ndf, n_layers=opt.n_layers_D,
                                         norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
        elif opt.target_of_D[1] == 'field':
            D2_net = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                         norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
        else:
            raise ValueError

    model = CGANModel(opt,
                      net_deform=deform_net,
                      net_D=D_net,
                      net_D2=D2_net)

    model.ModelPrepare(is_train=False)
    model.SetValidation()

    pbar = tqdm(total=len(pd.read_csv(opt.test_csv, index_col=0).index), ncols=80)
    pred_fake_1_list, pred_real_1_list, pred_fake_2_list, pred_real_2_list = [], [], [], []
    with torch.no_grad():
        for index, case in enumerate(test_dataset.case_list):
            data = test_dataset.getitem(index)
            moving = data[0]
            fixed = data[1]
            field = data[2]
            volume = data[3]
            ref_field = data[4][0]
            ref_moving = data[4][1]
            ref_fixed = data[4][2]
            moving_nodule = data[5]
            fixed_nudule = data[6]
            moving_mask = data[7]
            fixed_mask = data[8]
            if not torch.sum(moving_nodule) == 0: continue

            model.SetInput([moving, fixed, field, volume, moving_mask, fixed_mask, moving_nodule, fixed_nudule])  # unpack data from dataset and apply preprocessing
            if opt.test_of_D:
                if D_net:
                    pred_fake_1, pred_real_1 = model.TestDiscriminator(D_net, opt.target_of_D[0])
                    pred_fake_1_list.append(pred_fake_1)
                    pred_real_1_list.append(pred_real_1)
                if D2_net:
                    pred_fake_2, pred_real_2 = model.TestDiscriminator(D2_net, opt.target_of_D[1])
                    pred_fake_2_list.append(pred_fake_2)
                    pred_real_2_list.append(pred_real_2)

            else:
                flow_list, transform_image_list, transform_mask_list, transform_nodule_list = model.Validation(True)

                image_pred = data_tool.Tensor2Numpy(transform_image_list[-1])
                mask_pred = torch.round(transform_mask_list[-1]).cpu().detach().numpy().squeeze()
                mask_pred = median_filter(mask_pred, size=3)
                if torch.sum(moving_nodule) > 0:
                    nodule_pred = torch.round(transform_nodule_list[-1]).cpu().detach().numpy().squeeze()
                    nodule_pred = median_filter(nodule_pred, size=3)
                    data_tool.Numpy2Image(nodule_pred, ref_moving,
                                          save_folder=os.path.join(opt.save_path, case, 'nodule_{}.nii.gz'.format(opt.suffix)))

                field_pred = data_tool.Tensor2Numpy(flow_list[-1])
                if opt.is_min:
                    field_pred = field_pred * 128
                data_tool.Numpy2Image(image_pred, ref_moving,
                                      save_folder=os.path.join(opt.save_path, case, 'exhale_{}.nii.gz'.format(opt.suffix)))
                data_tool.Numpy2Image(mask_pred, ref_moving,
                                      save_folder=os.path.join(opt.save_path, case, 'lung_{}.nii.gz'.format(opt.suffix)))
                data_tool.Numpy2Image(field_pred, ref_field, isVector=True,
                                      save_folder=os.path.join(opt.save_path, case, 'field_{}.nii.gz'.format(opt.suffix)))
            pbar.update()
        if opt.test_of_D:
            np.save(os.path.join(opt.save_path, 'fake_{}_{}_test_1499.npy'.format(opt.target_of_D[0], opt.suffix)), torch.cat(pred_fake_1_list, dim=1).cpu().numpy().squeeze())
            np.save(os.path.join(opt.save_path, 'real_{}_{}_test_1499.npy'.format(opt.target_of_D[0], opt.suffix)), torch.cat(pred_real_1_list, dim=1).cpu().numpy().squeeze())
            np.save(os.path.join(opt.save_path, 'fake_{}_{}_test_1499.npy'.format(opt.target_of_D[1], opt.suffix)), torch.cat(pred_fake_2_list, dim=1).cpu().numpy().squeeze())
            np.save(os.path.join(opt.save_path, 'real_{}_{}_test_1499.npy'.format(opt.target_of_D[1], opt.suffix)), torch.cat(pred_real_2_list, dim=1).cpu().numpy().squeeze())

    pbar.close()











