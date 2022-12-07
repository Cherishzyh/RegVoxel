""" https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py """
import os
import sys
import time
import datetime
import warnings
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from Dataset.DataSet import NifitDataSet
from models.CGAN import CGANModel
from Networks.Subnets import VTN, NLayerDiscriminator, UNet, UNetNorm, UNetComplex, UNetComplexNorm
from Networks.Setting import get_norm_layer

sys.path.append('./')
warnings.filterwarnings("ignore")

def time_printer():
    now = datetime.datetime.now()
    ts = now.strftime('%Y-%M-%D %H:%M:%S')
    print('do func time : ', ts)

class Options():
    def __init__(self):
        super(Options, self).__init__()

    def Initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--data_path', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData', help='images path')
        parser.add_argument('--train_csv', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData/train.csv', help='Train images path')
        parser.add_argument('--val_csv', type=str, default=r'/data/data1/zyh/Data/CTLung/CropData/val.csv', help='Validation images path')
        ################################################################################################################################################################
        parser.add_argument('--moving_key', type=str, default='inhale')
        parser.add_argument('--fixed_key', type=str, default='rigid')
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--epoch_for_D', type=int, default=0)
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
        parser.add_argument('--gpu_ids', default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--suffix', default='1118_CGAN_2D', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--weights_path', type=str, default=r'', help='Pre-trained models')
        # parser.add_argument('--weights_path', type=str,
        #                     default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_0929_UNetComplex/48-0.065568.pt',
        #                     help='Pre-trained models')
        # parser.add_argument('--G_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1031_CGAN_2D/netG_257.pt')
        # parser.add_argument('--D_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1031_CGAN_2D/netD_257.pt')
        # parser.add_argument('--D2_weights_path', type=str, default=r'/data/data1/zyh/Model/RespiratoryCompensation/RespiratoryCompensation_1031_CGAN_2D/netD2_257.pt')
        parser.add_argument('--G_weights_path', type=str, default=r'')
        parser.add_argument('--D_weights_path', type=str, default=r'')
        parser.add_argument('--D2_weights_path', type=str, default=r'')

        parser.add_argument('--target_of_D', type=list, default=['image', 'field'], help='the target of discriminator: image or field')

        parser.add_argument('--deform_losses', type=list, default=['l1_flow', 'smooth'], help='Classes of loss functions for deform flow')
        parser.add_argument('--weights_deform_losses', type=list, default=[75, 75], help='weight for loss functions for deform flow')
        parser.add_argument('--image_losses', type=list, default=['l1'], help='Classes of loss functions for deform images')
        parser.add_argument('--weights_image_losses', type=list, default=[100], help='weight for loss functions for deform images')
        parser.add_argument('--mask_losses', type=list, default=[], help='Classes of loss functions for deform mask')
        parser.add_argument('--weights_mask_losses', type=list, default=[], help='weight for loss functions for deform mask')
        parser.add_argument('--field_compute_way', type=str, default='single', help='the computation way of field loss: single, add, add_step_by_step')
        ################################################################################################################################################################

        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--patch_size', default=[256, 256, 256], help='size of the patches extracted from the image')
        parser.add_argument('--train_batches', type=int, default=1, help='train batches')
        parser.add_argument('--val_batches', type=int, default=1, help='val batches')
        parser.add_argument('--is_min', default=True, help='whether to normalize the deformation field'
                                                           'true represents the deformation field divided by half Shape')
        parser.add_argument('--n_deform', type=int, default=1, help='elastic deformations times')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--deform_net', type=str, default='UNetComplex', help='deform network')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')

        parser.add_argument('--name', type=str, default='RespiratoryCompensation', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='cgan', help='chooses which model to use. cycle_gan')

        parser.add_argument('--checkpoints_dir', type=str, default='/data/data1/zyh/Model/RespiratoryCompensation', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
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
        file_name = os.path.join(expr_dir, 'opt.txt')
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


def Train():
    opt = Options().Initialize()

    # Get Train dataset
    train_dataset = NifitDataSet(data_path=opt.data_path,
                                 csv_path=opt.train_csv,
                                 transforms=None,
                                 shuffle=True,
                                 is_min=opt.is_min,
                                 moving_key=opt.moving_key,
                                 fixed_key=opt.fixed_key)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=True)
    opt.train_batches = np.ceil(len(train_dataset) / opt.batch_size)
    print("train case: {}".format(len(train_dataset)))

    # Get Validation dataset
    val_dataset = NifitDataSet(data_path=opt.data_path,
                               csv_path=opt.val_csv,
                               transforms=None,
                               shuffle=True,
                               is_min=opt.is_min,
                               moving_key=opt.moving_key,
                               fixed_key=opt.fixed_key)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)
    opt.val_batches = np.ceil(len(val_dataset) / opt.batch_size)
    print("val case: {}".format(len(val_dataset)))

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
                                        norm_layer=get_norm_layer(opt.norm), use_sigmoid=False)
        elif opt.target_of_D[0] == 'field':
            D_net = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                        norm_layer=get_norm_layer(opt.norm), use_sigmoid=False)
        else:
            raise ValueError
    if len(opt.target_of_D) > 1:
        if opt.target_of_D[1] == 'image':
            D2_net = NLayerDiscriminator(opt.input_nc + 1, opt.ndf, n_layers=opt.n_layers_D,
                                         norm_layer=get_norm_layer(opt.norm), use_sigmoid=False)
        elif opt.target_of_D[1] == 'field':
            D2_net = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
                                         norm_layer=get_norm_layer(opt.norm), use_sigmoid=False)
        else:
            raise ValueError
    model = CGANModel(opt,
                      net_deform=deform_net,
                      net_D=D_net,
                      net_D2=D2_net)
    model.ModelPrepare()
    # TODO: multi_gpu train

    for epoch in range(opt.start_epoch + 1, 1500):
        model.SetTrain()
        train_iter_start_time = time.time()  # timer for computation per iteration
        if opt.epoch_for_D == 0:
            update_d = True
        elif epoch % opt.epoch_for_D == 1:
            update_d = True
        else:
            update_d = False
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            model.SetInput(data)  # unpack data from dataset and apply preprocessing
            model.Train(update_d)
        train_iter_end_time = time.time()
        train_time = train_iter_end_time - train_iter_start_time

        model.SetValidation()
        val_iter_start_time = time.time()
        with torch.no_grad():
            for i, data in enumerate(val_loader):  # inner loop within one epoch
                model.SetInput(data)  # unpack data from dataset and apply preprocessing
                model.Validation()
        val_iter_end_time = time.time()
        val_time = val_iter_end_time - val_iter_start_time

        print('Epoch: {}, Time of train: {:3f}, Time of val: {:3f}'.format(epoch, train_time, val_time))
        if update_d: print('Update parameters of D')
        print('train:', end=' ')

        for key in model.train_loss_G:
            print('{}: {:3f}'.format(key, model.train_loss_G[key] / opt.train_batches), end=' ')
        print('\t', end=' ')
        if len(opt.target_of_D) > 0:
            for key in model.train_loss_D:
                print('{}: {:3f}'.format(key, model.train_loss_D[key] / opt.train_batches), end=' ')
        if len(opt.target_of_D) > 1:
            for key in model.train_loss_D2:
                print('{}: {:3f}'.format(key, model.train_loss_D2[key] / opt.train_batches), end=' ')
        print()

        print('val:', end=' ')
        for key in model.validation_loss_G:
            print('{}: {:3f}'.format(key, model.validation_loss_G[key] / opt.val_batches), end=' ')
        print('\t', end=' ')
        if len(opt.target_of_D) > 0:
            for key in model.validation_loss_D:
                print('{}: {:3f}'.format(key, model.validation_loss_D[key] / opt.val_batches), end=' ')
        if len(opt.target_of_D) > 1:
            for key in model.validation_loss_D2:
                print('{}: {:3f}'.format(key, model.validation_loss_D2[key] / opt.val_batches), end=' ')
        print()

        is_break = model.SaveAndUpdate(epoch, update_d, opt.val_batches)
        if is_break: break


if __name__ == '__main__':
    while True:
        time.sleep(0)
        time_printer()
        Train()
        break


        




# /opt/conda/bin/python3.6 -u /data/data1/zyh/Project/RegVoxel/CGanTrain.py

