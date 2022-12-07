import os
import sys
import torch
import shutil
import random
import numpy as np
import argparse
from collections import OrderedDict, Counter
import models
from Networks.Setting import get_scheduler

sys.path.append('../ModelSetting/')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, evaluation='min', delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.evaluation = evaluation
        self.is_save = False

        # self.store_name = store_path

    def __call__(self, val_loss):
        if self.evaluation == 'min':
            score = val_loss
            if self.best_score is None:
                self.best_score = score
                self.is_save = True
                # self.save_checkpoint(val_loss, model, store_key)
            elif score > self.best_score + self.delta:
                self.is_save = False
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.is_save = True
                # self.save_checkpoint(val_loss, model, store_key)
                self.counter = 0

        elif self.evaluation == 'max':
            score = val_loss
            if self.best_score is None:
                self.best_score = score
                self.is_save = True
                # self.save_checkpoint(val_loss, model, store_key)
            elif score < self.best_score + self.delta:
                self.is_save = False
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.is_save = True
                # self.save_checkpoint(val_loss, model, store_key)
                self.counter = 0


def mkdirs(paths):
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class Options():
    def __init__(self):
        self.initialized = False
        self.isTrain = False

    def Initialize(self, parser, is_train):
        parser.add_argument('--train_path', type=str, default='/data/data1/zyh/Data/CTLung/Test/Train', help='Train images path')
        parser.add_argument('--val_path', type=str, default='/data/data1/zyh/Data/CTLung/Test/Val', help='Validation images path')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--patch_size', default=[256, 256, 256], help='size of the patches extracted from the image')
        parser.add_argument('--is_train', default=is_train, help='if is train or test')
        parser.add_argument('--is_affine', default=True, help='whether the model adds an affine change module')
        parser.add_argument('--is_min', default=True, help='whether to normalize the deformation field'
                                                           'true represents the deformation field divided by half Shape')
        parser.add_argument('--n_deform', type=int, default=3, help='elastic deformations times')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--deform_net', type=str, default='VTN', help='VTN | UNet')
        parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--netG', type=str, default='resnet_6blocks', help='selects model to use for netG. Look on Networks3D to see the all list')

        parser.add_argument('--gpu_ids', default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='RespiratoryCompensation', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='cgan', help='chooses which model to use. cycle_gan')

        parser.add_argument('--checkpoints_dir', type=str, default='/data/data1/zyh/ModelSetting/RespiratoryCompensation/checkpoints', help='models are saved here')
        parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='0913', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        self.initialized = True

        if is_train:
            parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model')
            parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
            parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
            parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
            parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
            parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
            parser.add_argument('--lr_policy', type=str, default='plateau', help='learning rate policy: lambda|step|plateau|cosine')
            parser.add_argument('--is_box', default=False, help='Select loss calculation method, '
                                                                'true means to calculate only the loss function inside mask, '
                                                                'false means to calculate the loss of the whole picture')
            # parser.add_argument('--affine_losses', type=list, default=['mse_theta', 'ortho'], help='Classes of loss functions for affine theta')
            # parser.add_argument('--weights_affine_losses', type=list, default=[0.5, 0.5], help='weight for loss functions for affine theta')
            parser.add_argument('--affine_losses', type=list, default=['mse_theta'], help='Classes of loss functions for affine theta')
            parser.add_argument('--weights_affine_losses', type=list, default=[1], help='weight for loss functions for affine theta')
            parser.add_argument('--deform_losses', type=list, default=['mse_flow', 'smooth'], help='Classes of loss functions for deform flow')
            parser.add_argument('--weights_deform_losses', type=list, default=[1, 1], help='weight for loss functions for deform flow')
            parser.add_argument('--image_losses', type=list, default=['corr', 'mse'], help='Classes of loss functions for deform images')
            parser.add_argument('--weights_image_losses', type=list, default=[0.5, 0.5], help='weight for loss functions for deform images')
        else:
            parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
            parser.add_argument("--result", type=str, default='./Data_folder/test/images/result_0.nii', help='path to the .nii result to save')
            parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
            parser.set_defaults(model='test')

        return parser

    def GatherOptions(self, is_train):
        # initialize parser with basic options
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.Initialize(parser, is_train)

            # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.GetOptionSetter(model_name)
        parser = model_option_setter(parser, opt.is_train)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser

        return parser.parse_args()

    def PrintOptions(self, opt):
        message = ''
        message += '----------------- ModelSetting ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def Parse(self, is_train):
        opt = self.GatherOptions(is_train)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.PrintOptions(opt)

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
        self.opt = opt
        return self.opt


class BaseModel():
    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def ModifyCommandlineOptions(parser, is_train):
        return parser

    def Name(self):
        return 'BaseModel'

    def Initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.image_paths = []
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        else:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []

    def SetInput(self, input):
        pass

    def Forward(self):
        pass

    def Setup(self, opt, parser=None):
        if self.is_train:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.is_train or opt.continue_train:
            self.LoadNetworks(opt.which_epoch)
        self.PrintNetworks(opt.verbose)

    # make models train mode during train time
    def Train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # make models eval mode during test time
    def Eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def Test(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
        with torch.no_grad():
            self.Forward()

    # get image paths
    def GetImagePaths(self):
        return self.image_paths

    # update learning rate (called once every epoch)
    def UpdateLearningRate(self, val_loss):
        for scheduler in self.schedulers:
            scheduler.step(val_loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def GetCurrentVisuals(self):
        '''待定'''

        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def GetCurrentLosses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                loss = getattr(self, name)
                if isinstance(loss, dict):
                    for key in loss.keys():
                        errors_ret[key] = float(loss[key])
                else:
                    errors_ret[name] = float(loss)
        return errors_ret

    # save models to the disk
    def SaveNetworks(self, store_key):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_{:.5f}_net_{}.pth'.format(*store_key, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __PatchInstanceNormStateDict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__PatchInstanceNormStateDict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def LoadNetworks(self, which_epoch):
        '''
        name is the name of (G_A, G_B, D_A, D_B)
        '''
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__PatchInstanceNormStateDict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def PrintNetworks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def SetRequiresGrad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def EarlyStopping(self, store_key, is_save=True):
        if not isinstance(store_key, list) and not isinstance(store_key, tuple):
            store_key = [store_key]
        if is_save:
            self.SaveNetworks(store_key)


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images