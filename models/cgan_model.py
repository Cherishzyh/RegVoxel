import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import ChainMap
from models import BaseModel
from Networks.Subnets import VTNAffine, VTN, NLayerDiscriminator
from Networks.UNet import UNet
from Networks.Setting import get_norm_layer
from Utils.Loss import GANLoss, ThetaLoss, ImageLoss, FlowLoss
from Utils.Transform import SpatialTransformer, AffineTransform

sys.path.append('../Networks/')


class CGANModel(BaseModel):
    def Name(self):
        return 'CGANModel'

    @staticmethod
    def ModifyCommandlineOptions(parser, is_train=True):
        # default ModelSetting did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def Initialize(self, opt):
        BaseModel.Initialize(self, opt)

        # Define G & D
        # TODO: Define generator which consists of an affine transformation and N elastic deformations
        self.is_affine = opt.is_affine
        self.is_min = opt.is_min
        self.is_box = opt.is_box

        self.netG = nn.Sequential()
        if self.is_affine:
            self.netG.append(VTNAffine(opt.patch_size, opt.input_nc))

        for _ in range(opt.n_deform):
            if opt.deform_net == 'VTN':
                self.netG.append(VTN(opt.input_nc))
            elif opt.deform_net == 'UNet':
                self.netG.append(UNet(opt.input_nc, [16, 32, 64, 128, 256], 3, net_mode='3d'))
            else:
                raise ValueError('unfound deform network')
        self.netG.to(self.device)
        # TODO: Define discriminator

        # transform
        self.warpper = SpatialTransformer(opt.patch_size)

        # train process include train & val, both need loss
        # TODO:check discriminator
        if self.is_train:
            # self.netD = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
            #                                 norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
            # self.netD.to(self.device)
            #
            self.model_names = ['G']
            self.loss_names = ['G_loss_dict']
            # loss
            self.affine_loss = ThetaLoss(dict(zip(opt.affine_losses, opt.weights_affine_losses)))
            self.deform_loss = FlowLoss(dict(zip(opt.deform_losses, opt.weights_deform_losses)))
            self.image_loss = ImageLoss(dict(zip(opt.deform_losses, opt.weights_image_losses)))
            # self.criterionGAN = GANLoss(use_lsgan=False)
            # initialize optimizers / val need not to update parameters
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # if self.is_train:
        #     self.netD = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=opt.n_layers_D,
        #                                     norm_layer=get_norm_layer(opt.norm), use_sigmoid=True)
        #     self.netD.to(self.device)
        #
        #     self.model_names = ['G', 'D']
        #     self.loss_names = ['G_loss_dict', 'loss_D']
        #     # loss
        #     self.affine_loss = ThetaLoss(dict(zip(opt.affine_losses, opt.weights_affine_losses)))
        #     self.deform_loss = FlowLoss(dict(zip(opt.deform_losses, opt.weights_deform_losses)))
        #     self.image_loss = ImageLoss(dict(zip(opt.deform_losses, opt.weights_image_losses)))
        #     self.criterionGAN = GANLoss(use_lsgan=False)
        #     # initialize optimizers / val need not to update parameters
        #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        #     self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        #     self.optimizers = []
        #     self.optimizers.append(self.optimizer_G)
        #     self.optimizers.append(self.optimizer_D)
        # else:  # during test time, only load Gs
        #     self.model_names = ['G']

    def SetInput(self, input):
        self.moving = input[0].to(self.device)        # input of G
        self.fixed = input[1].to(self.device)         # input of D
        self.field = input[2].to(self.device)         # label of G deform part
        self.volume = input[3].to(self.device)        # input of G, concat with moving image
        self.moving_mask = input[4].to(self.device)   # only use to compute loss with mask
        self.affine_param = input[5].to(self.device)  # label of G affine part

    def Forward(self):
        '''
        :return: deform images[1 + 1 + n], theta, flow
        '''
        self.final_flow = torch.zeros_like(self.field)
        # self.theta = torch.zeros_like(self.affine_param)
        # save all transform image during training

        self.transform_image_list = [self.moving]    # transform_image_list[0] = moving image
        for net_num, net in enumerate(self.netG):
            # if len(self.generator) == 4, generator[0] = affine model
            if len(self.netG) == 4 and net_num == 0:
                self.theta = net(self.transform_image_list[net_num], self.volume)
                # print(self.theta)
                self.transform_image_list.append(AffineTransform(self.transform_image_list[net_num], self.theta))
            else:
                flow = net(self.transform_image_list[net_num], self.volume)
                self.transform_image_list.append(self.warpper(self.transform_image_list[net_num], flow, is_min=self.is_min))
                self.final_flow += flow

    def Loss_D(self):
        """Calculate GAN loss for the discriminator"""
        ##### Fake; stop backprop to the generator by detaching fake_B #####
        fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]), dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        plt.subplot(121)
        plt.hist(pred_fake.cpu().data.numpy().flatten())
        plt.title('fake')
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        ##### Real #####
        real_AB = torch.cat((self.moving, self.volume, self.fixed), dim=1)
        pred_real = self.netD(real_AB)
        plt.subplot(122)
        plt.hist(pred_real.cpu().data.numpy().flatten())
        plt.title('true')
        plt.show()
        self.loss_D_real = self.criterionGAN(pred_real, True)

        ##### combine loss and calculate gradients #####
        # print(pred_real, pred_fake)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def Loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        ##### First, G(A) = B #####
        affine_loss_dict = self.affine_loss(self.theta, self.affine_param)
        deform_loss_dict = self.deform_loss(self.final_flow, self.field)
        image_loss_dict = self.image_loss(self.fixed, self.transform_image_list[1:])

        self.G_loss_dict = dict(ChainMap(affine_loss_dict, deform_loss_dict, image_loss_dict))

        ##### Second, G(A) should fake the discriminator #####
        # self.real_A = 条件, self.fake_B = fixed image  ->应该需要添加moving image
        # fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]), dim=1)
        # pred_fake = self.netD(fake_AB)
        # print(pred_fake)
        # self.G_loss_dict['GAN'] = self.criterionGAN(pred_fake, True)
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        ##### combine loss and calculate gradients #####
        self.loss_G = sum(self.G_loss_dict.values())

    # def OptimizeParameters(self):
    #     self.Forward()  # compute fake images: G(A)
    #     # update G
    #     self.SetRequiresGrad(self.netD, False)  # D requires no gradients when optimizing G
    #     self.optimizer_G.zero_grad()  # set G's gradients to zero
    #     self.Loss_G()  # calculate graidents for G
    #     self.loss_G.backward()
    #     self.optimizer_G.step()
    #     # update D
    #     self.SetRequiresGrad(self.netD, True)  # enable backprop for D
    #     self.optimizer_D.zero_grad()  # set D's gradients to zero
    #     self.Loss_D()  # calculate gradients for D
    #     self.loss_D.backward()
    #     self.optimizer_D.step()  # update D's weights
    #
    # def ModelValidation(self):
    #     self.Forward()
    #     self.Loss_G()
    #     self.Loss_D()
    def OptimizeParameters(self):
        self.Forward()  # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.Loss_G()  # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_G.step()

    def ModelValidation(self):
        self.Forward()
        self.Loss_G()