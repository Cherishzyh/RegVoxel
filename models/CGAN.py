import os

import numpy as np
import pandas as pd
from collections import Counter, ChainMap

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from T4T.Utility.CallBacks import EarlyStopping

from Utils.Transform import SpatialTransformer, AffineTransform
from Utils.Loss import GANLoss, ThetaLoss, ImageLoss, FlowLoss, MaskLoss


class CGANModel():
    def __init__(self, opt, net_deform, net_D=None, net_D2=None):
        super(CGANModel, self).__init__()
        self.opt = opt
        self.net_deform = net_deform
        self.netD = net_D
        self.netD2 = net_D2
        if self.netD2:
            assert len(self.opt.target_of_D) == 2
        self.warpper = SpatialTransformer(opt.patch_size)
        self.save_folder = os.path.join(opt.expr_dir, opt.name)
        self.early_stopping = EarlyStopping(store_path=str(os.path.join(self.save_folder, '{}-{:.6f}.pt')), patience=100, verbose=True)
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')

    ############## Before Train ##############
    def ModelBuild(self, is_train=False):
        # Define G & D
        # Define generator which consists of an affine transformation and N elastic deformations
        self.is_min = self.opt.is_min
        self.is_box = self.opt.is_box

        self.netG = nn.ModuleList()

        for _ in range(self.opt.n_deform):
            self.netG.append(self.net_deform)

        if self.opt.G_weights_path:
            state_dict = torch.load(self.opt.G_weights_path, map_location=self.device)
            for key in list(state_dict.keys()):
                state_dict[key.replace('deforms.', '')] = state_dict.pop(key)
            self.netG.load_state_dict(state_dict)
        self.netG.to(self.device)

        # Define discriminator
        if self.netD:
            if self.opt.D_weights_path:
                self.netD.load_state_dict(torch.load(self.opt.D_weights_path, map_location=self.device))
            elif (not is_train) and (not self.opt.D_weights_path):
                assert FileNotFoundError('Please Give G_weights_path')
            else: pass
            self.netD.to(self.device)

        # Define discriminator 2
        if self.netD2:
            if self.opt.D2_weights_path:
                self.netD2.load_state_dict(torch.load(self.opt.D2_weights_path, map_location=self.device))
            elif (not is_train) and (not self.opt.D2_weights_path):
                assert FileNotFoundError('Please Give D2_weights_path')
            else: pass
            self.netD2.to(self.device)

    def DefineLoss(self):
        self.model_names = ['G']
        self.loss_names = ['G_loss_dict']
        if self.netD:
            self.model_names.append('D')
            self.loss_names.append('D_loss_dict')
        if self.netD2:
            self.model_names.append('D2')
            self.loss_names.append('D2_loss_dict')

        if self.netD:
            self.criterionGAN = GANLoss(use_lsgan=True)

        self.deform_loss = FlowLoss(dict(zip(self.opt.deform_losses, self.opt.weights_deform_losses)))
        self.image_loss = ImageLoss(dict(zip(self.opt.image_losses, self.opt.weights_image_losses)))
        self.mask_loss = MaskLoss(dict(zip(self.opt.mask_losses, self.opt.weights_mask_losses)))

    def DefineOptimizer(self):
        # initialize optimizers / val need not to update parameters
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        self.scheduler_G = self._GetScheduler(self.optimizer_G)
        if self.netD:
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
            self.scheduler_D = self._GetScheduler(self.optimizer_D)
        if self.netD2:
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
            self.scheduler_D2 = self._GetScheduler(self.optimizer_D2)

    def _GetScheduler(self, optimizer):
        if self.opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - 100) / 1400
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_decay_iters, gamma=0.1)
        elif self.opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif self.opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.opt.lr_policy)
        return scheduler

    def ModelPrepare(self, is_train=True):
        self.ModelBuild(is_train)
        if is_train:
            self.DefineLoss()
            self.DefineOptimizer()

    ############## Setting ##############
    def SetInput(self, input):
        self.moving = input[0].to(self.device)        # input of G
        self.fixed = input[1].to(self.device)         # input of D
        self.field = input[2].to(self.device)         # label of G deform part
        self.volume = input[3].to(self.device)        # input of G, concat with moving image
        self.moving_mask = input[4].to(self.device)   # only use to compute loss with mask
        self.fixed_mask = input[5].to(self.device)  # only use to compute loss with mask
        self.moving_nodule = input[6].to(self.device)  # only use to compute loss with mask
        self.fixed_nodule = input[7].to(self.device)  # only use to compute loss with mask
        # self.affine_param = input[8].to(self.device)  # label of G affine part

    def SetTrain(self):
        self.netG.train()
        self.train_loss_G = {}
        if self.netD:
            self.netD.train()
            self.train_loss_D = {}
        if self.netD2:
            self.netD2.train()
            self.train_loss_D2 = {}

    def SetValidation(self):
        self.netG.eval()
        self.validation_loss_G = {}
        if self.netD:
            self.netD.eval()
            self.validation_loss_D = {}
        if self.netD2:
            self.netD2.eval()
            self.validation_loss_D2 = {}

    def _SetRequiresGrad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    ############## Train ##############
    def GeneratorRun(self):
        '''
            :return: deform images[1 + 1 + n],
        '''
        self.transform_image_list = [self.moving]  # transform_image_list[0] = moving image
        self.transform_mask_list = [self.moving_mask]
        self.flow_list = []
        self.transform_nodule_list = [self.moving_nodule]

        for net_num, net in enumerate(self.netG):
            # generator[0] = affine model
            flow = net(self.transform_image_list[net_num], self.volume)
            self.transform_image_list.append(self.warpper(self.transform_image_list[net_num], flow, is_min=self.is_min))
            self.transform_mask_list.append(self.warpper(self.transform_mask_list[net_num], flow, is_min=self.is_min))
            if torch.sum(self.moving_nodule) > 0:
                self.transform_nodule_list.append(self.warpper(self.transform_nodule_list[net_num], flow, is_min=self.is_min))
            else:
                self.transform_nodule_list.append(self.moving_nodule)

            # flow1, flow2, flow3...
            self.flow_list.append(flow)

    def LossG(self):
        """Calculate GAN and L1 loss for the generator"""
        ##### First, G(A) = B #####
        deform_loss_dict = {}
        image_loss_dict = {}
        mask_loss_dict = {}

        # Compute image loss
        for idx, image in enumerate(self.transform_image_list):
            if idx == 0: continue  #transform_image_list[0] == moving image
            image_loss_dict = dict(Counter(self.image_loss(self.fixed, image)) + Counter(image_loss_dict))

        # Compute mask loss
        for idx, mask in enumerate(self.transform_mask_list):
            if idx == 0: continue  # transform_nodule_list[0] == moving nodule
            mask_loss_dict = dict(Counter(self.mask_loss(self.fixed_mask, mask, self.fixed_nodule, self.transform_nodule_list[idx])) + Counter(mask_loss_dict))

        # Compute field
        if self.opt.field_compute_way == 'add':
            deform_loss_dict = dict(Counter(self.deform_loss(sum(self.flow_list), self.field)) + Counter(deform_loss_dict))
        else:
            for idx, flow in enumerate(self.flow_list):
                if self.opt.field_compute_way == 'single':
                    deform_loss_dict = dict(Counter(self.deform_loss(flow, self.field)) + Counter(deform_loss_dict))
                elif self.opt.field_compute_way == 'add_step_by_step':
                    deform_loss_dict = dict(Counter(self.deform_loss(sum(self.flow_list[0: idx+1]), self.field)) + Counter(deform_loss_dict))
                else:
                    raise ValueError('Computation way that cannot be recognized')

        G_loss_dict = dict(Counter(deform_loss_dict) + Counter(image_loss_dict) + Counter(mask_loss_dict))

        ##### Second, G(A) should fake the discriminator #####
        # self.real_A = 条件, self.fake_B = fixed image  ->应该需要添加moving image
        if self.netD:
            if self.opt.target_of_D[0] == 'image':
                fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]), dim=1)
                pred_fake = self.netD(fake_AB)
                G_loss_dict['GAN'] = self.criterionGAN(pred_fake, True)
            elif self.opt.target_of_D[0] == 'field':
                fake_AB = torch.cat((self.moving, self.volume, sum(self.flow_list)), dim=1)
                pred_fake = self.netD(fake_AB)
                G_loss_dict['GAN'] = self.criterionGAN(pred_fake, True)
            else:
                raise ValueError('target_of_D only can be image or field, please check!')
            G_loss_dict['all'] += G_loss_dict['GAN']
        if self.netD2:
            if self.opt.target_of_D[1] == 'image':
                fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]), dim=1)
                pred_fake = self.netD2(fake_AB)
                G_loss_dict['GAN2'] = self.criterionGAN(pred_fake, True)
            elif self.opt.target_of_D[1] == 'field':
                fake_AB = torch.cat((self.moving, self.volume, sum(self.flow_list)), dim=1)
                pred_fake = self.netD2(fake_AB)
                G_loss_dict['GAN2'] = self.criterionGAN(pred_fake, True)
            else:
                raise ValueError('target_of_D only can be image or field, please check!')
            G_loss_dict['all'] += G_loss_dict['GAN2']
        #     TODO:
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        return G_loss_dict

    def LossD(self, model, target_of_D):
        """Calculate GAN loss for the discriminator"""
        #### Fake; stop backprop to the generator by detaching fake_B #####
        if target_of_D == 'image':
            fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]), dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            ##### Real #####
            real_AB = torch.cat((self.moving, self.volume, self.fixed), dim=1)
            pred_real = model(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
        elif target_of_D == 'field':
            fake_AB = torch.cat((self.moving, self.volume, sum(self.flow_list)), dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)

            ##### Real #####
            real_AB = torch.cat((self.moving, self.volume, self.field), dim=1)
            pred_real = model(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
        else:
            raise ValueError('target_of_D only can be image or field, please check!')

        ##### combine loss and calculate gradients #####
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def Train(self, update_d):
        self.GeneratorRun()
        # update D
        if self.netD and update_d:
            self._SetRequiresGrad(self.netD, True)
            self._SetRequiresGrad(self.netD2, False)
            self.optimizer_D.zero_grad()
            loss_D = self.LossD(self.netD, self.opt.target_of_D[0])
            loss_D.backward()
            self.optimizer_D.step()

        if self.netD2 and update_d:
            self._SetRequiresGrad(self.netD, False)
            self._SetRequiresGrad(self.netD2, True)
            self.optimizer_D2.zero_grad()
            loss_D2 = self.LossD(self.netD2, self.opt.target_of_D[1])
            loss_D2.backward()
            self.optimizer_D2.step()

        # update G
        if self.netD:
            self._SetRequiresGrad(self.netD, False)
        if self.netD2:
            self._SetRequiresGrad(self.netD2, False)
        self.optimizer_G.zero_grad()
        loss_G = self.LossG()
        loss_G['all'].backward()
        self.optimizer_G.step()

        self.train_loss_G = dict(Counter(self.train_loss_G) + Counter(loss_G))
        if self.netD and update_d:
            self.train_loss_D = dict(Counter(self.train_loss_D) + Counter({'GAN': loss_D}))
        if self.netD2 and update_d:
            self.train_loss_D2 = dict(Counter(self.train_loss_D2) + Counter({'GAN': loss_D2}))

    def Validation(self, is_test=False):
        self.GeneratorRun()
        if not is_test:
            loss_G = self.LossG()
            self.validation_loss_G = dict(Counter(self.validation_loss_G) + Counter(loss_G))
            if self.netD:
                loss_D = self.LossD(self.netD, self.opt.target_of_D[0])
                self.validation_loss_D = dict(Counter(self.validation_loss_D) + Counter({'GAN': loss_D}))
            if self.netD2:
                loss_D2 = self.LossD(self.netD2, self.opt.target_of_D[1])
                self.validation_loss_D2 = dict(Counter(self.validation_loss_D2) + Counter({'GAN': loss_D2}))
        return self.flow_list, self.transform_image_list, self.transform_mask_list, self.transform_nodule_list

    def TestDiscriminator(self, model, target_of_D):
        self.GeneratorRun()
        """Calculate GAN loss for the discriminator"""
        if target_of_D == 'image':
            fake_AB = torch.cat((self.moving, self.volume, self.transform_image_list[-1]),
                                dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model(fake_AB.detach())

            real_AB = torch.cat((self.moving, self.volume, self.fixed), dim=1)
            pred_real = model(real_AB)
        elif target_of_D == 'field':
            fake_AB = torch.cat((self.moving, self.volume, sum(self.flow_list)),
                                dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = model(fake_AB.detach())

            real_AB = torch.cat((self.moving, self.volume, self.field), dim=1)
            pred_real = model(real_AB)
        else:
            raise ValueError('target_of_D only can be image or field, please check!')

        return pred_fake, pred_real

    ############## After Train ##############
    def SaveAndUpdate(self, epoch, update_d, val_batches, early_stopping=False):
        self.scheduler_G.step(sum(self.validation_loss_G.values()) / val_batches)
        if early_stopping:
            self.early_stopping(sum(self.validation_loss_G.values()) / val_batches, self.netG, (epoch + 1, sum(self.validation_loss_G.values()) / val_batches))
            if self.early_stopping.early_stop:
                print("Early stopping")
                return True
            else:
                return False
        else:
            # 保存loss：所有的train和val。每次都写入csv
            train_loss = {}
            val_loss = {}
            torch.save(self.netG.state_dict(), os.path.join(self.opt.expr_dir, 'netG_{}.pt').format(epoch))
            for key in self.train_loss_G:
                train_loss['train_G_{}'.format(key)] = self.train_loss_G[key].item()
                val_loss['val_G_{}'.format(key)] = self.validation_loss_G[key].item()

            if self.netD and update_d:
                torch.save(self.netD.state_dict(), os.path.join(self.opt.expr_dir, 'netD_{}.pt').format(epoch))
                for key in self.train_loss_D:
                    train_loss['train_D_{}'.format(key)] = self.train_loss_D[key].item()
                    val_loss['val_D_{}'.format(key)] = self.validation_loss_D[key].item()
            if self.netD2 and update_d:
                torch.save(self.netD2.state_dict(), os.path.join(self.opt.expr_dir, 'netD2_{}.pt').format(epoch))
                for key in self.train_loss_D2:
                    train_loss['train_D2_{}'.format(key)] = self.train_loss_D2[key].item()
                    val_loss['val_D2_{}'.format(key)] = self.validation_loss_D2[key].item()
            dataframe_train = pd.DataFrame(train_loss, index=[epoch])
            dataframe_val = pd.DataFrame(val_loss, index=[epoch])
            if epoch == 1:
                dataframe_train.to_csv(os.path.join(self.opt.expr_dir, 'train_loss.csv'))
                dataframe_val.to_csv(os.path.join(self.opt.expr_dir, 'val_loss.csv'))
            else:
                dataframe_train.to_csv(os.path.join(self.opt.expr_dir, 'train_loss.csv'), mode='a', header=False)
                dataframe_val.to_csv(os.path.join(self.opt.expr_dir, 'val_loss.csv'), mode='a', header=False)













