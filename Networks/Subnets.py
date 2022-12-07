import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

import sys
sys.path.append('./')


def conv3x3_leaky(in_c, out_c, stride=1):
    return nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                         nn.LeakyReLU())


def conv3x3_leaky_norm(in_c, out_c, stride=1):
    return nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                         nn.InstanceNorm3d(out_c),
                         nn.LeakyReLU())


def deconv2x2_leaky(in_c, out_c):
    return nn.Sequential(nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2),
                         nn.LeakyReLU())


def deconv2x2(in_c, out_c):
    return nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2)


########################## with out norm #############################
class Down(nn.Module):
    def __init__(self, in_c, out_c, double=False) -> None:
        super().__init__()
        
        if double:
            self.conv = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                                      nn.LeakyReLU())
        else:
            self.conv = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                                      nn.LeakyReLU())
    
    def forward(self, x):
        y = self.conv(x)
        return y


class Up(nn.Module):
    def __init__(self, in_c, inplane_c, dims=3) -> None:
        super().__init__()
        
        self.pred = nn.Conv3d(in_c, dims, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose3d(dims, dims, kernel_size=2, stride=2)
        self.deconv = deconv2x2_leaky(in_c, inplane_c)
    
    def forward(self, x, x_tocat):
        """_summary_

        Args:
            x (_type_): (batch, c*2, h/2, w/2, d/2)
            x_tocat (_type_): (batch, c, h, w, d)
        """
        pred = self.pred(x)  # (batch, dims, h/2, w/2, d/2)
        up = self.upsample(pred)  # (batch, dims, h, w, d)
        deconv = self.deconv(x)  # (batch, c, h, w, d)
        
        # change padding in upsample to 1 when using pad
        # up = pad(up, x_tocat.shape)
        # deconv = pad(deconv, x_tocat.shape)
        
        x = torch.cat([x_tocat, deconv, up], dim=1)  # (batch, c*2+dims, h, w, d)
        return x


# class UpUNet(nn.Module):
#     def __init__(self, in_c) -> None:
#         super().__init__()
#
#         self.upsample = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
#         self.deconv = nn.Sequential(nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
#                                     nn.LeakyReLU())
#
#     def forward(self, x, x_tocat):
#         """_summary_
#
#         Args:
#             x (_type_): (batch, c*2, h/2, w/2, d/2)
#             x_tocat (_type_): (batch, c, h, w, d)
#         """
#         up = self.upsample(x)  # (batch, dims, h, w, d)
#         deconv = self.deconv(torch.cat([up, x_tocat], dim=1))  # (batch, c, h, w, d)
#         return deconv
class UpUNet(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv3d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU())

    def forward(self, x, x_tocat):
        """_summary_

        Args:
            x (_type_): (batch, c*2, h/2, w/2, d/2)
            x_tocat (_type_): (batch, c, h, w, d)
        """
        up = self.upsample(x)  # (batch, dims, h, w, d)
        deconv = self.conv(torch.cat([up, x_tocat], dim=1))  # (batch, c, h, w, d)
        return deconv


class UpUNetComplex(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.deconv = nn.Sequential(nn.Conv3d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(in_c//2, in_c // 2, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU()
                                    )

    def forward(self, x, x_tocat):
        """_summary_

        Args:
            x (_type_): (batch, c*2, h/2, w/2, d/2)
            x_tocat (_type_): (batch, c, h, w, d)
        """
        up = self.upsample(x)  # (batch, dims, h, w, d)
        deconv = self.deconv(torch.cat([up, x_tocat], dim=1))  # (batch, c, h, w, d)
        return deconv


########################## with norm #############################

class DownNorm(nn.Module):
    def __init__(self, in_c, out_c, double=False) -> None:
        super().__init__()

        if double:
            self.conv = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                                      nn.InstanceNorm3d(out_c),
                                      nn.LeakyReLU())
        else:
            self.conv = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                                      nn.InstanceNorm3d(out_c),
                                      nn.LeakyReLU(),
                                      nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                                      nn.InstanceNorm3d(out_c),
                                      nn.LeakyReLU())

    def forward(self, x):
        y = self.conv(x)
        return y


class UpUNetNorm(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.deconv = nn.Sequential(nn.Conv3d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                    nn.InstanceNorm3d(in_c // 2),
                                    nn.LeakyReLU())

    def forward(self, x, x_tocat):
        """_summary_

        Args:
            x (_type_): (batch, c*2, h/2, w/2, d/2)
            x_tocat (_type_): (batch, c, h, w, d)
        """
        up = self.upsample(x)  # (batch, dims, h, w, d)
        deconv = self.deconv(torch.cat([up, x_tocat], dim=1))  # (batch, c, h, w, d)
        return deconv


class UpUNetNormComplex(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.deconv = nn.Sequential(nn.Conv3d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                    nn.InstanceNorm3d(in_c // 2),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(in_c // 2, in_c // 2, kernel_size=3, stride=1, padding=1),
                                    nn.InstanceNorm3d(in_c // 2),
                                    nn.LeakyReLU()
                                    )

    def forward(self, x, x_tocat):
        """_summary_

        Args:
            x (_type_): (batch, c*2, h/2, w/2, d/2)
            x_tocat (_type_): (batch, c, h, w, d)
        """
        up = self.upsample(x)  # (batch, dims, h, w, d)
        deconv = self.deconv(torch.cat([up, x_tocat], dim=1))  # (batch, c, h, w, d)
        return deconv


######## net ########

class VTNAffine(nn.Module):
    def __init__(self, shape, in_channels, flow_multiplier=1) -> None:
        super().__init__()

        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier

        self.down1 = Down(in_channels, c)
        self.down2 = Down(c, c * 2)
        self.down3 = Down(c * 2, c * 4, double=True)
        self.down4 = Down(c * 4, c * 8, double=True)
        self.down5 = Down(c * 8, c * 16, double=True)
        self.down6 = Down(c * 16, c * 32, double=True)

        self.fc_theta = nn.Linear(c * 32 * shape[0] // 64 * shape[1] // 64 * shape[2] // 64, 12)

        self.init_weight()

    def forward(self, img_m, img_f):
        x = torch.cat([img_m, img_f], dim=1)  # (batch, in_channels, h, w, d)

        x = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x = self.down2(x)  # (batch, c*2, h/4, w/4, d/4)
        x = self.down3(x)  # (batch, c*4, h/8, w/8, d/8)
        x = self.down4(x)  # (batch, c*8, h/16, w/16, d/16)
        x = self.down5(x)  # (batch, c*16, h/32, w/32, d/32)
        x = self.down6(x)  # (batch, c*32, h/64, w/64, d/64)

        x = x.view(x.size(0), -1)  # (batch, c*32 * h/64 * w/64 * d/64)
        theta = self.fc_theta(x).reshape([-1, 3, 4]) * self.flow_multiplier  # (batch, 3, 3)
        eye = torch.tensor([[[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0]]], dtype=theta.dtype, device=theta.device)
        theta += eye
        return theta

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.fc_theta.weight, mean=0, std=1e-10)
        nn.init.constant_(self.fc_theta.bias, 0)


class VTN(nn.Module):
    def __init__(self, in_channels, flow_multiplier=1) -> None:
        super().__init__()
        
        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier
        
        self.down1 = Down(in_channels, c)
        self.down2 = Down(c, c*2)
        self.down3 = Down(c*2, c*4, double=True)
        self.down4 = Down(c*4, c*8, double=True)
        self.down5 = Down(c*8, c*16, double=True)
        self.down6 = Down(c*16, c*32, double=True)
        
        self.up6 = Up(c*32, c*16)
        self.up5 = Up(c*32+dims, c*8)
        self.up4 = Up(c*16+dims, c*4)
        self.up3 = Up(c*8+dims, c*2)
        self.up2 = Up(c*4+dims, c)
    
        self.up1 = deconv2x2(c*2+dims, dims)

        self.init_weight()

    def forward(self, img_m, img_f):
        """_summary_

        Args:
            img_m (tensor): (batch, in_channels, h, w, d)
            img_f (tensor): (batch, 1, h, w, d)

        Returns:
            _type_: (batch, dims, h, w, d)
        """
        x = torch.cat([img_m, img_f], dim=1)  # (batch, 2, h, w, d)
        
        x_d1 = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x_d2 = self.down2(x_d1)  # (batch, c*2, h/4, w/4, d/4)
        x_d3 = self.down3(x_d2)  # (batch, c*4, h/8, w/8, d/8)
        x_d4 = self.down4(x_d3)  # (batch, c*8, h/16, w/16, d/16)
        x_d5 = self.down5(x_d4)  # (batch, c*16, h/32, w/32, d/32)
        x_d6 = self.down6(x_d5)  # (batch, c*32, h/64, w/64, d/64)
        
        x = self.up6(x_d6, x_d5)  # (batch, c*32+dims, h/32, w/32, d/32)
        x = self.up5(x, x_d4)  # (batch, c*16+dims, h/16, w/16, d/16)
        x = self.up4(x, x_d3)  # (batch, c*8+dims, h/8, w/8, d/8)
        x = self.up3(x, x_d2)  # (batch, c*4+dims, h/4, w/4, d/4)
        x = self.up2(x, x_d1)  # (batch, c*2+dims, h/2, w/2, d/2)
        
        x = self.up1(x)  # (batch, dims, h, w, d)
        # x = pad(x, img_m.shape)
        
        x = x * 20 * self.flow_multiplier
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.up1.weight, mean=0, std=1e-10)
        nn.init.constant_(self.up1.bias, 0)


class UNet(nn.Module):
    def __init__(self, in_channels, flow_multiplier=1) -> None:
        super().__init__()

        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier

        self.down1 = Down(in_channels, c)
        self.down2 = Down(c, c * 2)
        self.down3 = Down(c * 2, c * 4, double=True)
        self.down4 = Down(c * 4, c * 8, double=True)
        self.down5 = Down(c * 8, c * 16, double=True)
        self.down6 = Down(c * 16, c * 32, double=True)

        self.up6 = UpUNet(c * 32)
        self.up5 = UpUNet(c * 16)
        self.up4 = UpUNet(c * 8)
        self.up3 = UpUNet(c * 4)
        self.up2 = UpUNet(c * 2)

        self.up1 = deconv2x2(c, dims)

        self.init_weight()

    def forward(self, img_m, img_f):
        """_summary_

        Args:
            img_m (tensor): (batch, in_channels, h, w, d)
            img_f (tensor): (batch, 1, h, w, d)

        Returns:
            _type_: (batch, dims, h, w, d)
        """
        x = torch.cat([img_m, img_f], dim=1)  # (batch, 2, h, w, d)

        x_d1 = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x_d2 = self.down2(x_d1)  # (batch, c*2, h/4, w/4, d/4)
        x_d3 = self.down3(x_d2)  # (batch, c*4, h/8, w/8, d/8)
        x_d4 = self.down4(x_d3)  # (batch, c*8, h/16, w/16, d/16)
        x_d5 = self.down5(x_d4)  # (batch, c*16, h/32, w/32, d/32)
        x_d6 = self.down6(x_d5)  # (batch, c*32, h/64, w/64, d/64)
        x = self.up6(x_d6, x_d5)  # (batch, c*32/2 + c*16, h/32, w/32, d/32) out (batch, c * 16, h/32, w/32, d/32)
        x = self.up5(x, x_d4)  # (batch, c*16//2+c*8, h/16, w/16, d/16)  out (batch, c*8, h/16, w/16, d/16)
        x = self.up4(x, x_d3)  # (batch, c*8+dims, h/8, w/8, d/8)
        x = self.up3(x, x_d2)  # (batch, c*4+dims, h/4, w/4, d/4)
        x = self.up2(x, x_d1)  # (batch, c*2+dims, h/2, w/2, d/2)
        x = self.up1(x)  # (batch, dims, h, w, d)
        # x = pad(x, img_m.shape)

        x = x * 20 * self.flow_multiplier
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.up1.weight, mean=0, std=1e-10)
        nn.init.constant_(self.up1.bias, 0)


class UNetComplex(nn.Module):
    def __init__(self, in_channels, flow_multiplier=1) -> None:
        super().__init__()

        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier

        self.down1 = Down(in_channels, c)
        self.down2 = Down(c, c * 2)
        self.down3 = Down(c * 2, c * 4, double=True)
        self.down4 = Down(c * 4, c * 8, double=True)
        self.down5 = Down(c * 8, c * 16, double=True)
        self.down6 = Down(c * 16, c * 32, double=True)

        self.up6 = UpUNetComplex(c * 32)
        self.up5 = UpUNetComplex(c * 16)
        self.up4 = UpUNetComplex(c * 8)
        self.up3 = UpUNetComplex(c * 4)
        self.up2 = UpUNetComplex(c * 2)

        self.up1 = deconv2x2(c, dims)

        self.init_weight()

    def forward(self, img_m, img_f):
        """_summary_

        Args:
            img_m (tensor): (batch, in_channels, h, w, d)
            img_f (tensor): (batch, 1, h, w, d)

        Returns:
            _type_: (batch, dims, h, w, d)
        """
        x = torch.cat([img_m, img_f], dim=1)  # (batch, 2, h, w, d)

        x_d1 = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x_d2 = self.down2(x_d1)  # (batch, c*2, h/4, w/4, d/4)
        x_d3 = self.down3(x_d2)  # (batch, c*4, h/8, w/8, d/8)
        x_d4 = self.down4(x_d3)  # (batch, c*8, h/16, w/16, d/16)
        x_d5 = self.down5(x_d4)  # (batch, c*16, h/32, w/32, d/32)
        x_d6 = self.down6(x_d5)  # (batch, c*32, h/64, w/64, d/64)
        x = self.up6(x_d6, x_d5)  # (batch, c*32/2 + c*16, h/32, w/32, d/32) out (batch, c * 16, h/32, w/32, d/32)
        x = self.up5(x, x_d4)  # (batch, c*16//2+c*8, h/16, w/16, d/16)  out (batch, c*8, h/16, w/16, d/16)
        x = self.up4(x, x_d3)  # (batch, c*8+dims, h/8, w/8, d/8)
        x = self.up3(x, x_d2)  # (batch, c*4+dims, h/4, w/4, d/4)
        x = self.up2(x, x_d1)  # (batch, c*2+dims, h/2, w/2, d/2)
        x = self.up1(x)  # (batch, dims, h, w, d)
        # x = pad(x, img_m.shape)

        x = x * 20 * self.flow_multiplier
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.up1.weight, mean=0, std=1e-10)
        nn.init.constant_(self.up1.bias, 0)


class UNetNorm(nn.Module):
    def __init__(self, in_channels, flow_multiplier=1) -> None:
        super().__init__()

        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier

        self.down1 = DownNorm(in_channels, c)
        self.down2 = DownNorm(c, c * 2)
        self.down3 = DownNorm(c * 2, c * 4, double=True)
        self.down4 = DownNorm(c * 4, c * 8, double=True)
        self.down5 = DownNorm(c * 8, c * 16, double=True)
        self.down6 = DownNorm(c * 16, c * 32, double=True)

        self.up6 = UpUNetNorm(c * 32)
        self.up5 = UpUNetNorm(c * 16)
        self.up4 = UpUNetNorm(c * 8)
        self.up3 = UpUNetNorm(c * 4)
        self.up2 = UpUNetNorm(c * 2)

        self.up1 = deconv2x2(c, dims)

        self.init_weight()

    def forward(self, img_m, img_f):
        """_summary_

        Args:
            img_m (tensor): (batch, in_channels, h, w, d)
            img_f (tensor): (batch, 1, h, w, d)

        Returns:
            _type_: (batch, dims, h, w, d)
        """
        x = torch.cat([img_m, img_f], dim=1)  # (batch, 2, h, w, d)

        x_d1 = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x_d2 = self.down2(x_d1)  # (batch, c*2, h/4, w/4, d/4)
        x_d3 = self.down3(x_d2)  # (batch, c*4, h/8, w/8, d/8)
        x_d4 = self.down4(x_d3)  # (batch, c*8, h/16, w/16, d/16)
        x_d5 = self.down5(x_d4)  # (batch, c*16, h/32, w/32, d/32)
        x_d6 = self.down6(x_d5)  # (batch, c*32, h/64, w/64, d/64)
        x = self.up6(x_d6, x_d5)  # (batch, c*32/2 + c*16, h/32, w/32, d/32) out (batch, c * 16, h/32, w/32, d/32)
        x = self.up5(x, x_d4)  # (batch, c*16//2+c*8, h/16, w/16, d/16)  out (batch, c*8, h/16, w/16, d/16)
        x = self.up4(x, x_d3)  # (batch, c*8+dims, h/8, w/8, d/8)
        x = self.up3(x, x_d2)  # (batch, c*4+dims, h/4, w/4, d/4)
        x = self.up2(x, x_d1)  # (batch, c*2+dims, h/2, w/2, d/2)
        x = self.up1(x)  # (batch, dims, h, w, d)
        # x = pad(x, img_m.shape)

        x = x * 20 * self.flow_multiplier
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.up1.weight, mean=0, std=1e-10)
        nn.init.constant_(self.up1.bias, 0)


class UNetComplexNorm(nn.Module):
    def __init__(self, in_channels, flow_multiplier=1) -> None:
        super().__init__()

        c = 16
        dims = 3
        self.flow_multiplier = flow_multiplier

        self.down1 = DownNorm(in_channels, c)
        self.down2 = DownNorm(c, c * 2)
        self.down3 = DownNorm(c * 2, c * 4, double=True)
        self.down4 = DownNorm(c * 4, c * 8, double=True)
        self.down5 = DownNorm(c * 8, c * 16, double=True)
        self.down6 = DownNorm(c * 16, c * 32, double=True)

        self.up6 = UpUNetNormComplex(c * 32)
        self.up5 = UpUNetNormComplex(c * 16)
        self.up4 = UpUNetNormComplex(c * 8)
        self.up3 = UpUNetNormComplex(c * 4)
        self.up2 = UpUNetNormComplex(c * 2)

        self.up1 = deconv2x2(c, dims)

        self.init_weight()

    def forward(self, img_m, img_f):
        """_summary_

        Args:
            img_m (tensor): (batch, in_channels, h, w, d)
            img_f (tensor): (batch, 1, h, w, d)

        Returns:
            _type_: (batch, dims, h, w, d)
        """
        x = torch.cat([img_m, img_f], dim=1)  # (batch, 2, h, w, d)

        x_d1 = self.down1(x)  # (batch, c, h/2, w/2, d/2)
        x_d2 = self.down2(x_d1)  # (batch, c*2, h/4, w/4, d/4)
        x_d3 = self.down3(x_d2)  # (batch, c*4, h/8, w/8, d/8)
        x_d4 = self.down4(x_d3)  # (batch, c*8, h/16, w/16, d/16)
        x_d5 = self.down5(x_d4)  # (batch, c*16, h/32, w/32, d/32)
        x_d6 = self.down6(x_d5)  # (batch, c*32, h/64, w/64, d/64)
        x = self.up6(x_d6, x_d5)  # (batch, c*32/2 + c*16, h/32, w/32, d/32) out (batch, c * 16, h/32, w/32, d/32)
        x = self.up5(x, x_d4)  # (batch, c*16//2+c*8, h/16, w/16, d/16)  out (batch, c*8, h/16, w/16, d/16)
        x = self.up4(x, x_d3)  # (batch, c*8+dims, h/8, w/8, d/8)
        x = self.up3(x, x_d2)  # (batch, c*4+dims, h/4, w/4, d/4)
        x = self.up2(x, x_d1)  # (batch, c*2+dims, h/2, w/2, d/2)
        x = self.up1(x)  # (batch, dims, h, w, d)
        # x = pad(x, img_m.shape)

        x = x * 20 * self.flow_multiplier
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(self.up1.weight, mean=0, std=1e-10)
        nn.init.constant_(self.up1.bias, 0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminatorSimple(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(NLayerDiscriminatorSimple, self).__init__()

        kw = 4
        padw = 1

        self.conv1 = nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw)
        self.norm2 = nn.InstanceNorm3d(ndf * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw)
        self.norm3 = nn.InstanceNorm3d(ndf * 4)
        self.leakyrelu3 = nn.LeakyReLU(0.2, True)
        self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw)
        self.norm4 = nn.InstanceNorm3d(ndf * 8)
        self.leakyrelu4 = nn.LeakyReLU(0.2, True)
        self.conv5 = nn.Conv3d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)    # torch.Size([1, 64, 128, 128, 128])
        x = self.leakyrelu1(x)
        x = self.conv2(x)        # torch.Size([1, 128, 64, 64, 64])
        x = self.norm2(x)
        x = self.leakyrelu2(x)
        x = self.conv3(x)        # torch.Size([1, 256, 32, 32, 32])
        x = self.norm3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)        # torch.Size([1, 512, 31, 31, 31])
        x = self.norm4(x)
        x = self.leakyrelu4(x)
        x = self.conv5(x)        # torch.Size([1, 1, 30, 30, 30])
        x = self.sigmoid(x)
        return x


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


if __name__ == '__main__':
    img_m = torch.randn([1, 1, 256, 256, 256])
    img_f = torch.randn([1, 1, 256, 256, 256])
    
    # vtn = VTN()
    # flow = vtn(img_m, img_f)
    # print(flow.shape)
    
    discriminator = UNetComplexNorm(2)
    print(discriminator)
    flow = discriminator(img_m, img_f)
    print(flow.shape)
