import math
import os
import time
from iunets.layers import InvertibleDownsampling2D,InvertibleUpsampling2D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.nn.utils import weight_norm
from PIL import Image
import numpy as np
logabs = lambda x: torch.log(torch.abs(x))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_uniform_(m.weight.data)
    else:
        pass
    
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("eye", eye)

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            lower = self.lower * self.l_mask + self.eye
            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, rev)
        weight = weight.to(input.device)
        if not rev:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DenseBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(channel_in, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv11x11 = nn.Conv2d(channel_in, 64, kernel_size=13, stride=1, padding=0, dilation=3, bias=False)
        self.conv_m1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_m2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_m3 = nn.Conv2d(32, channel_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.apply(weights_init)
        
    def forward(self, x):
        upsample = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',align_corners=True)
        x1 = self.conv1x1(x)
        x2 = self.conv11x11(x)
        x = torch.cat([x1, upsample(x2)], dim=1)
        x = self.conv_m3(self.relu(self.conv_m2(self.relu(self.conv_m1(self.relu(x))))))
        return x

def subnet(init='xavier'):
    def constructor(channel_in, channel_out):
        if init == 'xavier':
            return DenseBlock(channel_in, channel_out, init)
        else:
            return DenseBlock(channel_in, channel_out)
    return constructor

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor=subnet('DBNet'), channel_num=12, clamp=0.8):
        super(InvBlock, self).__init__()

        self.split_len1 = int(channel_num / 2)
        self.split_len2 = channel_num - self.split_len1

        self.clamp = clamp
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        self.invconv2 = InvertibleConv1x1(in_channels)
        self.flow_permutation2 = lambda z, logdet, rev: self.invconv2(z, logdet, rev)

    def forward(self, x, rev=False):
        if not rev:
            # split to 1 channel and 2 channel.
            x, logdet = self.flow_permutation(x, logdet=0, rev=False)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
            out = torch.cat((y1, y2), 1)
            out, _ = self.flow_permutation2(out, logdet=0, rev=False)
        else:
            # split.
            x, _ = self.flow_permutation2(x, logdet=0, rev=True)
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)
            x = torch.cat((y1, y2), 1)
            out, _ = self.flow_permutation(x, logdet=0, rev=True)
        return out


class CDNetpp(nn.Module):
    def __init__(self):
        super(CDNetpp, self).__init__()
        operations = []
        b = InvBlock(channel_num=12)
        operations.append(b)
        self.operations = nn.ModuleList(operations)
        self.initialize()
        self.downsampling = InvertibleDownsampling2D(
            in_channels=3,
            method='exp',
            stride=(2,2)
        ).cuda()
        self.upsampling = InvertibleUpsampling2D(
            in_channels=12,
            method='exp',
            stride=(2,2)
        ).cuda()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
    def coordinate_transform(self, x_hat, rev=False,interplote=False):
        if not rev:
            #forward
            x_hat = self.downsampling(x_hat)
            for op in self.operations:
                x_hat = op.forward(x_hat, rev)
            x_hat =self.upsampling(x_hat)
        else:
            #inverse
            x_hat =self.upsampling.inverse(x_hat)
            for j, op in enumerate(reversed(self.operations)):
                x_hat = op.forward(x_hat, rev)
            x_hat = self.downsampling.inverse(x_hat)
        return x_hat

    def forwardsample(self, x, rev=False, no_grad = False ):
        if no_grad == True:
            with torch.no_grad():
                x_hat = self.coordinate_transform(x, rev=rev)
        else:
            x_hat = self.coordinate_transform(x, rev=rev)
        return x_hat

    def forward(self, x, y, rev = False, Lips=False, cdmap=False, epsilon=0.1):
        x_hat = self.coordinate_transform(x, rev=False, interplote=False)
        y_hat = self.coordinate_transform(y, rev=False, interplote=False)
        delta_e = (x_hat - y_hat + 1e-5).pow(2).sum(1).sqrt().mean(dim=[1,2])
        delta_e = delta_e*20
        if Lips==True:
            d = torch.randn(x.shape)
            d = (d.view(d.shape[0], -1) / torch.norm(d.view(d.shape[0], -1), dim=0)).view(d.shape)
            x_e = x + (epsilon * d).cuda()
            z_noise = self.forwardsample(x_e, rev=False)

            d_inv = torch.randn(x.shape)
            d_inv = (d_inv.view(d_inv.shape[0], -1) / torch.norm(d_inv.view(d_inv.shape[0], -1), dim=0)).view(d_inv.shape)
            z_noise_inv = x_hat + (epsilon * d_inv ).cuda() * x_hat.detach().max()
            x_inv_noise = self.forwardsample(z_noise_inv, no_grad=False, rev=True)
            return delta_e, x_hat, z_noise, x_inv_noise
            
        if not rev:
            return delta_e

def scaleoutput(input,param):
    a,b,c,d = param[0],param[1],param[2],param[3]
    return (a-b)/(1+torch.exp((c-input)/torch.abs(d)))+b


if __name__ == '__main__':
    pt = "weight.pt"
    net = CDNetpp().cuda()
    net = nn.DataParallel(net, device_ids=[0])
    checkpoint = torch.load(pt)
    net.load_state_dict(checkpoint['netF_dict'])
    net.eval()
    x = Image.open("NatureImage1.png")
    y = Image.open("NatureImage2.png")
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize(1024), torchvision.transforms.ToTensor()])
    x = trans(x).cuda()
    y = trans(y).cuda()
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    CD= net(x, y)
    print("CDs between input images is {}".format(CD.item()))