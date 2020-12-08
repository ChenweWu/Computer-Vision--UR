#Chenwei Wu 
#HW3

import torch.nn as nn
import torch

def conjugate(x):
    x[..., 1] = - x[..., 1]
    return x

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_img  = x[..., 1] * z[..., 0] + x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_img), -1)

def complex_conjmul(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_img  = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_img), -1)

def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = None
        # xf: the fourier transformation of target patch x.
        self.xf = None
        self.config = config

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (num_scale, 3, crop_sz, crop_sz)
        :return response: the response of cross correlation. Shape (num_scale, 1, crop_sz, crop_sz)

        You are required to calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window

        z = self.feature(z) * self.config.cos_window
        # TODO: You are required to calculate response using self.wf to do cross correlation on the searching patch z
        # put your code here
        zf = torch.rfft(z, signal_ndim=2)
        croped_zf = torch.sum(complex_conjmul(zf, self.xf), dim=1, keepdim=True)
        response = torch.irfft(complex_mul(croped_zf, self.wf), signal_ndim=2)
        # print("zf", zf.shape)
        # print("xf", self.xf.shape)
        # print("wf", self.wf.shape)


        return response

    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula,
                    Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
        x = self.feature(x) * self.config.cos_window
        # TODO: calculate self.xf and self.wf
        # put your code here
        xf = torch.rfft(x, signal_ndim=2)

        if(self.xf is None):
            self.xf = xf.data
        
        # accumulate x 
        self.xf = (1 - lr) * self.xf.data + lr * xf.data

        # not accumulate x 
        # self.xf = xf.data

        wf = self.config.yf / (torch.sum(torch.sum(self.xf ** 2, dim=4, keepdim=True), dim=1, keepdim=True) + self.config.lambda0)

        # accumulate w 
        if(self.wf is None):
            self.wf = wf.data
        else:
            self.wf = (1 - lr) * self.wf.data + lr * wf.data

        # not accumulate w
        # self.wf = wf.data



    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

