import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This source code is based on 
1) "Scale-Equivariant Steerable Networks",  
paper url : https://arxiv.org/abs/1910.11093 ICLR 2019
github : https://github.com/ISosnovik/sesn

2) "DISCO: accurate Discrete Scale Convolutions", 
paper url : https://arxiv.org/abs/2106.02733 BMVC 2021
github : https://github.com/ISosnovik/disco

3) "Scale Equivariace Improves Siamse Tracking",
paper url : https://arxiv.org/abs/2007.09115 WACV 2021
github : https://github.com/ISosnovik/SiamSE

We've added and modifed some codes according to its purpose.
'''

class SEConv_Z2_H(nn.Module):

    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        permute (bool): Configuration for the basis function scale. 
            If True, activates scale-combined mode; otherwise, 
            utilizes the scale-isolated basis function.
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, basis, kernel_size,
                stride=1, padding=0, permute = False, bias=False,padding_mode='circular'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.basis = basis
        self.num_scales = self.basis.size(1)
        if not permute:
            self.num_funcs = self.basis.size(0)
        else:
            self.num_funcs = self.basis.size(2)
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias= None
        
        self.init_weight_()
    
    def init_weight_(self):
        nn.init.kaiming_uniform_(self.weight,a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        basis = self.basis.view(self.num_funcs, -1).to(x.device)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_scales, self.kernel_size, self.kernel_size)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        if self.padding > 0:
            x = F.pad(x, 4 * [self.padding], mode=self.padding_mode)
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y
    

class SEConv_H_H(nn.Module):

    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        permute (bool): Configuration for the basis function scale. 
            If True, activates scale-combined mode; otherwise, 
            utilizes the scale-isolated basis function.
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, basis, scale_size, kernel_size,
                stride=1, padding=0, permute = False, bias=False,padding_mode='circular'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.basis = basis
        self.num_scales = self.basis.size(1)
        if not permute:
            self.num_funcs = self.basis.size(0)
        else:
            self.num_funcs = self.basis.size(2)
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.init_weight_()
    
    def init_weight_(self):
        nn.init.kaiming_uniform_(self.weight,a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1).to(x.device)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            if self.padding > 0:
                x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
            output += F.conv2d(x_, kernel[:, :, i], groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output


class SEConv_H_H_1x1(nn.Module):

    def __init__(self, in_channels, out_channel, scale_size=1, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channel
        self.scale_size = scale_size
        self.stride = (1, stride, stride)
        if scale_size > 1:
            # workaround for being compatible with the old-style weight init
            self.weight = nn.Parameter(torch.Tensor(out_channel, in_channels, scale_size, 1, 1))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channel, in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        weight = self.weight
        if len(weight.shape) == 4:
            weight = weight[:, :, None]
        pad = self.scale_size - 1
        return F.conv3d(x, weight, padding=[pad, 0, 0], stride=self.stride)[:, :, pad:]

def SEMaxProjection(x):
    
    return x.max(2)[0]