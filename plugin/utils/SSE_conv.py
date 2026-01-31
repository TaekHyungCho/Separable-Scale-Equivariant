import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This source code is based on SE_Convolution and modified to achieve separable scale-equivariant convolution.
Even indices and odd indices are for extracting low and high-frequency feature maps respectively.
'''
class SSEConv_Z2_H(nn.Module):

    def __init__(self, in_channels, out_channels, basis, odd_indices, even_indices, kernel_size,
                stride=1, padding=0, permute = False, bias=False,padding_mode='circular'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        if not permute:
            self.basis_LF = basis[even_indices]
            self.basis_HF = basis[odd_indices]
            self.num_funcs_LF = self.basis_LF.size(0)
            self.num_funcs_HF = self.basis_HF.size(0)
        else:
            self.basis_LF = basis[:,:,even_indices]
            self.basis_HF = basis[:,:,odd_indices]
            self.num_funcs_LF = self.basis_LF.size(2)
            self.num_funcs_HF = self.basis_HF.size(2)
        self.num_scales = self.basis_LF.size(1)
        self.bias = bias
        self.weight_LF = nn.Parameter(torch.Tensor(out_channels,in_channels,self.num_funcs_LF))
        self.weight_HF = nn.Parameter(torch.Tensor(out_channels,in_channels,self.num_funcs_HF))
        if self.bias:
            self.bias_LF = nn.Parameter(torch.Tensor(out_channels))
            self.bias_HF = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias_LF= None
            self.bias_HF= None
        
        self.init_weight_()
    
    def init_weight_(self):
        nn.init.kaiming_uniform_(self.weight_LF,a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_HF,a=5**0.5)
        if self.bias:
            nn.init.zeros_(self.bias_LF)
            nn.init.zeros_(self.bias_HF)
    
    def forward(self, x):
        
        out_lf = self.conv(x,self.basis_LF,self.weight_LF,self.num_funcs_LF,self.bias_LF)
        out_hf = self.conv(x,self.basis_HF,self.weight_HF,self.num_funcs_HF,self.bias_HF)
        out = torch.cat([out_lf,out_hf],dim=0)
        return out
    
    def conv(self,x,basis,weight,num_func,bias = None):
        kernel = weight @ basis.view(num_func,-1).to(x.device)
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
        if bias is not None:
            y = y + bias.view(1, -1, 1, 1, 1)
        
        return y


class SSEConv_H_H(nn.Module):

    def __init__(self, in_channels, out_channels, basis, odd_indices, even_indices, scale_size, kernel_size,
                stride=1, padding=0, permute = False, bias=False,padding_mode='circular',basic=False,pool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        if not permute:
            self.basis_LF = basis[even_indices]
            self.basis_HF = basis[odd_indices]
            self.num_funcs_LF = self.basis_LF.size(0)
            self.num_funcs_HF = self.basis_HF.size(0)
        else:
            self.basis_LF = basis[:,:,even_indices]
            self.basis_HF = basis[:,:,odd_indices]
            self.num_funcs_LF = self.basis_LF.size(2)
            self.num_funcs_HF = self.basis_HF.size(2)
        self.num_scales = self.basis_LF.size(1)
        self.bias = bias
        self.basic = basic
        self.pool = pool
        self.weight_LF = nn.Parameter(torch.Tensor(out_channels,in_channels,self.num_funcs_LF))
        self.weight_HF = nn.Parameter(torch.Tensor(out_channels,in_channels,self.num_funcs_HF))
        if self.scale_size > 1:
            self.weight_1x1 = nn.Parameter(torch.Tensor(out_channels, out_channels+in_channels, scale_size, 1, 1))
        else:
            self.weight_1x1 = nn.Parameter(torch.Tensor(out_channels, out_channels+in_channels, 1, 1))
        if bias:
            self.bias_LF = nn.Parameter(torch.Tensor(out_channels))
            self.bias_HF = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias_LF= None
            self.bias_HF= None

        self.init_weight_()
    
    def init_weight_(self):
        nn.init.kaiming_uniform_(self.weight_LF,a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_HF,a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_1x1,a=5**0.5)
        if self.bias:
            nn.init.zeros_(self.bias_LF)
            nn.init.zeros_(self.bias_HF)

    def forward(self,x):

        if not self.basic:
            out_lf = self.conv(x,self.basis_LF,self.weight_LF,self.num_funcs_LF,self.stride,self.bias_LF)
            out_hf = self.conv(x,self.basis_HF,self.weight_HF,self.num_funcs_HF,self.stride,self.bias_HF)
            out = torch.cat([out_lf,out_hf],dim=0)
        else:
            #This code is for resnet-basic bottle structures
            B = x.size(0) // 2
            x_lf = x[:B]
            x_hf = x[B:]
            out_lf = self.conv(x_lf,self.basis_LF,self.weight_LF,self.num_funcs_LF,self.stride,self.bias_LF)
            out_hf = self.conv(x_lf,self.basis_HF,self.weight_HF,self.num_funcs_HF,1,self.bias_HF)
            out_hf = torch.cat([out_hf,x_hf],dim=1)
            out_hf = self.conv_1x1(out_hf,self.weight_1x1)
            if self.pool and self.stride !=1:
                out_hf = F.avg_pool3d(out_hf,[1,3,3],stride=[1, self.stride, self.stride],padding=[0,1,1])
            out = torch.cat([out_lf,out_hf],dim=0)
        return out

    def conv(self,x,basis,weight,num_func,stride,bias = None):
        # get kernel
        kernel = weight @ basis.view(num_func, -1).to(x.device)
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
            output += F.conv2d(x_, kernel[:, :, i], groups=S, stride=stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1, 1)
        return output
    
    def conv_1x1(self,x,weight,stride=1):

        if len(weight.shape) == 4:
            weight = weight[:, :, None]
        pad = self.scale_size - 1
        return F.conv3d(x, weight, padding=[pad, 0, 0], stride=stride)[:, :, pad:]




