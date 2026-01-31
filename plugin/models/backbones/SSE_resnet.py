import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from plugin.utils import SSEConv_Z2_H, SSEConv_H_H, SEConv_H_H_1x1, SEMaxProjection
from plugin.utils.basis import get_basis, get_basis_with_filename

class SSE_ResBlock(BaseModule):

    def __init__(self,in_channels,out_channels,basis,
                odd_indices,even_indices,scale_size,
                stride,expansion,permute=False,downsample=None,init_cfg=None):

        super(SSE_ResBlock,self).__init__(init_cfg)
        assert in_channels % 2 == 0

        self.downsample = downsample

        if self.downsample is not None:

            self.downsample_lf = nn.Sequential(
                SEConv_H_H_1x1(
                    in_channels= in_channels,
                    out_channel= out_channels * expansion,
                    stride=stride
                ),
                nn.BatchNorm3d(out_channels * expansion)
            )

            self.downsample_hf = nn.Sequential(
                SEConv_H_H_1x1(
                    in_channels= in_channels,
                    out_channel= out_channels * expansion,
                    stride=1
                ),
                nn.AvgPool3d(kernel_size=[1,5,5],stride=[1,stride,stride],padding=[0,2,2]),
                nn.BatchNorm3d(out_channels * expansion),
                nn.ReLU(True)
            )

        self.conv1 = nn.Sequential(
                        SEConv_H_H_1x1(in_channels=in_channels,
                            out_channel= out_channels,
                            stride=1,
                            scale_size=scale_size),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(True)
                        )

        self.conv2 = nn.Sequential(
                        SSEConv_H_H(in_channels=out_channels,
                            out_channels= out_channels,
                            kernel_size=5,padding=2,stride=stride,
                            scale_size=scale_size,
                            basis = basis,
                            odd_indices= odd_indices,
                            even_indices = even_indices,
                            permute = permute,
                            basic = False),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(True)
                    )
        self.conv3 = nn.Sequential(
            SEConv_H_H_1x1(in_channels=out_channels,
                out_channel= out_channels * expansion,
                stride=1,
                scale_size=scale_size),
            nn.BatchNorm3d(out_channels * expansion),
            nn.ReLU(True) # Add additional activation function -> Contributes to the reduction of scale-equivariance errors
        )

        self.conv4 = nn.Sequential(
            SEConv_H_H_1x1(in_channels=out_channels * expansion * 2,
                out_channel= out_channels * expansion,
                stride=1,
                scale_size=scale_size),
            nn.BatchNorm3d(out_channels * expansion),
            nn.ReLU(True) # Add additional activation function -> Contributes to the reduction of scale-equivariance errors
        )
        
    
    def forward(self,x):
        B = x.size(0) // 2
        x_lf = x[:B]
        x_hf = x[B:]
        out = self.conv1(x_lf)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            x_lf = self.downsample_lf(x_lf)
            x_hf = self.downsample_hf(x_hf)
            out_lf = out[:B] + x_lf
            out_hf = out[B:]+ x_hf
        else:
            out_hf = torch.cat([out[B:],x_hf],dim=1)
            out_hf = self.conv4(out_hf)
            out_lf = out[:B] + x_lf
            out_hf = out_hf + x_hf
        out = torch.cat([out_lf,out_hf],dim=0)
        out = F.relu(out)
        return out


@MODELS.register_module()
class SSE_ResNet(BaseModule):

    arch_settings = {
        50: (SSE_ResBlock,(3, 4, 6, 3),4),
        101: (SSE_ResBlock,(3, 4, 23, 3),4)
    }

    def __init__(self,
                depth,
                save_dir=None,
                filename=None,
                odd_indices=[1,3,4,5,7],
                even_indices=[0,2,6,8],
                scale_size = 1,
                base_channels = 64,
                strides=(1, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                norm_eval=True,
                permute = False,
                init_cfg=None):
        
        super(SSE_ResNet,self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
                
        self.depth = depth
        self.strides = strides
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.odd_indices = odd_indices
        self.even_indices = even_indices
        self.blocks,self.layers,self.expansion= self.arch_settings[depth]
        self.scale_size = scale_size
        self.permute = permute
        if (save_dir is None) and (filename is None):
            self.basis = get_basis(size=5, scales=[1.0,1.4,2.0],effective_size=3, permute=self.permute)
            self.basis_Z2_H = get_basis(size=7, scales=[1.0,1.4,2.0],effective_size=3, permute=self.permute)
        else:
            self.basis = get_basis_with_filename(dir=save_dir,filename=filename[0],permute=self.permute)
            self.basis_Z2_H = get_basis_with_filename(dir=save_dir,filename=filename[1],permute=self.permute)
        self.base_channels = base_channels
        self.in_c = base_channels
        self.conv1 = nn.Sequential(
            SSEConv_Z2_H(
                in_channels= 3,
                out_channels = 64,
                basis = self.basis_Z2_H,
                odd_indices = self.odd_indices,
                even_indices = self.even_indices,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                permute = self.permute
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        )
        self.maxpool = nn.MaxPool3d(kernel_size=[1,3,3],stride=[1,2,2],padding=[0,1,1])

        self.res_layer = []

        for i, n_layers in enumerate(self.layers):
            stride = self.strides[i]
            layer_name = f'conv_SE_res_block{i+1}'
            out_c = self.base_channels *2 **i
            self.add_module(
                layer_name,
                self.make_res_layer(self.in_c,out_c,n_layers,stride,expansion=self.expansion,permute=self.permute)
            )
            self.res_layer.append(layer_name)
            self.in_c = out_c * self.expansion
    
    def make_res_layer(self,in_channels,out_channels,res_repeat,stride,expansion,permute=False):

        model = nn.Sequential()

        model.add_module('SSE_res0',
                        self.blocks(in_channels=in_channels,
                                    out_channels = out_channels,
                                    basis = self.basis,
                                    scale_size = self.scale_size,
                                    stride = stride,
                                    odd_indices = self.odd_indices,
                                    even_indices = self.even_indices,
                                    downsample = True,
                                    expansion = expansion,
                                    permute = self.permute))
        
        in_channels2 = out_channels * expansion

        for idx in range(1,res_repeat):
            model.add_module('SSE_res{}'.format(idx),
                            self.blocks(in_channels=in_channels2,
                                         out_channels=out_channels,
                                         basis=self.basis,
                                         scale_size=self.scale_size,
                                         odd_indices = self.odd_indices,
                                         even_indices = self.even_indices,
                                         stride=1,
                                         expansion=expansion,
                                         permute = self.permute))
        
        return model
    
    def train(self,mode=True):
        super(SSE_ResNet,self).train(mode)
        
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    
 
    def forward(self,x):
        B = x.size(0)
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i,layer_name in enumerate(self.res_layer):
            res_layer = getattr(self,layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                x_ = SEMaxProjection(x[:B])
                y_ = SEMaxProjection(x[B:])
                outs.append(x_+y_)
            
        return tuple(outs)