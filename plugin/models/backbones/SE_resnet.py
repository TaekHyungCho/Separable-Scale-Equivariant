import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from plugin.utils import SEConv_Z2_H, SEConv_H_H, SEConv_H_H_1x1, SEMaxProjection
from plugin.utils.basis import get_basis,get_basis_with_filename

class SE_ResBlock(BaseModule):

    def __init__(self,in_channels,out_channels,basis,scale_size,
                stride,expansion,permute=False,downsample=None,init_cfg=None):

        super(SE_ResBlock,self).__init__(init_cfg)
        assert in_channels % 2 == 0

        
        if downsample is not None:
            self.downsample = nn.Sequential(
                SEConv_H_H_1x1(
                    in_channels= in_channels,
                    out_channel= out_channels * expansion,
                    stride=stride
                ),
                nn.BatchNorm3d(out_channels * expansion)
            )
        else:
            self.downsample = downsample

        self.conv1 = nn.Sequential(
                        SEConv_H_H_1x1(in_channels=in_channels,
                            out_channel= out_channels,
                            stride=1,
                            scale_size=scale_size),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(True)
                        )

        self.conv2 = nn.Sequential(
                        SEConv_H_H(in_channels=out_channels,
                            out_channels= out_channels,
                            kernel_size=5,padding=2,stride=stride,
                            scale_size=scale_size,
                            basis = basis,
                            permute = permute),
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
        
    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = F.relu(out)
        return out


@MODELS.register_module()
class SE_ResNet(BaseModule):

    arch_settings = {
        50: (SE_ResBlock,(3, 4, 6, 3),4),
        101: (SE_ResBlock,(3, 4, 23, 3),4)
    }

    def __init__(self,
                depth,
                save_dir=None,
                filename=None,
                scale_size = 1,
                base_channels = 64,
                strides=(1, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                norm_eval=True,
                permute = False,
                init_cfg=None):
        
        super(SE_ResNet,self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
                
        self.depth = depth
        self.strides = strides
        self.out_indices = out_indices
        self.norm_eval = norm_eval
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
            SEConv_Z2_H(
                in_channels= 3,
                out_channels = 64,
                basis = self.basis_Z2_H,
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

        model.add_module('SE_res0',
                        self.blocks(in_channels=in_channels,
                                    out_channels = out_channels,
                                    basis = self.basis,
                                    scale_size = self.scale_size,
                                    stride = stride,
                                    downsample = True,
                                    expansion = expansion,
                                    permute = permute))
        
        in_channels2 = out_channels * expansion

        for idx in range(1,res_repeat):
            model.add_module('SE_res{}'.format(idx),
                            self.blocks(in_channels=in_channels2,
                                         out_channels=out_channels,
                                         basis=self.basis,
                                         scale_size=self.scale_size,
                                         stride=1,
                                         expansion=expansion,
                                         permute = permute))
        
        return model
    
    def train(self,mode=True):
        super(SE_ResNet,self).train(mode)
        
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i,layer_name in enumerate(self.res_layer):
            res_layer = getattr(self,layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                x_ = SEMaxProjection(x)
                outs.append(x_)

        return tuple(outs)