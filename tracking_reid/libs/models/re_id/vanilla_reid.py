import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet
from tracking_reid.libs.models.backbone_fusion.fpn import REID_FPN

class KITTI_REID_Vanilla(REID_FPN):

    def __init__(self,**kwargs):

        super().__init__()

        self.backbone = ResNet(depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages= -1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            init_cfg = None)
        
        self.input_dims = kwargs['input_dims']
        self.output_dims = kwargs['output_dims']
        self.pretrained_backbone_ckpt_path = kwargs['pretrained_backbone_path']

        self.norm2 = nn.InstanceNorm2d(num_features=512,affine=True)
        self.norm3 = nn.InstanceNorm2d(num_features=1024,affine=True)
        self.norm4 = nn.InstanceNorm2d(num_features=2048,affine=True)
        
        self.re_id_head = nn.Sequential(
            nn.Conv2d(self.input_dims, self.input_dims//2, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(num_groups=32, num_channels=self.input_dims//2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_dims//2, self.output_dims, kernel_size = 1, stride = 1, padding = 0, bias = True),
        )

        self._load_pretrained_backbone()

        self._freeze_backbone()

        self._init_weights()
    
    def _load_pretrained_backbone(self):

        backbone_weights = torch.load(self.pretrained_backbone_ckpt_path)
        self.backbone.load_state_dict(backbone_weights, strict=False)
    
    def _freeze_backbone(self):

        """
        Freeze all backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        """
        Initialize weights for fusion layers and re-id head.
        """
        backbone_modules = set(self.backbone.modules())
        
        for name, module in self.named_modules():
            # Skip backbone and its submodules
            if module in backbone_modules: 
                continue    ## Already initialized
            
            if 're_id_head' in name:
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.001)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.GroupNorm):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, (nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.InstanceNorm2d):
                if module.affine:
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def _post_processing_featuremaps(self,p2,p3,p4):
        p2 = self.norm2(p2)
        p3 = self.norm3(p3)
        p4 = self.norm4(p4)
        
        return p2,p3,p4

    def train(self, mode=True):
        """
        Override train() to keep backbone in eval mode.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input):
        """Forward pass with bottom-up and top-down feature fusion."""

        p2, p3, p4 = self.backbone(input)
        p2, p3, p4 = self._post_processing_featuremaps(p2,p3,p4)

        output = self._fuse_features(p2,p3,p4)
        reid_features = self.re_id_head(output)
        return reid_features
