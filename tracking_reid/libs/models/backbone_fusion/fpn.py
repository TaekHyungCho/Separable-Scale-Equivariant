import torch
import torch.nn as nn
import torch.nn.functional as F

class REID_FPN(nn.Module):

    def __init__(self):
        super(REID_FPN, self).__init__()

        self.c2 = 512
        self.c3 = 1024
        self.c4 = 2048
        self.out_c = 512
        
        # 1x1 convs to reduce channels (lateral conv)
        self.reduce_conv2 = nn.Conv2d(self.c2, self.out_c, 1)
        self.reduce_conv3 = nn.Conv2d(self.c3, self.out_c, 1)
        self.reduce_conv4 = nn.Conv2d(self.c4, self.out_c, 1)
        
        # 3x3 convs for smoothing
        self.smooth_conv = nn.Conv2d(self.out_c, self.out_c, 3, padding=1)
        
    def _fuse_features(self, p2, p3, p4):
        """
        Simple top-down fusion
        """
        # Reduce channels
        p2 = self.reduce_conv2(p2)
        p3 = self.reduce_conv3(p3)
        p4 = self.reduce_conv4(p4)
        
        # Top-down fusion
        p3_fused = p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p2_fused = p2 + F.interpolate(p3_fused, size=p2.shape[2:], mode='bilinear', align_corners=False)
        
        # Smooth
        output = self.smooth_conv(p2_fused)
        
        return output