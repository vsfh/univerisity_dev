# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.attention import SpatialTransformer

import numpy as np
import torchvision.models as models

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        self.model.avgpool = None
        self.model.head = None
        self.model.norm = None

    def forward(self, x):
        x = self.model.features(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class TROGeo(nn.Module):
    def __init__(self, emb_size=768):
        super(TROGeo, self).__init__()

        base_model = SwinTransformer()
        self.query_model = base_model
        self.reference_model = base_model

        self.combine_clickptns_conv = double_conv(4, 3)

        self.cross_attention = SpatialTransformer(in_channels=emb_size, n_heads=12, d_head=64, depth=1,
                                                  context_dim=emb_size)

        self.fcn_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1),
        )
        
        self.coodrs_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=emb_size, out_channels=emb_size // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size // 2, 1, kernel_size=1),
        )


    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)

        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))

        query_fvisu = self.query_model(query_imgs)
        reference_fvisu = self.reference_model(reference_imgs)

        context = rearrange(query_fvisu, 'b c h w -> b (h w) c').contiguous()
        fused_features = self.cross_attention(x=reference_fvisu, context=context)
        outbox = self.fcn_out(fused_features)

        coodrs = self.coodrs_out(fused_features)

        return outbox, coodrs
