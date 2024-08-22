import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmcv_custom.cnn import ConvModule
from mmengine_custom.model import (BaseModule, ModuleList, caffe2_xavier_init)
from mmseg_custom.registry import MODELS
from mmseg_custom.utils import add_prefix
from einops import rearrange
from torch.nn import init, Sequential
import numbers
import numpy as np
import math
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,
                 groups=1,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        scale = 1
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale2 = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1,groups=groups, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1,groups=groups, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        out = x + out * self.scale2
        return out
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class GFE(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,
                 groups = 1,):
        super(GFE, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,groups = groups)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x
class FFRM(BaseModule):
    """Fused Feature Recalibration Module in RoadFormer"""
    def __init__(self, in_chan, out_chan, norm=None):
        super(FFRM, self).__init__()
        self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, norm_cfg=norm)
        self.sigmoid = nn.Sigmoid()
    def init_weights(self) -> None:
        """Initialize weights."""
        caffe2_xavier_init(self.conv_atten, bias=0)
    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))) 
        enhancefeat = torch.mul(x, atten)
        x = x + enhancefeat
        return x
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
class CA(BaseModule):
    """Fused Feature Recalibration Module in RoadFormer with Coordinate Attention"""
    def __init__(self, in_chan, out_chan, norm=None):
        super(CA, self).__init__()
        self.coord_atten = CoordinateAttention(in_chan, in_chan)
    def init_weights(self) -> None:
        """Initialize weights."""
        for m in self.coord_atten.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        atten = self.coord_atten(x)
        x = x + atten
        return x
class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
class GFFM(nn.Module):
    """Heterogeneous Feature Fusion Module in RoadFormer"""
    def __init__(self, feat_scale,dim):
        super().__init__()
        self.gammax = Scale(0)
        self.gammay = Scale(0)
        num_feats = feat_scale[0]*feat_scale[1]
        self.norm = nn.LayerNorm(num_feats)
    def forward(self, x):
        split_dim = x.size(1) // 2
        x, y = torch.split(x, (split_dim, split_dim), dim=1)
        batch_size, channels, height, width = x.size()
        qx = x.view(batch_size, channels, -1)
        kx = x.view(batch_size, channels, -1).permute(0, 2, 1)
        vx = x.view(batch_size, channels, -1)
        qy = y.view(batch_size, channels, -1)
        ky = y.view(batch_size, channels, -1).permute(0, 2, 1)
        vy = y.view(batch_size, channels, -1)
        energy_x = torch.bmm(vx, kx)
        energy_y = torch.bmm(vy, ky)
        attention_x = F.softmax(energy_x, dim=-1)
        attention_y = F.softmax(energy_y, dim=-1)
        outx = torch.bmm(attention_y, qx)
        outy = torch.bmm(attention_x, qy)
        outx = outx.view(batch_size, channels, height, width)
        outy = outy.view(batch_size, channels, height, width)
        outx = self.gammax(outx) + x
        outy = self.gammay(outy) + y
        outx = outx.view(batch_size, channels, -1)
        outy = outy.view(batch_size, channels, -1)
        out = torch.cat((outx, outy), dim=1)
        out = self.norm(out)
        out = out.view(batch_size, channels * 2, height, width)
        return out
class Scale2(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.scale2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
    def forward(self, x, y):
        return x * self.scale1 + y * self.scale2
class MobileNetV2(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(MobileNetV2, self).__init__()   
        hidden_dim = in_channel * 2
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channel, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
    def forward(self, x):       
        x = self.bottleneckBlock(x) * self.scale + x
        return x
@MODELS.register_module()
class RoadFormer2Neck(BaseModule):   
    def __init__(self,
                 in_channels,
                 out_channels,
                 layer = None,
                 norm_cfg=None,      
                 img_scale=None):
        super().__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(out_channels)
        self.in_channels = in_channels
        self.in_channels1 = in_channels
        self.in_channels2 = [i//2 for i in in_channels]
        self.out_channels = out_channels
        self.num_input_levels = len(out_channels)     
        if layer is not None:
            self.layer = layer
        else:
            self.layer = self.num_input_levels
        enhance_blocks = []
        for i in range(self.layer):            
            old_channel = in_channels[i]
            new_channel = out_channels[i]            
            enhance_block = FFRM(old_channel, new_channel, norm=norm_cfg)
            enhance_blocks.append(enhance_block)
        self.enhance_blocks = ModuleList(enhance_blocks)
        self.img_scale = list(img_scale)
        qkvchannels = 64
        qkvchannels = 128
        qkvchannels = 32
        self.global_feature_encoder_rgb = ModuleList([
            GFE(dim=ch // 2, num_heads=8, ffn_expansion_factor=2, qkv_bias=False,groups= qkvchannels)
            for ch in self.in_channels1
        ])        
        self.global_feature_encoder_sne = ModuleList([
            GFE(dim=ch // 2, num_heads=8, ffn_expansion_factor=2, qkv_bias=False,groups= qkvchannels)
            for ch in self.in_channels1
        ])       
        self.local_eature_encoder_rgb = ModuleList([
            MobileNetV2(in_channel = ch, out_channel = ch)
            for ch in self.in_channels2
        ])        
        self.local_eature_encoder_sne = ModuleList([
            MobileNetV2(in_channel = ch, out_channel = ch)
            for ch in self.in_channels2
        ])      
        ca_blocks = []
        for i in range(self.num_input_levels):            
            old_channel = in_channels[i]
            new_channel = out_channels[i]            
            ca_block = CA(old_channel, new_channel, norm=norm_cfg)
            ca_blocks.append(ca_block)
        self.ca_blocks = ModuleList(ca_blocks)
        feat_scales = []
        for i in range(self.num_input_levels):
            feat_scale = (self.img_scale[0]//2**(i+2), self.img_scale[1]//2**(i+2))
            feat_scales.append(feat_scale)
        fuse_blocks = []
        scale_layers = []
        for i in range(self.layer):
            fuse_block = GFFM(feat_scales[i],self.in_channels1[i])
            fuse_blocks.append(fuse_block)
            scale = Scale2()
            scale_layers.append(scale)
        self.fuse_blocks = ModuleList(fuse_blocks) 
        self.scale_layers = ModuleList(scale_layers) 
        self.detail_feature_extractions = ModuleList([
            Mlp(in_features=ch, ffn_expansion_factor=1,)  
            for ch in self.in_channels1[:self.layer]
        ])
    def forward(self, feats):
        assert len(feats) == len(self.in_channels1)
        assert len(feats) == len(self.in_channels2)
        feats_g = []
        feats_l = []        
        feats_rgb_g = []
        feats_sne_g = []
        feats_rgb_l = []
        feats_sne_l = []
        newfeats = []
        loss_decomp = []
        for i, feat in enumerate(feats):
            split_dim = self.in_channels[i] // 2
            feat_rgb, feat_sne = torch.split(feat, (split_dim, split_dim), dim=1)
            feat_rgb_g = self.global_feature_encoder_rgb[i](feat_rgb)
            feat_sne_g = self.global_feature_encoder_sne[i](feat_sne)
            feat_rgb_l = self.local_eature_encoder_rgb[i](feat_rgb)
            feat_sne_l = self.local_eature_encoder_sne[i](feat_sne)
            feats_g.append(torch.cat((feat_rgb_g, feat_sne_g), dim=1))
            feats_l.append(torch.cat((feat_rgb_l, feat_sne_l), dim=1))
        for i in range(self.layer):
            feats_g[i] = self.fuse_blocks[i](feats_g[i])
            feats_l[i] = self.detail_feature_extractions[i](feats_l[i])
            feats_g[i] = self.enhance_blocks[i](feats_g[i])
            feats[i] = self.scale_layers[i](feats_g[i],feats_l[i])
        for i in range(self.num_input_levels):
            feat = feats[i]
            feat_enhanced = self.ca_blocks[i](feat)
            feats[i] = feat_enhanced
        losses = None
        return feats, losses
