# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmcv_custom.cnn import build_norm_layer
from mmcv_custom.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine_custom import ConfigDict
from mmengine_custom.model import BaseModule, ModuleList
from torch import Tensor

from mmdet_custom.utils import ConfigType, OptConfigType


from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class PEMTransformer(BaseModule):
    """Decoder of PEM.
    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`,
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            PEMTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_pos: Tensor, key_pos: Tensor, key_padding_mask: Tensor,
                **kwargs) -> Tensor:
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)


# class LocalRepresentation(nn.Module):
#     """
#     Local Representation module for generating feature vectors from input features.

#     Args:
#         d_model (int): The dimensionality of the input and output feature vectors (default: 256).

#     Attributes:
#         to_query_3x3 (nn.Conv2d): 3x3 depth-wise convolutional layer for local feature extraction.
#         bn (nn.BatchNorm2d): Batch normalization layer.
#         out (nn.Linear): Linear transformation layer.
#         d_model (int): The dimensionality of the input and output feature vectors.

#     Methods:
#         forward(self, x): Forward pass through the LocalRepresentation module.
#     """
#     def __init__(self, d_model=256):
#         super().__init__()

#         self.to_query_3x3 = nn.Conv2d(d_model, d_model, 3, groups=d_model, padding=1)
#         self.bn = nn.SyncBatchNorm(d_model)
#         self.out = nn.Linear(d_model, d_model)

#         self.d_model = d_model

#     def forward(self, x):
#         # Retrieve input tensor shape
#         B, C, H, W = x.shape

#         # Apply pre-normalisation followed by 3x3 local convolution to extract local features
#         x = self.bn(x)
#         x_3x3 = self.to_query_3x3(x)

#         # Reshape the local features and permute dimensions for linear transformation
#         return self.out(x_3x3.view(B, self.d_model, H*W).permute(0, 2, 1))

class LocalRepresentation(nn.Module):
    """
    Local Representation module for generating feature vectors from input features.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).

    Attributes:
        to_query_3x3 (nn.Conv1d): 3x3 depth-wise convolutional layer for local feature extraction.
        bn (nn.BatchNorm1d): Batch normalization layer.
        out (nn.Linear): Linear transformation layer.
        d_model (int): The dimensionality of the input and output feature vectors.

    Methods:
        forward(self, x): Forward pass through the LocalRepresentation module.
    """
    def __init__(self, d_model=256):
        super().__init__()

        self.to_query_3x3 = nn.Conv1d(d_model, d_model, 3, groups=d_model, padding=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, x):
        # Retrieve input tensor shape
        B, N, C = x.shape

        # Permute the dimensions to match the expected input format for Conv1d (B, C, N)
        x = x.permute(0, 2, 1)

        # Apply pre-normalisation followed by 3x3 local convolution to extract local features
        x = self.bn(x)
        x_3x3 = self.to_query_3x3(x)

        # Permute dimensions back and apply linear transformation
        x_3x3 = x_3x3.permute(0, 2, 1)
        return self.out(x_3x3)

class PEM_CA(nn.Module):
    """
    Prototype-based Masked Cross-Attention module.

    This module implements a variant of the cross-attention mechanism for use in segmentation heads.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).
        nhead (int): The number of attention heads (default: 8).

    Attributes:
        to_query (LocalRepresentation): Module for converting input to query representations.
        to_key (nn.Sequential): Sequential module for transforming input to key representations.
        proj (nn.Linear): Linear transformation layer.
        final (nn.Linear): Final linear transformation layer.
        alpha (nn.Parameter): Parameter for scaling in the attention mechanism.
        num_heads (int): Number of attention heads.

    Methods:
        with_pos_embed(self, tensor, pos): Adds positional embeddings to the input tensor.
        most_similar_tokens(self, x, q, mask=None): Finds the most similar tokens based on content-based attention.
        forward(self, q, x, memory_mask, pos, query_pos): Forward pass through the PEM_CA module.
    """

    def __init__(self, d_model=256, nhead=8):
        super().__init__()

        self.feature_proj = LocalRepresentation(d_model)
        self.query_proj = nn.Sequential(nn.LayerNorm(d_model),
                                    nn.Linear(d_model, d_model))

        self.proj = nn.Linear(d_model, d_model)
        self.final = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.ones(1, 1, d_model))
        self.num_heads = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def most_similar_tokens(self, x, q, mask=None):
        # Retrieve input tensors shapes
        B, N, C = x.shape

        Q, D = q.shape[1], C // self.num_heads

        # Reshape tensors in multi-head fashion
        x = x.view(B, N, self.num_heads, D).permute(0, 2, 1, 3)
        q = q.view(B, Q, self.num_heads, D).permute(0, 2, 1, 3)

        # Compute similarity scores between features and queries
        sim = torch.einsum('bhnc, bhqc -> bhnq', x, q)

        # Apply mask to similarity scores if provided
        if mask is not None:
            mask = (mask.flatten(2).permute(0, 2, 1).detach() < 0.0).bool()
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            mask[torch.all(mask.sum(2) == mask.shape[2], dim=2)] = False
            sim.masked_fill_(mask, float('-inf'))

        # Find indices of most similar tokens
        most_similar_indices = torch.argmax(sim, dim=2)

        # Gather most similar tokens
        return torch.gather(x, 2, most_similar_indices.unsqueeze(-1).expand(-1, -1, -1, D)).permute(0, 2, 1, 3).reshape(B, Q, C)

    def forward(self, tgt, memory, memory_mask, pos, query_pos):
        res = tgt

        # Add positional embeddings to input tensors
        # input(memory.shape)
        # input(pos.shape)
        memory, tgt = self.with_pos_embed(memory, pos), self.with_pos_embed(tgt, query_pos)


        # Project input tensors
        memory = self.feature_proj(memory)  # BxDxHxW
        tgt = self.query_proj(tgt)  # BxQxD

        # Normalize input tensors
        memory = torch.nn.functional.normalize(memory, dim=-1)
        tgt = torch.nn.functional.normalize(tgt, dim=-1)

        # input(memory.shape) # 8,300,256
        # input(tgt.shape) # 100,8,256
        # Find the most similar feature token to each query
        memory = self.most_similar_tokens(memory, tgt, memory_mask)  # BxQxD

        # input(memory.shape) # 
        # input(tgt.shape) # 
        # input(self.proj(memory * tgt).shape)
        # input(self.alpha)


        # Perform attention mechanism with projection and scaling
        out = nn.functional.normalize(self.proj(memory * tgt), dim=1) * self.alpha + memory  # BxQxD

        # Final linear transformation
        out = self.final(out)  # BxQxD

        return out + res


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class PEMTransformerDecoderLayer(BaseModule):
    """Implements decoder layer in PEM transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     ffn_drop=0.,
                     batch_first=True,
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()

        self.self_attn= SelfAttentionLayer(
            d_model = self.self_attn_cfg.embed_dims,
            nhead = self.self_attn_cfg.num_heads,
            dropout = self.self_attn_cfg.attn_drop,
            normalize_before = self.self_attn_cfg.batch_first
        )

        self.cross_attn = PEM_CA(                    
            d_model = self.cross_attn_cfg.embed_dims,
            nhead = self.cross_attn_cfg.num_heads
        )
        
        self.ffn = FFNLayer(
            d_model = self.ffn_cfg.embed_dims,
            dim_feedforward = self.ffn_cfg.feedforward_channels,
            dropout = self.ffn_cfg.ffn_drop,
            normalize_before = self.ffn_cfg.batch_first,
        )
            
        self.embed_dims = self.self_attn_cfg.embed_dims
        
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        
        query = self.cross_attn(
            query,
            key,  # key=value=memory
            # attn_mask=cross_attn_mask,
            memory_mask=key_padding_mask,
            pos=key_pos,
            query_pos=query_pos,
            **kwargs)        

        query = self.norms[0](query)

        query = self.self_attn(
            query,
            query_pos=query_pos,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query
