# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)#不需要iou_token
        # self.num_mask_tokens = num_multimask_outputs + 1  #输出的mask数量这里包含了背景
        self.num_mask_tokens = num_multimask_outputs  # 输出的mask数量=2
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Prepare output
        return masks

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = self.mask_tokens.weight  # torch.Size([2, 256])
        # output_tokens扩展到和sparse_prompt_embeddings一样的维度
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)

        src = torch.repeat_interleave(image_embeddings, 1, dim=0)  # 直接1，image_embeddings本来就是batch=48

        src = src + dense_prompt_embeddings


        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)#

        # print("hs", hs.shape)#[48, 2, 256]
        # print("src", src.shape)#[48, 484, 256]
        # input("pause")

        # iou_token_out = hs[:, 0, :]
        # print(self.num_mask_tokens)
        # input("pause")
        # mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # torch.Size([48, 256, 484])

        #
        # print(self.num_mask_tokens)#1
        # input("pause")
        mask_tokens_out = hs[:, 0: (0 + self.num_mask_tokens), :]  # torch.Size([48, 256, 484])

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # torch.Size([48, 256, 22, 22])

        upscaled_embedding = self.output_upscaling(src)  # torch.Size([4, 32, 88, 8])

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h,
                                                                       w)  # [1, 4, 256, 256], 256 = 4 * 64,
        # the size of image embeddings

        # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)
        # print("iou_pred", iou_pred.shape)
        # input("pause")

        return masks


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
