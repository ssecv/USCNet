# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder, PositionEmbeddingRandom
from .myTransformer import MyWayTransformer


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            OneWayTransformer: nn.Module,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:

        super().__init__()

        # self.oneTransformer = MyWayTransformer

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)



        # ARM  ARM module initalization------------------------------------------
        self.MyTransformer = MyWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
        # # ARM module initalization

        self.coarsehead = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 3, kernel_size=3, padding=1))
        #
        self.BG_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.COD_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SOD_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.BG_prompt_fc = nn.Linear(121, 4)
        self.COD_prompt_fc = nn.Linear(121, 4)
        self.SOD_prompt_fc = nn.Linear(121, 4)
        #
        self.static_token_embedding = nn.Embedding(12, 256)
        self.pe_layer = PositionEmbeddingRandom(256 // 2)  #
        # ARM module initalization  end----------------------------------

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        # if len(batched_input) == 1:
        #     outputs = self.forward_my_test(batched_input, multimask_output, image_size)  
        # else:
        outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        # print("train----")
        input_images = self.preprocess(batched_input)

        image_embeddings = self.image_encoder(input_images)  #

        # coarse_map_out = self.coarsehead(image_embeddings)
        #
        # # print("coarse_map_out", coarse_map_out.shape)
        # # sigmoid
        # coarse_map = torch.sigmoid(coarse_map_out)
        #
        # # print("coarse_map", coarse_map.shape)
        #
        # # BG SOD COD
        # coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        # coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        # coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        #
        # # prompt image_embeddings*coarseAttention  BG COD SOD prompt
        # BG_prompt = image_embeddings * coarseBGAttention
        # SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
        # COD_prompt = image_embeddings * coarseCODAttention
        #
        # # BG COD SOD prom[1, 256, 22, 22]  [1, 256, 11, 11]
        # BG_prompt = self.BG_prompt_downsample(BG_prompt)
        # SOD_prompt = self.SOD_prompt_downsample(SOD_prompt)
        # COD_prompt = self.COD_prompt_downsample(COD_prompt)
        #
        # src = image_embeddings
        # bs = src.size(0)
        #
        # # BG COD SOD的prompt[bs, 256, 11, 11]，reshape[bs,256,121]，[bs,256,30]，reshape[bs,30,256]
        # BG_prompt = self.BG_prompt_fc(BG_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        # SOD_prompt = self.SOD_prompt_fc(SOD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        # COD_prompt = self.COD_prompt_fc(COD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        #
        # # print("BG_prompt", BG_prompt)
        # # input(111111)
        #
        # static_tokens = self.static_token_embedding.weight  # prompt#torch.Size([90, 256])
        # # # 复制tokens
        # static_tokens = static_tokens.unsqueeze(0).expand(bs, -1, -1)  # torch.Size([48, 90, 256])
        #
        # dynamic_tokens = torch.cat((BG_prompt, SOD_prompt, COD_prompt), dim=1)  # torch.Size([48, 90, 256])
        #
        # #
        # tokens = dynamic_tokens + static_tokens
        #
        # image_pe = self.pe_layer([22, 22]).unsqueeze(0)
        # # one-way-transformer
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # # Run the transformer
        # hs = self.oneTransformer(src, pos_src, tokens)  # torch.Size([48,100, 256])#selfattention
        #
        # BG_prompt = hs[:, :4, :]  # torch.Size([48, 50, 256])
        # sod_prompt = hs[:, 4:8, :]  # torch.Size([48, 50, 256])
        # cod_prompt = hs[:, 8:, :]  # torch.Size([48, 50, 256])

        #  ARM ARM module
        image_embeddings =image_embeddings
        #  attention map
        coarse_map_out = self.coarsehead(image_embeddings)

        coarse_map = torch.sigmoid(coarse_map_out)
        # intra-SPQ
        #
        coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        # prompt image_embeddings*coarseAttention BG COD SOD prompt
        BG_prompt = image_embeddings * coarseBGAttention
        SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
        COD_prompt = image_embeddings * coarseCODAttention
        # BG COD SOD的prompt[1, 256, 22, 22][1, 256, 11, 11]
        BG_prompt = self.BG_prompt_downsample(BG_prompt)
        SOD_prompt = self.SOD_prompt_downsample(SOD_prompt)
        COD_prompt = self.COD_prompt_downsample(COD_prompt)
        src = image_embeddings
        bs = src.size(0)
        # BG COD SOD的prompt[bs, 256, 11, 11]，reshape[bs,256,121]，[bs,256,30]，reshape[bs,30,256]
        BG_prompt = self.BG_prompt_fc(BG_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        SOD_prompt = self.SOD_prompt_fc(SOD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        COD_prompt = self.COD_prompt_fc(COD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)

        # inter-SPQ
        static_tokens = self.static_token_embedding.weight  # prompt#torch.Size([90, 256])
        # # token
        static_tokens = static_tokens.unsqueeze(0).expand(bs, -1, -1)  # torch.Size([48, 90, 256])
        dynamic_tokens = torch.cat((BG_prompt, SOD_prompt, COD_prompt), dim=1)  # torch.Size([48, 90, 256])
        #
        tokens = dynamic_tokens + static_tokens
        image_pe = self.pe_layer([22, 22]).unsqueeze(0)

        # one-way-transformer
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # print("pos_src.shape", pos_src.shape)
        # print("tokens.shape", tokens.shape)
        # print("src.shape", src.shape)
        # pos_src.shape torch.Size([22, 256, 22, 22])#
        # tokens.shape torch.Size([22, 12, 256])  #Q
        # src.shape torch.Size([22, 256, 22, 22]) #K V
        # Run the transformer
        queries, keys = self.MyTransformer(src, pos_src, tokens)  # torch.Size([48,100, 256])#selfattention 会交互
        hs = queries
        SmartImage_embeddings = keys

        # print("hs.shape", hs.shape)
        # print("SmartImage_embeddings.shape", SmartImage_embeddings.shape)#22,484,256
        SmartImage_embeddings = SmartImage_embeddings.permute(0, 2, 1).reshape(bs, 256, 22, 22)

        BG_prompt = hs[:, :4, :]  # torch.Size([48, 50, 256])
        SOD_prompt = hs[:, 4:8, :]  # torch.Size([48, 50, 256])
        COD_prompt = hs[:, 8:, :]  # torch.Size([48, 50, 256])


        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            bs=image_embeddings.shape[0],
            points=None, boxes=None, masks=None
        )

        # BG task ：BG prompt BG mask###################################################
        # sparse_embeddings task_prompt
        BG_sparse_embeddings = BG_prompt
        BG_low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=BG_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        BG_masks = self.postprocess_masks(
            BG_low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        BG_outputs = {
            'masks': BG_masks,
            'low_res_logits': BG_low_res_masks
        }

        # sod task ：sod prompt sod mask###################################################
        # sparse_embeddings task_prompt
        sod_sparse_embeddings = SOD_prompt
        sod_low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sod_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        sod_masks = self.postprocess_masks(
            sod_low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        sod_outputs = {
            'masks': sod_masks,
            'low_res_logits': sod_low_res_masks
        }

        # cod task ：cod prompt cod mask###################################################
        # sparse_embeddings task_prompt
        cod_sparse_embeddings = COD_prompt
        cod_low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=cod_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        cod_masks = self.postprocess_masks(
            cod_low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        cod_outputs = {
            'masks': cod_masks,
            'low_res_logits': cod_low_res_masks
        }
        return coarse_map_out, BG_outputs, sod_outputs, cod_outputs

    #another good adapter
    # def forward_my_test(self, batched_input, multimask_output, image_size):
    #     input_images = self.preprocess(batched_input)
    #
    #     intermediate_results = self.image_encoder(input_images)  # vit-b 12
    #
    #     # intermediate_results[0]-[6],[8]-[14]，[16]-[22]，[24]-[30]
    #     Catfeature1 = torch.cat((intermediate_results[0], intermediate_results[1], intermediate_results[2],
    #                              intermediate_results[3], intermediate_results[4], intermediate_results[5],
    #                              intermediate_results[6]), dim=1)
    #     Catfeature2 = torch.cat((intermediate_results[8], intermediate_results[9], intermediate_results[10],
    #                              intermediate_results[11], intermediate_results[12], intermediate_results[13],
    #                              intermediate_results[14]), dim=1)
    #     Catfeature3 = torch.cat((intermediate_results[16], intermediate_results[17], intermediate_results[18],
    #                              intermediate_results[19], intermediate_results[20], intermediate_results[21],
    #                              intermediate_results[22]), dim=1)
    #     Catfeature4 = torch.cat((intermediate_results[24], intermediate_results[25], intermediate_results[26],
    #                              intermediate_results[27], intermediate_results[28], intermediate_results[29],
    #                              intermediate_results[30]), dim=1)
    #
    #     # 4 cat
    #     Catfeature1 = self.cat_Conv_F1(Catfeature1)
    #     Catfeature2 = self.cat_Conv_F2(Catfeature2)
    #     Catfeature3 = self.cat_Conv_F3(Catfeature3)
    #     Catfeature4 = self.cat_Conv_F4(Catfeature4)
    #     # Catfeature1intermediate_results[2]，Catfeature2intermediate_results[
    #     # 5]，Catfeature3intermediate_results[8]，Catfeature4intermediate_results[11]
    #     multiplyfeature1 = Catfeature1 * intermediate_results[7]
    #     multiplyfeature2 = Catfeature2 * intermediate_results[15]
    #     multiplyfeature3 = Catfeature3 * intermediate_results[23]
    #     multiplyfeature4 = Catfeature4 * intermediate_results[31]
    #     # 4multiply
    #     multiplyfeature1 = self.multiply_Conv_F1(multiplyfeature1)
    #     multiplyfeature2 = self.multiply_Conv_F2(multiplyfeature2)
    #     multiplyfeature3 = self.multiply_Conv_F3(multiplyfeature3)
    #     multiplyfeature4 = self.multiply_Conv_F14(multiplyfeature4)
    #     # 3add
    #     addfeature1 = self.add_Conv_F1(multiplyfeature1 + multiplyfeature2)
    #     addfeature2 = self.add_Conv_F2(addfeature1 + multiplyfeature3)
    #     addfeature3 = self.add_Conv_F3(addfeature2 + multiplyfeature4)
    #     # 1residual
    #     image_embeddings = self.residual_Conv_F1(
    #         addfeature3 + intermediate_results[-1])  # torch.Size([bs, 256, 22, 22])
    #
    #
    #     coarse_map_out = self.coarsehead(image_embeddings)
    #
    #     # print("coarse_map_out", coarse_map_out.shape)
    #     # sigmoid
    #     coarse_map = torch.sigmoid(coarse_map_out)
    #
    #     # print("coarse_map", coarse_map.shape)
    #
    #     #
    #     coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
    #     coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
    #     coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
    #
    #
    #     BG_prompt = image_embeddings * coarseBGAttention
    #     SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
    #     COD_prompt = image_embeddings * coarseCODAttention
    #
    #
    #     BG_prompt = self.BG_prompt_downsample(BG_prompt)
    #     SOD_prompt = self.SOD_prompt_downsample(SOD_prompt)
    #     COD_prompt = self.COD_prompt_downsample(COD_prompt)
    #
    #     src = image_embeddings
    #     bs = src.size(0)
    #
    #     BG_prompt = self.BG_prompt_fc(BG_prompt.reshape(bs, 256, 121)).reshape(bs, 30, 256)
    #     SOD_prompt = self.SOD_prompt_fc(SOD_prompt.reshape(bs, 256, 121)).reshape(bs, 30, 256)
    #     COD_prompt = self.COD_prompt_fc(COD_prompt.reshape(bs, 256, 121)).reshape(bs, 30, 256)
    #
    #     # print("BG_prompt", BG_prompt)
    #     # input(111111)
    #
    #     static_tokens = self.static_token_embedding.weight  #
    #     # #
    #     static_tokens = static_tokens.unsqueeze(0).expand(bs, -1, -1)  # torch.Size([48, 90, 256])
    #
    #     dynamic_tokens = torch.cat((BG_prompt, SOD_prompt, COD_prompt), dim=1)  # torch.Size([48, 90, 256])
    #
    #     #
    #     tokens = dynamic_tokens + static_tokens
    #
    #     image_pe = self.pe_layer([22, 22]).unsqueeze(0)
    #     #
    #     pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    #     # Run the transformer
    #     hs = self.oneTransformer(src, pos_src, tokens)  # torch.Size([48,100, 256])#selfattention
    #
    #     BG_prompt = hs[:, :30, :]  # torch.Size([48, 50, 256])
    #     sod_prompt = hs[:, 30:60, :]  # torch.Size([48, 50, 256])
    #     cod_prompt = hs[:, 90:, :]  # torch.Size([48, 50, 256])
    #
    #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #         bs=image_embeddings.shape[0],
    #         points=None, boxes=None, masks=None
    #     )
    #
    #
    #     BG_sparse_embeddings = BG_prompt
    #     BG_low_res_masks = self.mask_decoder(
    #         image_embeddings=image_embeddings,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=BG_sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=multimask_output,
    #     )
    #
    #     BG_masks = self.postprocess_masks(
    #         BG_low_res_masks,
    #         input_size=(image_size, image_size),
    #         original_size=(image_size, image_size)
    #     )
    #     BG_outputs = {
    #         'masks': BG_masks,
    #         'low_res_logits': BG_low_res_masks
    #     }
    #
    #
    #     # sparse_embeddings 等于task_prompt
    #     sod_sparse_embeddings = sod_prompt
    #     sod_low_res_masks = self.mask_decoder(
    #         image_embeddings=image_embeddings,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=sod_sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=multimask_output,
    #     )
    #     sod_masks = self.postprocess_masks(
    #         sod_low_res_masks,
    #         input_size=(image_size, image_size),
    #         original_size=(image_size, image_size)
    #     )
    #     sod_outputs = {
    #         'masks': sod_masks,
    #         'low_res_logits': sod_low_res_masks
    #     }
    #
    #
    #     # sparse_embeddings task_prompt
    #     cod_sparse_embeddings = cod_prompt
    #     cod_low_res_masks = self.mask_decoder(
    #         image_embeddings=image_embeddings,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=cod_sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=multimask_output,
    #     )
    #     cod_masks = self.postprocess_masks(
    #         cod_low_res_masks,
    #         input_size=(image_size, image_size),
    #         original_size=(image_size, image_size)
    #     )
    #     cod_outputs = {
    #         'masks': cod_masks,
    #         'low_res_logits': cod_low_res_masks
    #     }
    #     return coarse_map_out, BG_outputs, sod_outputs, cod_outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]

        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
