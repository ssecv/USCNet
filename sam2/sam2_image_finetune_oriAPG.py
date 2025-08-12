# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms

from sam2.modeling.sam.myTransformer import MyWayTransformer
import torch.nn as nn
from sam2.modeling.sam.prompt_encoder import PromptEncoder, PositionEmbeddingRandom

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(A, B):
    """
    计算两个特征分布 A 和 B 的 FID 分数。

    参数:
    A: ndarray, shape (1, 1, 256) - 第一个特征
    B: ndarray, shape (1, 1, 256) - 第二个特征

    返回:
    FID 分数 (float)
    """
    # 确保输入特征形状为 [N, D]
    A = A.reshape(-1, A.shape[-1])  # 从 [1, 1, 256] 转为 [1, 256]
    B = B.reshape(-1, B.shape[-1])  # 从 [1, 1, 256] 转为 [1, 256]

    # Step 1: 计算均值
    mu1, mu2 = A.mean(axis=0), B.mean(axis=0)

    # Step 2: 计算协方差
    if A.shape[0] > 1:
        sigma1 = np.cov(A, rowvar=False)
    else:
        sigma1 = np.zeros((A.shape[1], A.shape[1]))

    if B.shape[0] > 1:
        sigma2 = np.cov(B, rowvar=False)
    else:
        sigma2 = np.zeros((B.shape[1], B.shape[1]))

    print("sigma1:", sigma1)
    print("sigma2:", sigma2)

    # Step 3: 计算 FID
    diff = mu1 - mu2
    # 矩阵开方
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # 处理 sqrtm 返回的复数情况
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # FID 公式
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def rbf_kernel(x, y, gamma=None):
    """
    计算 RBF 核矩阵
    参数:
        x: ndarray or torch.Tensor, shape (N, D) - 第一个样本集
        y: ndarray or torch.Tensor, shape (M, D) - 第二个样本集
        gamma: float, RBF 核的带宽参数 (默认使用 1 / D)
    返回:
        核矩阵，形状为 (N, M)
    """
    # 如果输入是 Torch 张量，将其转换为 NumPy 数组
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if gamma is None:
        gamma = 1.0 / x.shape[1]
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    return np.exp(-gamma * np.sum(diff ** 2, axis=-1))


def calculate_mmd(A, B, kernel='rbf', gamma=None):
    """
    计算 MMD 分数
    参数:
        A: ndarray or torch.Tensor, shape (1, 1, 256) - 第一个特征分布样本
        B: ndarray or torch.Tensor, shape (1, 1, 256) - 第二个特征分布样本
        kernel: str, 核函数类型 ('rbf', 'linear')
        gamma: float, RBF 核的带宽参数
    返回:
        MMD^2 值 (float)
    """
    # 如果输入是 Torch 张量，将其转换为 NumPy 数组
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    # 将输入特征从 [1, 1, 256] 展平为 [1, 256]
    A = A.reshape(-1, A.shape[-1])
    B = B.reshape(-1, B.shape[-1])

    if kernel == 'rbf':
        Kxx = rbf_kernel(A, A, gamma)
        Kyy = rbf_kernel(B, B, gamma)
        Kxy = rbf_kernel(A, B, gamma)
    elif kernel == 'linear':
        Kxx = A @ A.T
        Kyy = B @ B.T
        Kxy = A @ B.T
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")

    # 计算 MMD^2
    mmd2 = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    return mmd2





class SAM2ImagePredictor(torch.nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          fill_hole_area (int): If fill_hole_area > 0, we fill small holes in up to
            the maximum area of fill_hole_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps  352
        self._bb_feat_sizes = [
            (88, 88),
            (44, 44),
            (22, 22),
        ]

        # APG module initalization
        self.MyTransformer = MyWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )

        # # APG module initalization
        # self.oneTransformer2 = OneWayTransformer(
        #     depth=2,
        #     embedding_dim=256,
        #     mlp_dim=2048,
        #     num_heads=8,
        # )


        self.coarsehead = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 3, kernel_size=3, padding=1))
        # -version2
        self.BG_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.COD_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SOD_prompt_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # BG COD SODprompt(bs，256,121)[bs，256,30]
        self.BG_prompt_fc = nn.Linear(121, 4)
        self.COD_prompt_fc = nn.Linear(121, 4)
        self.SOD_prompt_fc = nn.Linear(121, 4)
        #
        self.static_token_embedding = nn.Embedding(12, 256)
        self.pe_layer = PositionEmbeddingRandom(256 // 2)  #
         # APG module initalization  end
    # tips from experiments
    # for sam2 ,can directly process the image with 352*352,but for sam1,need to resize weight that design for 1024*1024
    # add parameter or not frozen is good
    # use initial weight is good for convergence
    # use low- and high- resolution is good,it is ok to finetune the low-resolution weight,just two conv layer.
    # Ladder-Slode-Tun is good way to repalce adapter in lowly memory cost（here can aggregate SAM'S attention layers）)
    # for sam2 attention is not adjust that may be a good way to improve the performance
    # adawm 1e-4 or 3e-4is good
    # two way attention is necessary for graident flow to the image encoder maybe can get refined image embedding ,that is vital.
    # Can other methods be found to distinguish between salient and camouflaged targets, rather than just optimizing the model?
    # low-level,fft,visual attention, contrastive learning？  fft is not suitable. maybe contrastive learning is good.
    # try triplet loss,contrastive loss？for future work salient camouflaged  background triplet loss
    # keep original resolution is better 1024,could try.
    # prompt set 4 is enough ,it is resonable cause align the number of prompt with manul way,point bbox（x1,y1,x2,y2)
    # other way should explored to make SAM distiguish saiient and conceled visual attention
    # oneTransformer coarsehead BG_prompt_downsample COD_prompt_downsample SOD_prompt_downsample BG_prompt_fc COD_prompt_fc SOD_prompt_fc static_token_embedding pe_layer

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model)

    def set_image_batch(
        self,
        img_batch,
        # image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        # assert isinstance(image_list, list)
        self._orig_hw = []
        # for image in image_list:
        #     assert isinstance(
        #         image, np.ndarray
        #     ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
        #     self._orig_hw.append(image.shape[:2])

        # # Transform the image to the form expected by the model
        # img_batch = self._transforms.forward_batch(image_list)
        # img_batch = img_batch.to(self.device)

        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"

        logging.info("Computing image embeddings for the provided images...")

        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]


        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}#last

        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")


    def forward(self, batched_input, multimask_output, image_size=None):
        if len(batched_input) == 1:##
            outputs = self.forward_test(batched_input, multimask_output, image_size)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs


    def forward_train(
        self,
        batched_input,
        multimask_output: bool = True,
        image_size=352,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        self.set_image_batch(batched_input)
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )

        # sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
        #     points=None,boxes=None,masks=None,
        # )

        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        #  APG module
        image_embeddings=self._features["image_embed"]
        coarse_map_out = self.coarsehead(image_embeddings)

        coarse_map = torch.sigmoid(coarse_map_out)
        coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])

        BG_prompt = image_embeddings * coarseBGAttention
        SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
        COD_prompt = image_embeddings * coarseCODAttention

        BG_prompt = self.BG_prompt_downsample(BG_prompt)
        SOD_prompt = self.SOD_prompt_downsample(SOD_prompt)
        COD_prompt = self.COD_prompt_downsample(COD_prompt)
        src = image_embeddings
        bs = src.size(0)

        BG_prompt = self.BG_prompt_fc(BG_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        SOD_prompt = self.SOD_prompt_fc(SOD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        COD_prompt = self.COD_prompt_fc(COD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)

        static_tokens = self.static_token_embedding.weight  # prompt#torch.Size([90, 256])
        # # token
        static_tokens = static_tokens.unsqueeze(0).expand(bs, -1, -1)  # torch.Size([48, 90, 256])
        dynamic_tokens = torch.cat((BG_prompt, SOD_prompt, COD_prompt), dim=1)  # torch.Size([48, 90, 256])
        #
        # tokens = dynamic_tokens + static_tokens
        # tokens = dynamic_tokens #print("only dynamic tokens")
        # print("only static tokens")
        # input(111111111111111)
        tokens = static_tokens


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
        queries, keys = self.MyTransformer(src, pos_src, tokens)  # torch.Size([48,100, 256])#selfattention
        hs=queries
        SmartImage_embeddings=keys

        # print("hs.shape", hs.shape)
        # print("SmartImage_embeddings.shape", SmartImage_embeddings.shape)#22,484,256
        SmartImage_embeddings = SmartImage_embeddings.permute(0, 2, 1).reshape(bs, 256, 22, 22)

        BG_prompt = hs[:, :4, :]  # torch.Size([48, 50, 256])
        SOD_prompt= hs[:, 4:8, :]  # torch.Size([48, 50, 256])
        COD_prompt = hs[:, 8:, :]  # torch.Size([48, 50, 256])


        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                bs=image_embeddings.shape[0],
                points=None,boxes=None,masks=None,
            )

        #BG prompt to decoder
        BG_sparse_embeddings = BG_prompt
        BG_low_res_masks, BG_iou_predictions, BG_tokens_out, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=BG_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        BG_masks = self._transforms.postprocess_masks(
            BG_low_res_masks, [352,352]
        )
        BG_low_res_masks = torch.clamp(BG_low_res_masks, -32.0, 32.0)

        #SOD prompt to decoder
        SOD_sparse_embeddings = SOD_prompt
        SOD_low_res_masks, SOD_iou_predictions, SOD_tokens_out, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=SOD_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        SOD_masks = self._transforms.postprocess_masks(
            SOD_low_res_masks, [352,352]
        )
        SOD_low_res_masks = torch.clamp(SOD_low_res_masks, -32.0, 32.0)

        #COD prompt to decoder
        COD_sparse_embeddings = COD_prompt
        COD_low_res_masks, COD_iou_predictions, COD_tokens_out, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=COD_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        COD_masks = self._transforms.postprocess_masks(
            COD_low_res_masks, [352,352]
        )
        COD_low_res_masks = torch.clamp(COD_low_res_masks, -32.0, 32.0)

        return coarse_map_out, BG_masks, SOD_masks, COD_masks





    @torch.no_grad()
    def set_image(self,
                  img_batch,
                  # image: Union[np.ndarray, Image],
                ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # # Transform the image to the form expected by the model
        # if isinstance(image, np.ndarray):
        #     logging.info("For numpy array image, we assume (HxWxC) format")
        self._orig_hw = [img_batch.shape[:2]]
        # elif isinstance(image, Image):
        #     w, h = image.size
        #     self._orig_hw = [(h, w)]
        # else:
        #     raise NotImplementedError("Image format not supported")

        # input_image = self._transforms(image)
        # input_image = input_image[None, ...].to(self.device)

        # assert (
        #     len(input_image.shape) == 4 and input_image.shape[1] == 3
        # ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        # logging.info("Computing image embeddings for the provided image...")

        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")



    #推理
    def forward_test(
            self,
            batched_input,
            multimask_output: bool = False,
            image_size=1024,
            return_logits: bool = False,
            normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        self.set_image(batched_input)  # batch=1,backbone,_features

        # assert self._is_batch, "This function should only be used when in batched mode"
        # if not self._is_image_set:
        #     raise RuntimeError(
        #         "An image must be set with .set_image_batch(...) before mask prediction."
        #     )

        # sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
        #     bs=image_embeddings.shape[0],points=None, boxes=None, masks=None,
        # )

        ##
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        #  APG module
        image_embeddings = self._features["image_embed"]
        # 3
        coarse_map_out = self.coarsehead(image_embeddings)

        coarse_map = torch.sigmoid(coarse_map_out)
        # BG SOD COD
        coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        # prompt  image_embeddings*coarseAttention  BG COD SODprompt
        BG_prompt = image_embeddings * coarseBGAttention
        SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
        COD_prompt = image_embeddings * coarseCODAttention
        # BG COD SOD prompt [1, 256, 22, 22] [1, 256, 11, 11]
        BG_prompt = self.BG_prompt_downsample(BG_prompt)
        SOD_prompt = self.SOD_prompt_downsample(SOD_prompt)
        COD_prompt = self.COD_prompt_downsample(COD_prompt)
        src = image_embeddings
        bs = src.size(0)
        # BG COD SODprompt[bs, 256, 11, 11]，reshape[bs,256,121]，[bs,256,30]，reshape[bs,30,256]
        BG_prompt = self.BG_prompt_fc(BG_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        SOD_prompt = self.SOD_prompt_fc(SOD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        COD_prompt = self.COD_prompt_fc(COD_prompt.reshape(bs, 256, 121)).reshape(bs, 4, 256)
        static_tokens = self.static_token_embedding.weight  # prompt#torch.Size([90, 256]) #
        # # token
        static_tokens = static_tokens.unsqueeze(0).expand(bs, -1, -1)  # torch.Size([48, 90, 256])
        dynamic_tokens = torch.cat((BG_prompt, SOD_prompt, COD_prompt), dim=1)  # torch.Size([48, 90, 256])
        #
        tokens = dynamic_tokens + static_tokens
        # tokens = dynamic_tokens
        # input("only dynamic tokens111111111111111111111111111111111111111111111111111111")
        # tokens = static_tokens

        image_pe = self.pe_layer([22, 22]).unsqueeze(0)
        # one-way-transformer
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # Run the transformer
        queries, keys = self.MyTransformer(src, pos_src, tokens)
        hs = queries # torch.Size([48,100, 256])#selfattention
        SmartImage_embeddings=keys
        SmartImage_embeddings = SmartImage_embeddings.permute(0, 2, 1).reshape(bs, 256, 22, 22)
        BG_prompt = hs[:, :4, :]  # torch.Size([48, 50, 256])
        SOD_prompt = hs[:, 4:8, :]  # torch.Size([48, 50, 256])
        COD_prompt = hs[:, 8:, :]  # torch.Size([48, 50, 256])

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            bs=image_embeddings.shape[0],
            points=None, boxes=None, masks=None,
        )

        # BG prompt to decoder
        BG_sparse_embeddings = BG_prompt
        BG_low_res_masks, BG_iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=BG_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        BG_masks = self._transforms.postprocess_masks(
            BG_low_res_masks, [352, 352]
        )
        BG_low_res_masks = torch.clamp(BG_low_res_masks, -32.0, 32.0)

        # SOD prompt to decoder
        SOD_sparse_embeddings = SOD_prompt
        SOD_low_res_masks, SOD_iou_predictions, SOD_tokens_out, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=SOD_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        SOD_masks = self._transforms.postprocess_masks(
            SOD_low_res_masks, [352, 352]
        )
        SOD_low_res_masks = torch.clamp(SOD_low_res_masks, -32.0, 32.0)

        # COD prompt to decoder
        COD_sparse_embeddings = COD_prompt
        COD_low_res_masks, COD_iou_predictions, COD_tokens_out, _ = self.model.sam_mask_decoder(
            image_embeddings=SmartImage_embeddings,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=COD_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        COD_masks = self._transforms.postprocess_masks(
            COD_low_res_masks, [352, 352]
        )
        COD_low_res_masks = torch.clamp(COD_low_res_masks, -32.0, 32.0)


        # print("SOD_tokens_out:", SOD_tokens_out.shape)
        # print("COD_tokens_out:", COD_tokens_out.shape)

        # input("")

        # SOD_tokens_out COD_tokens_out calculate FID
        # FID_Score = calculate_fid(SOD_tokens_out,COD_tokens_out)
        # print("FID_Score:", FID_Score)
        # input("FID_Score")

        # mmd_score = calculate_mmd(SOD_tokens_out, COD_tokens_out, kernel='rbf', gamma=0.01)
        # FID_Score= mmd_score




        return coarse_map_out, BG_masks, SOD_masks, COD_masks,SOD_tokens_out, COD_tokens_out

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
