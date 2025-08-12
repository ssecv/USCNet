#class APG
#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam.myTransformer import OneWayTransformer
from sam2.modeling.sam.prompt_encoder import PromptEncoder, PositionEmbeddingRandom
from sam2.modeling.sam.weight_init import weight_init


class APG(nn.Module):
    def __init__(self):
        super(APG, self).__init__()

        # APG  module initalization
        self.oneTransformer = OneWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        )
        #
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
        # APG module initalization  end
        #initial needed for  convergencey,important？
        weight_init(self)

    def forward(self, image_embeddings):
        #
        coarse_map_out = self.coarsehead(image_embeddings)

        coarse_map = torch.sigmoid(coarse_map_out)
        # BG SOD COD
        coarseBGAttention = coarse_map[:, 0, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseSODAttention = coarse_map[:, 1, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        coarseCODAttention = coarse_map[:, 2, :, :].unsqueeze(1)  # torch.Size([bs, 1, 22, 22])
        #
        BG_prompt = image_embeddings * coarseBGAttention
        SOD_prompt = image_embeddings * coarseSODAttention  # torch.Size([1, 256, 22, 22])
        COD_prompt = image_embeddings * coarseCODAttention
        #
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
        # ARM intra-inter-SPQ
        tokens = dynamic_tokens + static_tokens
        print("tokens.shape", tokens.shape)
        # tokens = dynamic_tokens #no inter-SPQ
        # tokens = static_tokens #no intra-SPQ

        image_pe = self.pe_layer([22, 22]).unsqueeze(0)
        # one-way-transformer
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # Run the transformer
        hs = self.oneTransformer(src, pos_src, tokens)  # torch.Size([48,100, 256])#selfattention
        BG_prompt = hs[:, :4, :]  # torch.Size([48, 50, 256])
        sod_prompt = hs[:, 4:8, :]  # torch.Size([48, 50, 256])
        cod_prompt = hs[:, 8:, :]  # torch.Size([48, 50, 256])
        # ----APG module end

        return coarse_map, BG_prompt, sod_prompt, cod_prompt

