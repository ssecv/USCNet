from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_image_finetune_oriAPG import SAM2ImagePredictor

import torch
import torch.nn as nn


class SAM_baseline(nn.Module):

    def __init__(self, cfg):
        super(SAM_baseline, self).__init__()
        self.cfg = cfg

        #sam2_hiera_large
        self.sam2_checkpoint = "/home/segment-anything-2-main/checkpoints/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"

        # self.sam2_checkpoint = "/home/segment-anything-2-main/checkpoints/sam2_hiera_base_plus.pt"
        # self.model_cfg = "sam2_hiera_b+.yaml"

        # self.sam2_checkpoint = "/home/segment-anything-2-main/checkpoints/sam2_hiera_tiny.pt"
        # self.model_cfg = "sam2_hiera_t.yaml"

        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint , device="cuda")
        self.sam2image  = SAM2ImagePredictor(self.sam2_model)

        select=0

        # for n, p in self.sam2image.named_parameters():
        #     p.requires_grad = False

        for n, p in self.sam2image.named_parameters():
            # print(n)
            if "image_encoder" in n:
                print("encoder frozon", n)
                p.requires_grad = False
            if "prompt_generator" in n:  # prompt
                print("adapter", n)
                p.requires_grad = True
            if "mask_downsample" in n:
                p.requires_grad = False
            if "memory_attention" in n:
                p.requires_grad = False
            if "emory_encoder" in n:
                p.requires_grad = False
            if "sam_prompt_encoder" in n:
                p.requires_grad = False
            if "sam_mask_decoder" in n:
                p.requires_grad = False
            if "obj_ptr_proj" in n:
                # p.requires_grad = True
                p.requires_grad = False

            if p.requires_grad == True:
                select += len(p.reshape(-1))

        print("select:", select / 1e6)


        total_params = sum(p.numel() for p in self.sam2image.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.sam2image.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # input("pause")

        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            self.load_state_dict(torch.load(self.cfg.snapshot))

    def forward(self, x, multimask_output=False, image_size=None):
        coarse_map, Background_outputs, sod_outputs, cod_outputs,SOD_tokens_out, COD_tokens_out = self.sam2image(batched_input=x, multimask_output=multimask_output, image_size=image_size)

        # print(masks.shape)
        return coarse_map, Background_outputs, sod_outputs, cod_outputs,SOD_tokens_out, COD_tokens_out





