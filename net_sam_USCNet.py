from segment_anything import sam_model_registry
import torch
import torch.nn as nn


class SAM_USCNet(nn.Module):

    def __init__(self, cfg):
        super(SAM_USCNet, self).__init__()
        self.cfg = cfg

        # print("load vit_b")
        # self.sam, img_embedding_size = sam_model_registry["vit_b"](image_size=352,
        #                                                            num_classes=1,
        #                                                            checkpoint="/home/SAM/sam_vit_b_01ec64.pth",
        #                                                            pixel_mean=[0, 0, 0],
        #                                                            pixel_std=[1, 1, 1])
        # print("load vit_l")
        # self.sam, img_embedding_size = sam_model_registry["vit_l"](image_size=352,
        #                                                            num_classes=1,
        #                                                            checkpoint="/home/SAM/sam_vit_l_0b3195.pth",
        #                                                            pixel_mean=[0, 0, 0],
        #                                                            pixel_std=[1, 1, 1])

        print("load vit_h")
        self.sam, img_embedding_size = sam_model_registry["vit_h"](image_size=352,
                                                                   num_classes=1,
                                                                   checkpoint="/home/desktop/benchmark-code/SAM-USC12K-ViT-h-baseline+adapter/sam_vit_h_4b8939.pth",
                                                                   pixel_mean=[0, 0, 0],
                                                                   pixel_std=[1, 1, 1])

        # self.sam_encoder = self.sam.image_encoder

        select = 0

        # for n, p in self.sam2image.named_parameters():
        #     p.requires_grad = False

        for n, p in self.sam.named_parameters():
            print(n)
            if "image_encoder" in n:
                p.requires_grad = False
            if "prompt_generator" in n:  # prompt
                print("adapter", n)
                p.requires_grad = True

            if "mask_downsample" in n:
                p.requires_grad = False

            if "prompt_encoder" in n:
                p.requires_grad = False

            if "mask_decoder" in n:
                p.requires_grad = False


            if p.requires_grad == True:
                print(n)
                
                print("true")
                select += len(p.reshape(-1))

        print("select:", select / 1e6)

            
        if self.cfg is not None and self.cfg.snapshot:
                print('load checkpoint')
                self.load_state_dict(torch.load(self.cfg.snapshot))


    def forward(self, x, multimask_output=True, image_size=None):

        coarse_map, Background_outputs, sod_outputs,cod_outputs= self.sam(batched_input=x, multimask_output=multimask_output, image_size=image_size)

        return coarse_map, Background_outputs, sod_outputs,cod_outputs

       #mistake
        # x = self.sam(batched_input=x, multimask_output=multimask_output, image_size=image_size)

        # return x


if __name__ == "__main__":

    net = SAM_USCNet().cuda()
    out = net(torch.rand(1, 3, 512, 512).cuda(), 1, 512)
    parameter = 0
    select = 0

    for n, p in net.named_parameters():
        parameter += len(p.reshape(-1))
        if p.requires_grad == True:
            select += len(p.reshape(-1))
    print(select / parameter * 100)

    print(out['masks'].shape)
