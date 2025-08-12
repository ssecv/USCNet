#!/usr/bin/python3
# coding=utf-8
import os
import sys
from metrics.miou import SegmentationMetric
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm  #
import argparse
import torch
from dataloaders import make_data_loader
from net_sam_baseline import SAM_baseline as network


def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 1] = [255, 0, 0]
    color_mask[mask == 2] = [0, 255, 0]
    new_mask = Image.fromarray(color_mask)
    return new_mask
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--dataset', type=str, default='pascal', help='dataset name (default: pascal)')
    parser.add_argument('--snapshot', type=str, default="/home/benchmark-code/SAM2-USC12K-Hirea-l-adapter-APG-twoway/save/weight-SAM2-APG-twoway-doogmaxdata/model-46", help='set the checkpoint name')
    parser.add_argument('--base-size', type=int, default=352, help='base image size')
    parser.add_argument('--crop-size', type=int, default=352, help='crop image size')

    parser.parse_args()
    return parser.parse_args()

def test(Network):
    ## dataset
    args = parser()
    cfg = args
    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    ## network
    net = Network(cfg)
    net.load_state_dict(torch.load(cfg.snapshot))
    net.train(False)
    net.cuda()
    with torch.no_grad():
        metric = SegmentationMetric(3)


        for sample, name in tqdm(val_loader, desc="Processing"):
            img_name = name[0]

            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()

            coarse_map, Background_outputs, sod_outputs, cod_outputs,SOD_tokens_out, COD_tokens_out = net(image, multimask_output=False,
                                                                           image_size=352)
            logits = torch.cat((Background_outputs, sod_outputs, cod_outputs), dim=1) 

            logits = F.softmax(logits, dim=1)  # [48, 3, 352, 352]
            prediction = logits.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            prediction = colorize_mask(prediction)
            path = os.path.join(r"/home/benchmark-result/USC12K", 'SAM2-ARM-epoch46-train-nointer')
            if not os.path.exists(path):
                os.makedirs(path)
            prediction.save(os.path.join(path, img_name + '.png'))

if __name__ == '__main__':
    test(network)