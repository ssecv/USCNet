#!/usr/bin/python3
# coding=utf-8
import os
import sys

from metrics.miou import SegmentationMetric

sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

import argparse

plt.ion()
import torch

from net_sam_USCNet import SAM_USCNet as network
from dataloaders import make_data_loader


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='dataset name (default: pascal)')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='set the checkpoint name')
    # parser.add_argument('--base-size', type=int, default=352,
    #                     help='base image size')
    parser.add_argument('--crop-size', type=int, default=352,
                        help='crop image size')

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
    net.train(False)
    net.cuda()
    with torch.no_grad():
        metric = SegmentationMetric(3)


        for step, sample in enumerate(val_loader):
            if step % 100 == 0:
                print('step:', step)

            image, target = sample[0]['image'], sample[0]['label']
            image, target = image.cuda(), target.cuda()

            # image, shape = image.cuda().float(), (H, W)

            coarse_map, Background_outputs, sod_outputs,cod_outputs= net(image)
            

            Background_outputs = Background_outputs['masks']
            sod_outputs = sod_outputs['masks']
            cod_outputs = cod_outputs['masks']
            
            logits = torch.cat((Background_outputs, sod_outputs, cod_outputs), dim=1)  # 16 3 256 256
            logits = F.softmax(logits, dim=1)  # [48, 3, 352, 352]
            imgPredict = logits.data.max(1)[1].cpu().numpy()

            imgPredict = torch.from_numpy(imgPredict).cuda()
            # .long()
            imgPredict = imgPredict.long().cpu()
            target = target.long().cpu()

            hist = metric.addBatch(imgPredict, target, ignore_labels='')  #


        print(hist.shape)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        IoU = metric.IntersectionOverUnion()
        mIoU = metric.meanIntersectionOverUnion()
        # MCA = metric.meanCODSODConfusionAccuracy()
        CSCS = metric.CSCS()
        # MCA3=  metric.meanCODSODConfusionAccuracy3()
        # MCA4 = metric.meanCODSODConfusionAccuracy4()
        CS = metric.printColSum()


        print('hist is :\n', hist)
        print('PA is : %f' % pa)
        print('cPA is :', cpa)
        print('mPA is : %f' % mpa)
        print('IoU is : ', IoU)
        print('mIoU is : ', mIoU)
        # print('MCA is : ', MCA)
        print('CSCS is : ', CSCS)
        # print('MCA3 is : ', MCA3)
        # print('MCA4 is : ', MCA4)


                # logits = self.net(image)  # 16 2 512 512
                # logits = F.interpolate(logits, size=shape, mode='bilinear')  # 1 3 512 512
                #
                # # print("logits.shape:", logits.shape)
                # index_logits = logits.data.max(1)[1].cpu().numpy()  #0，1（co），2(so)
                # # print("index_logits.shape:", index_logits.shape)  # 1 512 512



                # index_logits[index_logits == 2] = 255
                # index_logits[index_logits != 255] = 0
                # predSOD_PRE = index_logits.squeeze(0)  # 512 512
                # # print("predCOD_PRE.shape:", predCOD_PRE.shape)
                # # input("pause")
                #
                # head = self.cfg.datapath + '/' + save_name_SOD
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head + '/' + name[0] + '.png', np.round(predSOD_PRE))









if __name__ == '__main__':
    test(network)
