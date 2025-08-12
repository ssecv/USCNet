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
from tqdm import tqdm
import argparse
import torch
from dataloaders import make_data_loader
from net_sam_baseline import SAM_baseline as network


def colorize_mask(mask):
    binary_mask = (mask == 2).astype(np.uint8) * 255  # 类别1的像素值设置为255（白色），其他为0（黑色）
    new_mask = Image.fromarray(binary_mask)
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

        print("2-3minutes...")
        for sample, name in tqdm(val_loader, desc="Processing"):
            img_name = name[0]

            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()

            coarse_map, Background_outputs, sod_outputs, cod_outputs,SOD_tokens_out, COD_tokens_out = net(image, multimask_output=False,
                                                                           image_size=352)
            logits = torch.cat((Background_outputs, sod_outputs, cod_outputs), dim=1) 

            # logits = F.softmax(logits, dim=1)  # [48, 3, 352, 352]
            # # visual
            # # name = ['Background', 'SOD', 'COD']
            # # feature_map_path = "/home/benchmark-result/AWA"
            # # for i in range(3):
            # #     feature_map = logits[0][i]*255
            # #     feature_map = feature_map.cpu().numpy()
            # #     feature_map = Image.fromarray(feature_map.astype(np.uint8))
            # #     feature_map.save(os.path.join(feature_map_path, img_name + name[i]+'.png'))
            # prediction = logits.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            # prediction = colorize_mask(prediction)
            # path = os.path.join(r"/home/benchmark-result/SISOD/SAM2-APG-46", 'USC12K-COD')
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # prediction.save(os.path.join(path, img_name + '.png'))
            # print(img_name + '.png' + " has been saved.")



            logits = F.softmax(logits, dim=1)  # [48, 3, 352, 352]
            imgPredict = logits.data.max(1)[1].cpu().numpy()
            # tensor
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
        # meanCODSODConfusionAccuracy
        # MCSCA = metric.meanCODSODConfusionAccuracy()
        CSCS = metric.CSCS()
        # MCA3 = metric.meanCODSODConfusionAccuracy3()
        # MCA4 = metric.meanCODSODConfusionAccuracy4()
        # MCA5 = metric.meanCODSODConfusionAccuracy5()

        print('hist is :\n', hist)
        print('PA is : %f' % pa)
        print('cPA is :', cpa)
        print('mPA is : %f' % mpa)
        print('IoU is : ', IoU)
        print('mIoU is : ', mIoU)
        # print('MCSCA is : ', MCSCA)
        print('MCA2(CSCS) is : ', CSCS)
        # print('MCA3 is : ', MCA3)
        # print('MCA4 is（review） : ', MCA4)
        # print('MCA5 is（allcs） : ', MCA5)

if __name__ == '__main__':
    test(network)