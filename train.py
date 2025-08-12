#!/usr/bin/python3
# coding=utf-8
import datetime
import argparse
import sys
from torch import nn

sys.path.insert(0, '/')
sys.dont_write_bytecode = True
from dataloaders import make_data_loader
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from apex import amp
import torch.optim as optim

from net_sam_USCNet import SAM_USCNet as network
from PIL import Image

# from net_sam_baseline import SAM_baseline as network

def focal_loss(logits, targets, class_weights=[1, 4, 6], gamma=2.0):
    """
    Computes the focal loss for multi-class classification.
    Args:
        logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W).
        targets (torch.Tensor): Ground truth labels of shape (batch_size, H, W) with values in {0, 1, 2}.
        class_weights (list): Weighting for each class.
        gamma (float): Focusing parameter to reduce the contribution of easy examples.
    Returns:
        torch.Tensor: The computed focal loss.
    """
    # Ensure targets are long type
    targets = targets.long()
    # Convert class weights to a tensor
    class_weights = torch.tensor(class_weights, device=logits.device, dtype=torch.float32)
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes, H, W)
    # Gather the probabilities corresponding to the target classes
    probs_for_targets = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Shape: (batch_size, H, W)
    # Compute the focal weight
    focal_weight = (1 - probs_for_targets) ** gamma
    # Apply the class weights
    weights_for_targets = class_weights[targets]  # Shape: (batch_size, H, W)
    # Compute the log probabilities
    log_probs = -torch.log(probs_for_targets + 1e-6)
    # Combine focal weight, class weight, and log probabilities
    loss = focal_weight * weights_for_targets * log_probs
    # Return the mean loss
    return loss.mean()



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--savepath', default="/home/desktop/new-cod-sod-dataset/train/weight", type=str)
    # parser.add_argument('--datapath', default="/home/desktop/new-cod-sod-dataset/train", type=str)

    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')

    # parser.add_argument('--batch_size', type=int,
    #                     default=12, help='batch_size per gpu')
    # parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    # parser.add_argument('--deterministic', type=int, default=1,
    #                     help='whether use deterministic training')
    # parser.add_argument('--img_size', type=int,
    #                     default=352, help='input patch size of network input')
    # parser.add_argument('--seed', type=int,
    #                     default=1234, help='random seed')
    # parser.add_argument('--vit_name', type=str,
    #                     default='vit_b', help='select one vit model')

    parser.add_argument('--ckpt', type=str, default='/mnt/data3/chai/SAM/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--epoch', type=int, default=48)
    parser.add_argument('--mode', type=str, default='train',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='dataset name (default: pascal)')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--base-size', type=int, default=352,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=352,
                        help='crop image size')



    parser.parse_args()
    return parser.parse_args()


def train(Network):
    ## dataset
    args = parser()
    cfg = args
    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    print("学习率", cfg.lr)
    print("batch_size", cfg.batch_size)

    ## network
    net = Network(cfg)

    net.train(True)
    net.cuda()
    ## parameter

    frozon_decoder, tuning_slide = [], []
    for name, param in net.named_parameters():
        # # print(name)
        # if 'mask_decoder' in name:  # decoder
        #     frozon_decoder.append(param)
        # else:
        tuning_slide.append(param)  #

    if args.AdamW:
        optimizer = optim.AdamW([{'params': frozon_decoder}, {'params': tuning_slide}], lr=cfg.lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr, momentum=0.9,
                              weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False`

        # layers will not update

    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    criterion = nn.CrossEntropyLoss()

    print("start...")
    for epoch in range(cfg.epoch):

        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr


        for step, sample in enumerate(train_loader):
            image, target = sample['image'], sample['label']
            image, target=image.cuda(), target.cuda()


            if target.max() > 2 or target.min() < 0:
                continue

            coarse_map, Background_outputs, sod_outputs,cod_outputs= net(image)
            Background_outputs = Background_outputs['masks']
            sod_outputs = sod_outputs['masks']
            cod_outputs = cod_outputs['masks']
            logits = torch.cat((Background_outputs, sod_outputs, cod_outputs), dim=1)  # 16 3 256 256


            # criterion = nn.CrossEntropyLoss()
            coarse_map = torch.nn.functional.interpolate(coarse_map, size=(target.shape[1], target.shape[2]),mode='nearest')  #
            # coarse_map
            loss_CE_coarse = criterion(coarse_map, target.long())  # logit: (N, C, H, W), target: (N, H, W)
            n, c, h, w = coarse_map.size()
            loss_CE_coarse = loss_CE_coarse / n

            loss_CE = criterion(logits, target.long())  # logit: (N, C, H, W), target: (N, H, W)
            n, c, h, w = logits.size()
            loss_CE = loss_CE / n

            loss = (loss_CE + loss_CE_coarse) / 2

            # # focal loss
            # loss_CE = focal_loss(logits, target, class_weights=[1, 4, 6], gamma=2.0)
            # # coarse_map
            # loss_CE_coarse = focal_loss(coarse_map, target, class_weights=[1, 4, 6], gamma=2.0)
            # loss = (loss_CE + loss_CE_coarse) / 2

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr0', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('lr1', optimizer.param_groups[1]['lr'], global_step=global_step)
            sw.add_scalars('loss_CE', {'loss_CE': loss_CE.item()}, global_step=global_step)

            if step % 80 == 0:
                print('%s | step:%d/%d/%d | loss=%.6f|lossCE=%.6f|' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, loss.item(), loss_CE.item()))


        if epoch == 0:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
        if epoch > (cfg.epoch - 5):
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
        # if epoch % 2 == 0:
        #     torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))

if __name__ == '__main__':
    train(network)
