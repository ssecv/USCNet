"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""

import torch
import cv2
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  #

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  #

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean()  #
        return meanAcc  #


    def meanCODSODConfusionAccuracy(self):
        acc1 = self.confusionMatrix[1][2] / self.confusionMatrix.sum()
        acc2 = self.confusionMatrix[2][1] / self.confusionMatrix.sum()
        return (acc1 + acc2) / 2

    # 2025.5.16 final
    def CSCS(self):
        print(self.confusionMatrix.sum(axis=0))

        # # print每一个值 [hang][lie]
        # print(self.confusionMatrix[0][0])
        # print(self.confusionMatrix[0][1])
        # print(self.confusionMatrix[0][2])
        # print(self.confusionMatrix[1][0])
        # print(self.confusionMatrix[1][1])
        # print(self.confusionMatrix[1][2])
        # print(self.confusionMatrix[2][0])
        # print(self.confusionMatrix[2][1])
        # print(self.confusionMatrix[2][2])

        acc1 = self.confusionMatrix[1][2] / self.confusionMatrix.sum(axis=0)[2]
        acc2 = self.confusionMatrix[2][1] / self.confusionMatrix.sum(axis=0)[1]
        return (acc1 + acc2) / 2


    # def meanCODSODConfusionAccuracy3(self):
    #     print(self.confusionMatrix.sum(axis=0))
    #     if self.confusionMatrix.sum(axis=1)[1] == 0:
    #         acc1 = 0
    #     else:
    #         acc1 = self.confusionMatrix[1][2] / self.confusionMatrix.sum(axis=1)[1]
    #     if self.confusionMatrix.sum(axis=1)[2] == 0:
    #         acc2 = 0
    #     else:
    #         acc2 = self.confusionMatrix[2][1] / self.confusionMatrix.sum(axis=1)[2]
    #     return (acc1 + acc2) / 2
    #
    # def meanCODSODConfusionAccuracy4(self):
    #     if self.confusionMatrix.sum(axis=1)[1] == 0:
    #         acc1 = 0
    #     else:
    #         acc1 = self.confusionMatrix[1][2] / (self.confusionMatrix[1][2] +self.confusionMatrix[2][2] )
    #     if self.confusionMatrix.sum(axis=1)[2] == 0:
    #         acc2 = 0
    #     else:
    #         acc2 = self.confusionMatrix[2][1] / (self.confusionMatrix[2][1]+self.confusionMatrix[1][1])
    #     return (acc1 + acc2) / 2
    #
    # def meanCODSODConfusionAccuracy5(self):
    #     if self.confusionMatrix.sum(axis=1)[1] == 0:
    #         acc1 = 0
    #     else:
    #         acc1 = self.confusionMatrix[1][2] / (self.confusionMatrix[1][2] +self.confusionMatrix[2][2]+self.confusionMatrix[2][1]+self.confusionMatrix[1][1] )
    #     if self.confusionMatrix.sum(axis=1)[2] == 0:
    #         acc2 = 0
    #     else:
    #         acc2 = self.confusionMatrix[2][1] / (self.confusionMatrix[1][2] +self.confusionMatrix[2][2]+self.confusionMatrix[2][1]+self.confusionMatrix[1][1])
    #     return (acc1 + acc2) / 2

    # 打印每一列总和
    def printColSum(self):
        print(torch.sum(self.confusionMatrix, axis=0))#


    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  #
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  #
        IoU = intersection / union  #
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float('inf')].mean()  #
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  #
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))


# 测试内容
if __name__ == '__main__':
    imgPredict = torch.tensor([[0, 1, 2], [2, 1, 1]]).long()  #
    imgLabel = torch.tensor([[0, 1, 255], [1, 1, 2]]).long()  #



    ignore_labels = [255]
    metric = SegmentationMetric(3)  #

    hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)




    imgPredict = torch.tensor([[0, 1, 2], [2, 1, 1]]).long()  # 可直接换成预测图片
    imgLabel = torch.tensor([[0, 1, 2], [2, 1, 1]]).long()  # 可直接换成标注图片

    hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)

##output
# hist is :
# tensor([[1., 0., 0.],
#        [0., 2., 1.],
#        [0., 1., 0.]])
# PA is : 0.600000
# cPA is : tensor([1.0000, 0.6667, 0.0000])
# mPA is : 0.555556
# IoU is :  tensor([1.0000, 0.5000, 0.0000])
# mIoU is :  tensor(0.5000)
