# USCNet [ICCV'25]


Official Implementation of ICCV 2025 paper "[Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes](https://dl.acm.org/doi/pdf/10.1145/3581783.3611811)"

[Zhangjun Zhou](https://scholar.google.com/citations?hl=zh-CN&user=lvx5k9cAAAAJ), [Yiping Li](https://scholar.google.com/citations?user=QUHsxCoAAAAJ&hl=en), [Chunlin Zhong](https://scholar.google.com/citations?user=ai328a4AAAAJ&hl=en), [Jialun Pei](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en), [Hua Li](https://scholar.google.com.sg/citations?hl=zh-CN&user=0O2iY34AAAAJ&view_op=list_works&sortby=pubdate), and [He Tang](https://scholar.google.com/citations?hl=en&user=70XLFUsAAAAJ)✉ 

[[Paper]](https://arxiv.org/pdf/2412.10943); [[Official Version]]()

**Contact:** hetang@hust.edu.cn

## Environment preparation

### Requirements
- **Please refer to the link: [SAM2](https://github.com/facebookresearch/sam2)** 

## Dataset preparation :fire:

### Download the datasets and annotation files

- **[USC12K](https://pan.baidu.com/s/1JkJlNh0A4NI4_0elMmo5ug?pwd=9999)**/ **[Google](https://drive.google.com/file/d/1MIVCH7sLOzFwrzEDjKs2PSba7UpzLG7I/view?usp=sharing).**


### Register datasets

1. Download the datasets and put them in the same folder. To match the folder name in the dataset mappers, you'd better not change the folder names, its structure may be:

```
    DATASET_ROOT/
    ├── VOC-USC12K
       ├── ImageSets
          ├── Segmentation
              ├── Scene-A.txt
              ├── Scene-B.txt
              ├── Scene-C.txt
              ├── Scene-D.txt
              ├── train.txt
              ├── val.txt
       ├── JPEGImages
       ├── SegmentationClass

```

## Pre-trained models :
- Download the pre-training weights : **[sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt).**
- Download the pre-trained weights on USC12K: **[Baidu](https://pan.baidu.com/s/1voZOkIOZGnSUjMakzG0Zjw?pwd=9999)**/ **[Google](https://drive.google.com/file/d/1_F0icyu5Tcc7ojo4eAsaTb-SkinDCNPv/view?usp=sharing).**

## Visualization results &#x26A1;


The visual results of  **SOTAs** on **USC12K test set**.
- Results on the Overall Scene test set: **[Baidu](https://pan.baidu.com/s/1f2W0x7LbR0Ueu3CufT3BLQ?pwd=9999)**/ **[Google](https://drive.google.com/file/d/1S-tz1u5eK7Ehy1gejyoC1ulEZnLiclWX/view?usp=sharing).**

## Usage

### Train&Test
- To train our USCNet on single GPU by following command,the trained models will be saved in savePath folder. You can modify datapath if you want to run your own datases.
```shell
bash train.sh
```
- To test and evaluate our USCNet on USC12K:
```shell
bash test.sh
```


## Acknowledgement

[//]: # (This work is based on:)

[//]: # (- [SAM]&#40;https://github.com/facebookresearch/segment-anything&#41;)

[//]: # ()
[//]: # (Thanks for their great work!)

## Citation

If this helps you, please cite this work:

```
```

