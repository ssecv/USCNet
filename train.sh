CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--lr 1e-4 \
--batch_size 22 \
--epoch 48 \
--savepath "/home/benchmark-code/SAM2-USC12K-Hirea-l-adapter-APG-twoway/save/weight-SAM2-APG-twoway-repeat" \
--AdamW \


#CUDA_VISIBLE_DEVICES=0 \
#python3 train.py \
#--lr 1e-4 \
#--batch_size 22 \
#--epoch 48 \
#--savepath "/home/benchmark-code/SAM2-USC12K-Hirea-l-adapter-APG-twoway/save/weight-SAM2-APG-twoway-repeat" \
#--AdamW \

#
#the traing unstable to try more lr
#--lr 3e-4 \
#--batch_size 32 \
#--epoch 100 \
#--savepath "/home/VOC-USC12K/weight-SAM2-hirea-l-Adapter-APG-3e-4-epoch100-weightinitialkaiming" \
#--AdamW \

#bs=256
#bs=128
#sam2bs

#--lr 1e-4 \
#--batch_size 24 \
#--epoch 90 \gogddddss
#--savepath "/home/VOC-USC12K/weight-SAM2-hirea-l-Adapter-APG-weightinitialkaiming" \
#--AdamW \

#--lr 1e-4 \
#--batch_size 22 \
#--epoch 48 \
#--savepath "/home/VOC-USC12K/weight-SAM2-hirea-l-Adapter-APG" \
#--AdamW \

#--lr 3e-4 \
#--batch_size 22 \
#--epoch 100 \
#--savepath "/home/VOC-USC12K/weight-SAM2-hirea-l-Adapter-APG-3e-4-epoch100" \
#--AdamW \