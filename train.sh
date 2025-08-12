#CUDA_VISIBLE_DEVICES=0 \
#python3 train.py \
#--lr 1e-4 \
#--batch_size 8 \
#--epoch 48 \
#--savepath "/home/benchmark-code/SAM-USC12K-ViT-h-baseline+adapter-ARM-twoway/save2/weight-vit-h-adapter-ARM" \
#--AdamW \

#L40
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--lr 1e-4 \
--batch_size 24 \
--epoch 48 \
--savepath "/home/um20/SAM/SAM-USC12K-ViT-h-adapter-ARM-twoway/save/weight-vit-h-adapter-ARM" \
--AdamW \

##L40
#CUDA_VISIBLE_DEVICES=0 \
#python3 train.py \
#--lr 1e-4 \
#--batch_size 24 \
#--epoch 90 \
#--savepath "/home/um20/SAM/SAM-USC12K-ViT-h-adapter-ARM-twoway/save/weight-vit-h-adapter-ARM" \
#--AdamW \

#L40
#--lr 1e-4 \
#--batch_size 24 \  
#--epoch 48 \
# --savepath "/save/weight-vit-h-adapter-ARM" \
#--AdamW \

#L40
#--lr 3e-4 \
#--batch_size 32 
#--epoch 48 \
# --savepath "/save/weight-vit-h-adapter-ARM" \
#--AdamW \

#L40
#--lr 1e-4 \
#--batch_size 22 \
#--epoch 48 \
# --savepath "/y/save/weight-vit-h-adapter-ARM" \
#--AdamW \


#L40
#--lr 1e-4 \
#--batch_size 8 \
#--epoch 48 \
# --savepath "/y/save/weight-vit-h-adapter-ARM" \
#--AdamW \

