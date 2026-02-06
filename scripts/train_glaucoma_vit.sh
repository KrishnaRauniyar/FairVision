#!/bin/bash

# Set the dataset and result directories
#DATASET_DIR=/ddnB/project/mshi/datasets/FairDomain/classification
DATASET_DIR=/ddnB/project/mshi/datasets/Harvard-GF
RESULT_DIR=.
MODEL_TYPE=( ViT-B ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext | ViT-B 
MODALITY_TYPE=oct_bscans # Options: 'oct_bscans_3d' | 'slo_fundus' | 'oct_bscans' | rnflt | oct_fundus
ATTRIBUTE_TYPE=( race gender hispanic ) # Options: race | gender | hispanic
DATASET=harvardgf # Options: fairvision | harvardgf | fairdomain

# Args (slo_fundus)
VIT_WEIGHTS=mae_color_fundus #Options : scratch, imagenet, mae, mocov3, mae_chest_xray, mae_color_fundus
BATCH_SIZE=16
BLR=5e-4 
WD=0.05
LD=0.55
DP=0.2
EXP_NAME=${VIT_WEIGHTS}_oct_bscans
TASK=cls

PERF_FILE=Glaucoma_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python scripts/train_glaucoma_fair.py \
        --epochs 50 \
        --seed 1\
        --batch_size ${BATCH_SIZE} \
        --blr ${BLR} \
        --min_lr 1e-6 \
        --warmup_epochs 10 \
        --weight_decay ${WD} \
        --layer_decay ${LD} \
        --drop_path ${DP} \
        --data_dir ${DATASET_DIR}/ \
        --dataset ${DATASET}\
        --result_dir ${RESULT_DIR}/results/Glaucoma1F_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE} \
        --model_type ${MODEL_TYPE} \
        --modality_types ${MODALITY_TYPE} \
        --task ${TASK} \
        --perf_file ${PERF_FILE} \
        --vit_weights ${VIT_WEIGHTS} \
        --attribute_type ${ATTRIBUTE_TYPE} 
