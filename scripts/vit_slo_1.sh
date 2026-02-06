#!/bin/bash

# Set the dataset and result directories
#DATASET_DIR=/ddnB/project/mshi/datasets/FairDomain/classification
DATASET_DIR=/project/mshi/datasets/FairVision/Glaucoma/
#DATASET_DIR=/ddnB/project/mshi/datasets/Harvard-GF
RESULT_DIR=.
MODEL_TYPE=( ViT-B ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans' | 'slo_fundus'
ATTRIBUTE_TYPE=( race gender hispanic ) # Options: race | gender | hispanic
DATASET=fairvision # Options: fairvision | harvardgf | fairdomain

VIT_WEIGHTS=imagenet
BATCH_SIZE=64
BLR=5e-4
WD=0.01
LD=0.55
DP=0.1

PERF_FILE=Glaucoma_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python scripts/train_glaucoma_fair.py \
        --epochs 50 \
        --seed 1\
        --batch_size ${BATCH_SIZE} \
        --blr ${BLR} \
        --min_lr 1e-6 \
        --warmup_epochs 5 \
        --weight_decay ${WD} \
        --layer_decay ${LD} \
        --drop_path ${DP} \
        --data_dir ${DATASET_DIR}/ \
        --dataset ${DATASET}\
        --result_dir ${RESULT_DIR}/results/Glaucoma1sl_${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE} \
        --model_type ${MODEL_TYPE} \
        --modality_types ${MODALITY_TYPE} \
        --perf_file ${PERF_FILE} \
        --vit_weights ${VIT_WEIGHTS} \
        --attribute_type ${ATTRIBUTE_TYPE} 
