#!/bin/bash

echo "Generating dataset..."
python generate_dataset.py --input_dir BraTS2020/MICCAI_BraTS2020_TrainingData --output_dir ../TrainingData_2d_images

echo "Pretraining Classifiers..."
python pretrain_clnet.py --project_path "bimex-ms_saved_models" --record_path "pretrain_record" --modality "flair_t1ce_t2" --binary_epochs 100 --multiclass_epochs 50 --batch_size 155 --learning_rate 1e-3 --img_size 224 --gpu_ids 0 --dataset_type "brats"

echo "Training CNet..."
python train_cnet.py --project_path "bimex-ms_saved_models" --record_path "train_record" --modality "flair_t1ce_t2" --binary_epochs 50 --multiclass_epochs 50 --batch_size 155 --learning_rate 5e-4 --img_size 224 --gpu_ids 0 --dataset_type "brats"

echo "Training AggNet..."
python train_aggnet.py --project_path "bimex-ms_saved_models" --record_path "agg_train_record" --modality "flair_t1ce_t2" --epochs 50 --batch_size 156 --learning_rate 1e-3 --img_size 224 --gpu_ids 0 1 --dataset_type "brats"

echo "Generating pure CAM-based hard pseudo-labels..."
python gen_pseudo_labels.py --project_path "bimex-ms_saved_models"

echo "Training ResNet38 Segmentation Model on refined pseudo-labels..."
python train_seg.py \
    --network resnet38_seg \
    --init_weights res38_cls.pth \
    --num_epochs 30 \
    --batch_size 16 \
    --lr 0.002

echo "Evaluating Final Supervised Segmentation Metrics..."
# Tests the model from the final epoch
python infer_seg.py \
    --project_path "bimex-ms_saved_models" \
    --weights seg_weights/brats_model_29.pth

echo "Pipeline execution completed."
