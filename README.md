# BiMEx-MS: Binary-Guided Mutually Exclusive Multiclass Segmentation

**Mutually Exclusive Multiclass Lesion Segmentation in Neuroimaging: Binary-Guided Weak Supervision with Inter-Class Orthogonality**

BiMEx-MS is a weakly supervised multiclass segmentation framework for neuroimaging that addresses a core failure mode of existing methods: the absence of explicit spatial mutual exclusivity between co-occurring, spatially adjacent lesion subregions (e.g., tumor core and edema, or overlapping hemorrhage subtypes). BiMEx-MS decomposes multiclass segmentation into two disentangled stages — (i) a binary lesion localization module that provides a class-frequency-agnostic structural prior confining predictions to the detected lesion domain, and (ii) a class-specific multi-exit CAM aggregation network trained under a tri-partite loss (per-class separation, inter-class orthogonality, and binary–multiclass agreement) — followed by hierarchical morphological pseudo-label refinement and a final segmentation network trained on the refined pseudo-labels.

Using only image-level classification labels for training, BiMEx-MS is evaluated on brain tumor MRI (BraTS 2020, BraTS 2023 SSA) and intracranial hemorrhage CT (RSNA-ICH → BHSD), consistently outperforming sixteen weakly supervised state-of-the-art baselines, with the largest gains on boundary metrics (HD95, ASSD) and rare, long-tailed subtypes.

This repository contains the official implementation of BiMEx-MS, covering the full pipeline from raw BraTS NIfTI volumes to final pseudo-label-supervised segmentation.

---

## Installation

### Setting Up the Environment

**Prerequisites:** [Anaconda](https://www.anaconda.com/) / Miniconda, Python 3.10

1. Clone the repository:
```bash
git clone https://github.com/ashutoshkr45/BiMEx-MS-Neuro.git
cd BiMEx-MS-Neuro
```

2. Create and activate the conda environment:
```bash
conda create --name bimex_env python==3.10 -y
conda activate bimex_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Your environment is now ready to use BiMEx-MS!

---

## Dataset

We use the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset for 2D slice-level tumor classification and segmentation. Image-level labels are derived from the voxel-level annotations and used solely for training; voxel-level masks are reserved for evaluation.

### Preprocessing

Convert the BraTS NIfTI scans into 2D PNG slices:

```bash
python generate_dataset.py \
  --input_dir <path_to_BraTS_NIfTI_scans> \
  --output_dir <path_to_output_dir>
```

This also generates `train.csv`, `val.csv`, and `test.csv` for the downstream training scripts.

### Folder Structure

After running `generate_dataset.py`, the following structure is created inside `--output_dir`:

```
<output_dir>/
├── <subject_id_1>/
│   ├── flair/
│   │   ├── <subject_id_1>_flair_0.png
│   │   ├── <subject_id_1>_flair_1.png
│   │   └── ...
│   ├── t1/
│   │   └── <subject_id_1>_t1_*.png
│   ├── t1ce/
│   │   └── <subject_id_1>_t1ce_*.png
│   ├── t2/
│   │   └── <subject_id_1>_t2_*.png
│   └── seg/
│       └── <subject_id_1>_seg_*.png
├── <subject_id_2>/
│   └── ...
└── ...
```

---

## Pipeline Overview

BiMEx-MS is trained in five sequential stages, matching Algorithm 1 and Fig. 1 of the paper:

1. **Contrastive pretraining** of the binary and multiclass multi-exit classifiers (`pretrain_clnet.py`)
2. **Fine-tuning** the multi-exit classifiers under the classification and tri-partite loss objectives (`train_cnet.py`)
3. **Training the class-specific CAM aggregation network** with frozen classifier backbones (`train_aggnet.py`)
4. **Generating hierarchically refined pseudo-labels** from confidence-gated, binary-guided CAMs (`gen_pseudo_labels.py`)
5. **Training the final Wide-ResNet-38 segmentation network** on the refined pseudo-labels, and evaluating it (`train_seg.py`, `infer_seg.py`)

The full pipeline can be run end-to-end via `run.sh`, or stage-by-stage as described below.

---

## Usage

### Running the Full Pipeline

```bash
bash run.sh
```

### Running Stage-by-Stage

**1. Generate the dataset**
```bash
python generate_dataset.py \
  --input_dir BraTS2020/MICCAI_BraTS2020_TrainingData \
  --output_dir ../TrainingData_2d_images
```

**2. Pretrain the binary and multiclass classifiers (contrastive pretraining)**
```bash
python pretrain_clnet.py \
  --project_path "bimex-ms_saved_models" \
  --record_path "pretrain_record" \
  --modality "flair_t1ce_t2" \
  --binary_epochs 100 \
  --multiclass_epochs 50 \
  --batch_size 155 \
  --learning_rate 1e-3 \
  --img_size 224 \
  --gpu_ids 0 \
  --dataset_type "brats"
```

**3. Fine-tune the multi-exit classification network (CNet)**
```bash
python train_cnet.py \
  --project_path "bimex-ms_saved_models" \
  --record_path "train_record" \
  --modality "flair_t1ce_t2" \
  --binary_epochs 50 \
  --multiclass_epochs 50 \
  --batch_size 155 \
  --learning_rate 5e-4 \
  --img_size 224 \
  --gpu_ids 0 \
  --dataset_type "brats"
```

**4. Train the class-specific CAM aggregation network (AggNet)**
```bash
python train_aggnet.py \
  --project_path "bimex-ms_saved_models" \
  --record_path "agg_train_record" \
  --modality "flair_t1ce_t2" \
  --epochs 50 \
  --batch_size 156 \
  --learning_rate 1e-3 \
  --img_size 224 \
  --gpu_ids 0 1 \
  --dataset_type "brats"
```

**5. Generate binary-guided, morphologically refined pseudo-labels**
```bash
python gen_pseudo_labels.py --project_path "bimex-ms_saved_models"
```

**6. Train the final segmentation network on refined pseudo-labels**
```bash
python train_seg.py \
  --network resnet38_seg \
  --init_weights res38_cls.pth \
  --num_epochs 30 \
  --batch_size 16 \
  --lr 0.002
```

**7. Evaluate final supervised segmentation performance**
```bash
python infer_seg.py \
  --project_path "bimex-ms_saved_models" \
  --weights seg_weights/brats_model_29.pth
```

---

## Requirements

```
torch==2.4.0
torchvision==0.19.0
tqdm==4.64.1
numpy==2.2.6
scikit-learn==1.5.0
Pillow==9.4.0
scikit-image==0.24.0
matplotlib==3.9.2
MedPy==0.5.2
pandas==2.3.2
nibabel==5.2.1
pyarrow==21.0.0
albumentations==2.0.8
```

---
