import os
import argparse
import numpy as np
import pandas as pd
import importlib
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import InferenceDataset
from evaluation import compute_seg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="resnet38_seg", type=str)
    parser.add_argument("--csv_path", default="test.csv", type=str)
    parser.add_argument("--dataset_type", type=str, default='brats')
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--project_path", default="bimex-ms_saved_models", type=str)
    args = parser.parse_args()

    # Load Model
    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights))
    model.cuda().eval()

    # Natively load the Test dataset via the author's logic
    config = {'dataset': args.dataset_type, 'task': 'multiclass', 'combine': {'core': ['necrosis', 'enhancing'], 'edema': ['edema']}}
    test_df = pd.read_csv(args.csv_path)
    test_loader = DataLoader(InferenceDataset(test_df, args.img_size, config), batch_size=1, shuffle=False)
    
    metrics = {'Core Dice': [], 'Core IoU': [], 'Core HD95': [], 'Core ASSD': [],
           'Edema Dice': [], 'Edema IoU': [], 'Edema HD95': [], 'Edema ASSD': []}

    print(f"Running Supervised Inference on Test Set...")

    with torch.no_grad():
        for img_name, case_batch, seg_batch in tqdm(test_loader, desc="Evaluating"):
            img_name = img_name[0][:-4]
            
            # Forward Pass
            prob = model(x=case_batch.cuda())
            prob = F.interpolate(prob, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
            pred = np.argmax(F.softmax(prob, dim=1).cpu().data[0].numpy(), axis=0)

            # Evaluate Core (pred == 1)
            gt_core = np.where(seg_batch[0][0].numpy()!=0, 1, 0) + np.where(seg_batch[0][1].numpy()!=0, 1, 0)
            core_res = compute_seg_metrics(gt_core, (pred == 1).astype(np.uint8))
            metrics['Core Dice'].append(core_res['Dice'])
            metrics['Core IoU'].append(core_res['IoU'])
            metrics['Core HD95'].append(core_res['HD95'])
            metrics['Core ASSD'].append(core_res['ASSD'])

            # Evaluate Edema (pred == 2)
            gt_edema = np.where(seg_batch[0][2].numpy()!=0, 1, 0)
            edema_res = compute_seg_metrics(gt_edema, (pred == 2).astype(np.uint8))
            metrics['Edema Dice'].append(edema_res['Dice'])
            metrics['Edema IoU'].append(edema_res['IoU'])
            metrics['Edema HD95'].append(edema_res['HD95'])
            metrics['Edema ASSD'].append(edema_res['ASSD'])
    print("\n--- Final Supervised Segmentation Results ---")
    for k, v in metrics.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

if __name__ == '__main__':
    main()
    