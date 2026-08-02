import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import cv2
from scipy import ndimage
from torch.utils.data import DataLoader

from dataset import InferenceDataset
from design_cam import Design_CAM
from evaluation import compute_seg_metrics

# BraTS Palette: 0=BG(Black), 1=Core(Red), 2=Edema(Blue)
palette = [0, 0, 0, 255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 9)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_path', default='multi_cam_project')
    parser.add_argument('--record_path', default='Thresh_Pseudo_Eval')
    parser.add_argument('--task', default='multiclass')
    parser.add_argument('--num_classes', type=int, default=2)   # Core & Edema
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--dataset_type', type=str, default='brats')
    parser.add_argument('--out_dir', default='pseudo_labels_brats')
    return parser.parse_args()

# def seed_and_expand(soft_cam, high_t, low_t):
#     """Hysteresis Thresholding: Expands confident seeds into faint connected tails."""
#     seed_mask = (soft_cam > high_t).astype(np.uint8)
#     loose_mask = (soft_cam > low_t).astype(np.uint8)

#     labeled_loose, num_features = ndimage.label(loose_mask)
#     final_mask = np.zeros_like(loose_mask)
    
#     for i in range(1, num_features + 1):
#         component = (labeled_loose == i)
#         if np.any(component & seed_mask):
#             final_mask[component] = 1
            
#     return final_mask

def refine_mask(mask, close_kernel=5, dilate_kernel=0):
    """Fills internal holes, smooths boundaries, and optionally inflates."""
  
    filled_mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    
    if close_kernel > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        smoothed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, k_close)
    else:
        smoothed_mask = filled_mask
        
    if dilate_kernel > 0:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        smoothed_mask = cv2.dilate(smoothed_mask, k_dilate, iterations=1)
        
    return smoothed_mask

def generate_pseudo_labels_and_eval(des_cam, loader, bin_threshold, save_dir=None):
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    
    metrics = {'Core Dice': [], 'Core IoU': [], 'Core HD95': [], 
               'Edema Dice': [], 'Edema IoU': [], 'Edema HD95': []}
               
    with torch.no_grad():
        for img_name, case_batch, seg_batch in tqdm(loader, desc=f"Processing"):
            img_name = img_name[0][:-4]
            ame_map, binary_logit, class_logit = des_cam.step(case_batch)
            
            binary_logit = binary_logit.squeeze(0).cpu().numpy()
            class_logit = class_logit.squeeze(0).cpu().numpy()
            logit = np.concatenate((class_logit, binary_logit), axis=0)
            
            input_image = case_batch[0].permute(1, 2, 0)
            ame_map = des_cam.CAM_algo(input_image, ame_map, img_name)
            
            final_mask = np.zeros((224, 224), dtype=np.uint8)
            
            # Temporary storage for the individual masks
            class_preds = {}
            
            for i, class_name in enumerate(['core', 'edema']):
                final_seg = des_cam.postprocess_cam(ame_map[i], ame_map[-1], bin_threshold)
                
                # Image-level gate
                if logit[i] < 0.5: 
                    final_seg = np.zeros_like(final_seg)
                    
                class_preds[class_name] = final_seg
                
                if class_name == 'core':
                    gt = np.where(seg_batch[0][0].numpy()!=0, 1, 0) + np.where(seg_batch[0][1].numpy()!=0, 1, 0)
                else:
                    gt = np.where(seg_batch[0][2].numpy()!=0, 1, 0)
                    
                res = compute_seg_metrics(gt, final_seg)
                metrics[f'{class_name.capitalize()} Dice'].append(res['Dice'])
                metrics[f'{class_name.capitalize()} IoU'].append(res['IoU'])
                metrics[f'{class_name.capitalize()} HD95'].append(res['HD95'])
                
            raw_core = class_preds['core'].astype(np.uint8)
            raw_edema = class_preds['edema'].astype(np.uint8)
            refined_core = refine_mask(raw_core, close_kernel=5, dilate_kernel=0)
            
            raw_wt = np.logical_or(raw_core, raw_edema).astype(np.uint8)
            refined_wt = refine_mask(raw_wt, close_kernel=10, dilate_kernel=0)

            final_mask[refined_wt == 1] = 2
            final_mask[refined_core == 1] = 1
            
            if save_dir:
                out_img = Image.fromarray(final_mask, mode='P')
                out_img.putpalette(palette)
                out_img.save(os.path.join(save_dir, img_name + '.png'))
                
    return {k: np.mean(v) for k, v in metrics.items()}


if __name__ == '__main__':
    args = parse_args()
    
    config = {'dataset': args.dataset_type, 'task': 'multiclass', 'combine': {'core': ['necrosis', 'enhancing'], 'edema': ['edema']}}
    
    args.bin_pretrained_path = os.path.join(args.project_path, 'train_record', 'binary_clstrain', 'binary_classifier.pth')
    args.multi_pretrained_path = os.path.join(args.project_path, 'train_record', 'multiclass_clstrain', 'multi_classifier.pth')
    args.bin_score_model_pretrained_path = os.path.join(args.project_path, 'agg_train_record', 'binary_aggtrain', 'score_model.pth')
    args.multi_score_model_pretrained_path = os.path.join(args.project_path, 'agg_train_record', 'multiclass_aggtrain', 'score_model.pth')
    
    train_loader = DataLoader(InferenceDataset(pd.read_csv('train.csv'), args.img_size, config), batch_size=1, shuffle=False)
    test_loader = DataLoader(InferenceDataset(pd.read_csv('test.csv'), args.img_size, config), batch_size=1, shuffle=False)
    
    des_cam = Design_CAM(args)
    best_thr = 0.50 # Binary Threshold
    
    print(f"\n--- Generating Train Pseudo-Labels (Bin_Thr={best_thr:.2f}) ---")
    generate_pseudo_labels_and_eval(des_cam, train_loader, best_thr, save_dir=os.path.join(args.out_dir, 'train'))
    
    print(f"\n--- Evaluating Test Set Performance (Bin_Thr={best_thr:.2f}) ---")
    test_res = generate_pseudo_labels_and_eval(des_cam, test_loader, best_thr, save_dir=None)
    
    print("\n--- Final Strategy Test Results ---")
    for k, v in test_res.items(): print(f"{k}: {v:.3f}")
    