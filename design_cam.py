import torch
import os
from datetime import datetime
import numpy as np
from models import *
from tqdm import tqdm
import torch
from evaluation import *
from utils import *

class Design_CAM(object):

    def __init__(self, args):
        self.project_path = args.project_path
        self.record_path  = args.record_path
        self.task         = args.task

        multi_model       = Res18_Classifier(num_classes=args.num_classes)
        binary_model      = Res18_Classifier(num_classes=1)
        bin_score_model   = Res_Scoring().cuda()
        multi_score_model = Res_Scoring().cuda()

        binary_model.load_pretrain_weight(args.bin_pretrained_path)
        multi_model.load_pretrain_weight(args.multi_pretrained_path)
        bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_path)
        multi_score_model.load_pretrain_weight(args.multi_score_model_pretrained_path)

        for param in binary_model.parameters():   param.requires_grad = False
        for param in multi_model.parameters():    param.requires_grad = False
        for param in bin_score_model.parameters(): param.requires_grad = False
        for param in multi_score_model.parameters(): param.requires_grad = False

        if len(args.gpu_ids) > 1:
            binary_model      = torch.nn.DataParallel(binary_model,      device_ids=args.gpu_ids)
            multi_model       = torch.nn.DataParallel(multi_model,       device_ids=args.gpu_ids)
            bin_score_model   = torch.nn.DataParallel(bin_score_model,   device_ids=args.gpu_ids)
            multi_score_model = torch.nn.DataParallel(multi_score_model, device_ids=args.gpu_ids)

        self.binary_model      = binary_model.to('cuda').eval()
        self.multi_model       = multi_model.to('cuda').eval()
        self.bin_score_model   = bin_score_model.to('cuda').eval()
        self.multi_score_model = multi_score_model.to('cuda').eval()

        self.save_dir = os.path.join(self.project_path, self.record_path, f"{self.task}_eval")
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, img):
        img = img.cuda()

        logits_collect_binary, map_collect_binary = self.binary_model(img)
        logits_collect_multi,  map_collect_multi  = self.multi_model(img)

        map_collect_binary_copy = [t.clone() for t in map_collect_binary]
        _, _, _, bin_ame_map = self.bin_score_model(img, map_collect_binary_copy)
        bin_ame_map = self.normalize_map(bin_ame_map)

        map_collect_multi = torch.stack(map_collect_multi, dim=0)
        map_collect       = bin_ame_map * map_collect_multi
        map_collect       = list(map_collect.unbind(0))

        _, _, _, ame_map = self.multi_score_model(img, map_collect)
        ame_map = torch.cat((ame_map, bin_ame_map), dim=1)

        return (ame_map.detach().cpu(),
                logits_collect_binary[-1].detach().cpu(),
                logits_collect_multi[-1].detach().cpu())

    def normalize_map(self, att_map: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        n, c, h, w = att_map.size()
        flat    = att_map.view(n, c, -1)
        min_val = flat.min(2, keepdim=True)[0]
        max_val = flat.max(2, keepdim=True)[0]
        normalized = (flat - min_val) / (max_val - min_val + eps)
        return normalized.view(n, c, h, w)

    def postprocess_cam(self, cam, binary_cam, thresholds=None):
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        binary_cam = np.maximum(binary_cam, 0)
        binary_cam = binary_cam / binary_cam.max()
        cam = np.where(binary_cam > thresholds, cam * binary_cam, 0)
        cam = np.where(cam > 0.4, 1, 0)
        return cam

    def run_tumor_test(self, loader, threshold=None):
        self.binary_model.eval()
        self.multi_model.eval()
        self.bin_score_model.eval()
        self.multi_score_model.eval()

        log_path = os.path.join(self.save_dir, "results.log")
        csv_path = os.path.join(self.save_dir, "tumor_result.csv")

        # Folder to save per-slice predicted masks
        pred_save_dir = os.path.join(self.save_dir, "pred_masks")
        os.makedirs(pred_save_dir, exist_ok=True)

        with open(log_path, "w+") as f:
            f.write(str(datetime.now()) + "\n")

        with open(csv_path, "w+") as f:
            f.write("Img Name,Core Dice,Core IoU,Core HD95,Core ASSD,"
                    "Edema Dice,Edema IoU,Edema HD95,Edema ASSD\n")

        result_metric = {
            'Core Dice':  [], 'Core IoU':  [], 'Core HD95':  [], 'Core ASSD':  [],
            'Edema Dice': [], 'Edema IoU': [], 'Edema HD95': [], 'Edema ASSD': []
        }

        test_bar = tqdm(loader)

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]

                ame_map, binary_logit, class_logit = self.step(case_batch)

                binary_logit = binary_logit.squeeze(0).cpu().numpy()
                class_logit  = class_logit.squeeze(0).cpu().numpy()
                logit        = np.concatenate((class_logit, binary_logit), axis=0)

                input_image = case_batch[0].permute(1, 2, 0)
                ame_map     = self.CAM_algo(input_image, ame_map, img_name)

                results    = {}
                pred_masks = {}   # ← accumulate predictions for saving

                for i, class_name in enumerate(['core', 'edema']):
                    if class_name == 'core':
                        gt = (np.where(seg_batch[0][0].numpy() != 0, 1, 0) +
                            np.where(seg_batch[0][1].numpy() != 0, 1, 0))
                    elif class_name == 'edema':
                        gt = np.where(seg_batch[0][2].numpy() != 0, 1, 0)
                    else:
                        raise ValueError(f"Unknown class name: {class_name}")
                    gt = np.clip(gt, 0, 1)

                    final_seg = self.postprocess_cam(ame_map[i], ame_map[-1], threshold)

                    if logit[i] < 0.5:
                        final_seg = np.zeros_like(gt)

                    pred_masks[class_name] = final_seg.astype(np.uint8)  # ← store
                    results[class_name]    = compute_seg_metrics(gt, final_seg)

                # Save predicted masks for this slice
                np.save(os.path.join(pred_save_dir, f"{img_name}.npy"), pred_masks)

                # CSV write
                with open(csv_path, "a") as f:
                    f.write(
                        f"{img_name},"
                        f"{results['core']['Dice']:.3f},{results['core']['IoU']:.3f},"
                        f"{results['core']['HD95']:.3f},{results['core']['ASSD']:.3f},"
                        f"{results['edema']['Dice']:.3f},{results['edema']['IoU']:.3f},"
                        f"{results['edema']['HD95']:.3f},{results['edema']['ASSD']:.3f}\n"
                    )

                # Metric accumulation
                for k, v in results.items():
                    result_metric[f"{k.capitalize()} Dice"].append(v['Dice'])
                    result_metric[f"{k.capitalize()} IoU"].append(v['IoU'])
                    result_metric[f"{k.capitalize()} HD95"].append(v['HD95'])
                    result_metric[f"{k.capitalize()} ASSD"].append(v['ASSD'])

        test_bar.close()

        avg_metrics = {k: np.nanmean(v) for k, v in result_metric.items()}
        std_metrics = {k: np.nanstd(v)  for k, v in result_metric.items()}

        with open(log_path, "a") as f:
            f.write("\n=== Final Average Results ===\n")
            print("\n=== Final Average Results ===")
            for k in avg_metrics:
                line = f"{k}: {avg_metrics[k]:.3f} +- {std_metrics[k]:.3f}"
                f.write(line + "\n")
                print(line)

        print(f"\nPredicted masks saved to: {pred_save_dir}")
        return avg_metrics

    def CAM_algo(self, input_image, ame_map, img_name, output_hist=False):
        for i in range(ame_map.shape[1]):
            if (ame_map[0][i].max() - ame_map[0][i].min()) > 0:
                ame_map[0][i] = ((ame_map[0][i] - ame_map[0][i].min()) /
                                 (ame_map[0][i].max() - ame_map[0][i].min() + 1e-5))
        ame_map = ame_map.squeeze(0).numpy()
        ame_map = (1 - ame_map)
        return ame_map
    