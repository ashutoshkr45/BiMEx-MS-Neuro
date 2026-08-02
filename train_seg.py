import os
import argparse
import numpy as np
import pandas as pd
import importlib
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tool import pyutils, torchutils
from dataset import InferenceDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        true_1hot = torch.eye(num_classes, device=logits.device)[targets]
        true_1hot = true_1hot.permute(0, 3, 1, 2).float()
        
        probs = F.softmax(logits, dim=1)
        dice_loss = 0.0
        
        for class_idx in range(1, num_classes):
            prob_c = probs[:, class_idx]
            true_c = true_1hot[:, class_idx]
            intersection = (prob_c * true_c).sum((1, 2))
            cardinality = prob_c.sum((1, 2)) + true_c.sum((1, 2))
            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss += (1 - dice.mean())
            
        return dice_loss / (num_classes - 1)

class BraTSPseudoDatasetWrapper(Dataset):
    def __init__(self, df, img_size, config, pseudo_mask_dir):
        self.inf_dataset = InferenceDataset(df, img_size, config)
        self.mask_dir = pseudo_mask_dir

    def __len__(self):
        return len(self.inf_dataset)

    def __getitem__(self, idx):
        # Natively load image tensor from the project's own dataset
        img_name, case_batch, _ = self.inf_dataset[idx]
        
        # Load our generated pseudo-label
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = np.array(Image.open(mask_path))
        mask = torch.from_numpy(mask).long()

        return case_batch, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="train.csv", type=str)
    parser.add_argument("--seg_pgt_path", default="pseudo_labels_brats/train", type=str)
    parser.add_argument("--save_path", default="seg_weights", type=str)
    
    parser.add_argument("--dataset_type", type=str, default='brats')
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_classes", default=3, type=int) # BG, Core, Edema
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--network", default='resnet38_seg', type=str)
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--init_weights", default='res38_cls.pth', type=str)
    parser.add_argument("--session_name", default="brats_model_", type=str)
    parser.add_argument('--print_intervals', type=int, default=50)
    args = parser.parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    pyutils.Logger(os.path.join(args.save_path, args.session_name + '.log'))

    # Load Model & Initialize Weights
    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)
    
    if os.path.exists(args.init_weights):
        print(f"Loading pre-trained weights from {args.init_weights}...")
        weights_dict = torch.load(args.init_weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print(f"WARNING: {args.init_weights} not found. Training from scratch!")
        
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    config = {'dataset': args.dataset_type, 'task': 'multiclass', 'combine': {'core': ['necrosis', 'enhancing'], 'edema': ['edema']}}
    train_df = pd.read_csv(args.csv_path)
    train_dataset = BraTSPseudoDatasetWrapper(train_df, args.img_size, config, args.seg_pgt_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    max_step = args.num_epochs * len(train_loader)

    # Optimizer & Losses
    optimizer = torchutils.PolyOptimizer_cls([
        {'params': model.module.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.module.get_10x_lr_params(), 'lr': 10 * args.lr}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    dice_criterion = DiceLoss().cuda()
    timer = pyutils.Timer("Session started: ")
    
    epoch_losses = []

    print(f"Starting Segmentation Training for {args.num_epochs} Epochs...")
    for ep in range(args.num_epochs):
        ep_loss_sum = 0.0
        
        for iter, (images, seg_labels) in enumerate(train_loader):
            images = images.cuda()
            seg_labels = seg_labels.cuda()

            pred = model(x=images)
            pred = F.interpolate(pred, size=(images.size(2), images.size(3)), mode='bilinear', align_corners=False)
            
            loss_ce = ce_criterion(pred, seg_labels)
            loss_dice = dice_criterion(pred, seg_labels)
            loss = loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ep_loss_sum += loss.item()

            if (optimizer.global_step - 1) % args.print_intervals == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d Iter:%5d/%5d' % (ep + 1, optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % loss.item(),
                      'Fin:%s' % (timer.str_est_finish()), flush=True)

        epoch_losses.append(ep_loss_sum / len(train_loader))

        if (ep + 1) % 10 == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_path, args.session_name + str(ep) + '.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), epoch_losses, marker='o', color='b', linewidth=2)
    plt.title('Segmentation Network Training Loss (BraTS)')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (CE + Dice)')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_path, 'seg_loss_curve.png'))
    print(f"Training Complete! Curve saved to {os.path.join(args.save_path, 'seg_loss_curve.png')}")
    