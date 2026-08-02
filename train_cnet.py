import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ImageDataset
from cnet_multi import CNet
from utils import set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNet on BraTS Binary and Multiclass classification')
    
    parser.add_argument('--project_path', default='multi_cam_project')
    parser.add_argument('--record_path', default='train_record_classification')
    parser.add_argument('--modality', default='flair_t1ce_t2')
    parser.add_argument('--binary_epochs', type=int, default=20)
    parser.add_argument('--multiclass_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=155)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--dataset_type', type=str, default='brats')

    return parser.parse_args()

def train_model(args, train_df, val_df, config, task, encoder_pretrained_path=None):
    args.task = task

    if config['task'] == 'binary':
        args.num_classes = 1
        args.epochs = args.binary_epochs
        args.learning_rate = 1e-3
    elif config['combine'] is not None:
        args.num_classes = len(config['combine'])
        args.epochs = args.multiclass_epochs
        args.learning_rate = 5e-4
    else:
        raise ValueError("For multiclass task, 'combine' must be specified in config.")

    model_dir = os.path.join(args.project_path, args.record_path, f"{task}_clstrain")
    os.makedirs(model_dir, exist_ok=True)
    args.model_name = f"{task}_train"
    args.encoder_pretrained_path = os.path.join(args.project_path, 'pretrain_record', f'{task}_pretrain', f'{task}_encoder.pth') if encoder_pretrained_path is None else encoder_pretrained_path
    args.pretrained_path = None
    train_dataset = ImageDataset(train_df, args.img_size, config, mode='train')
    val_dataset = ImageDataset(val_df, args.img_size, config, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    cnet = CNet(args, train_loader, val_loader)
    cnet.run()

    model_name = "binary_classifier.pth" if task == "binary" else "multi_classifier.pth"
    model_path = os.path.join(model_dir, model_name)
    return model_path


def main():
    args = parse_args()
    set_seed(42)

    # --- Binary Training ---
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    binary_config = {
        'dataset': args.dataset_type,
        'task': 'binary',
        'combine': None
    }

    train_model(args, train_df, val_df, binary_config, task='binary')

    # # --- Multiclass Training ---

    multiclass_config = {
        'dataset': args.dataset_type,
        'task': 'multiclass',
        'combine': {
            'core': ['necrosis', 'enhancing'],
            'edema': ['edema']
        }
    }

    train_model(args, train_df, val_df, multiclass_config, task='multiclass')

if __name__ == '__main__':
    main()
