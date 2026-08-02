import torch
import torch.optim as optim
import os
from datetime import datetime
from tqdm import tqdm

from models import Res18
from loss import *
from utils import plot_loss

class CLNet(object):
    def __init__(self, args, train_loader, val_loader):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.project_path = args.project_path
        self.record_path = args.record_path
        self.model_name = args.model_name
        self.task = args.task  # 'binary' or 'multiclass'

        model = Res18()
        if args.pretrained_path is not None:
            model.load_pretrain_weight(args.pretrained_path)

        self.lr = args.learning_rate
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1)

        if len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

        self.model = model.to('cuda')
        self.loss = Multiclass_SupConLoss()

    def run(self):
        # Define save directory
        save_dir = os.path.join(self.project_path, self.record_path, f"{self.task}_pretrain")
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "log.log")
        encoder_name = "binary_encoder.pth" if self.task == "binary" else "multiclass_encoder.pth"
        encoder_path = os.path.join(save_dir, encoder_name)

        record = {'train_loss': [], 'val_loss': []}
        val_interval = 2
        best_val_loss = float('inf')  # Track best val loss
        best_epoch = -1

        with open(log_path, "w") as log_file:
            log_file.write(str(datetime.now()) + "\n")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(epoch)
            record['train_loss'].append(train_loss)

            if epoch >= 10:
                self.scheduler.step()

            with open(log_path, "a") as log_file:
                log_file.write(
                    f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f}\n'
                )

            if epoch % val_interval == 0:
                val_loss = self.val(epoch)
                record['val_loss'].append(val_loss)

                with open(log_path, "a") as log_file:
                    log_file.write(
                        f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.4f}\n"
                    )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), encoder_path)
                    with open(log_path, "a") as log_file:
                        log_file.write(f">>> Saved best model at epoch {epoch} with val loss {val_loss:.4f}\n")

            with open(log_path, "a") as log_file:
                log_file.write(str(datetime.now()) + "\n")

            # Save loss curve after each epoch
            plot_path = os.path.join(save_dir, "loss_curve.png")
            plot_loss(record['train_loss'], record['val_loss'], val_interval=val_interval, save_path=plot_path)

        # Final log message
        with open(log_path, "a") as log_file:
            log_file.write(f"Best model saved at epoch {best_epoch} with val loss {best_val_loss:.4f}\n")

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        for aug1, aug2, label in train_bar:
            self.optimizer.zero_grad()
            loss = self.step(aug1, aug2, label)
            loss.backward()
            self.optimizer.step()

            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            average_loss = total_loss / total_num
            train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {average_loss:.4f}')

        return total_loss / total_num

    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for aug1, aug2, label in val_bar:
                loss = self.step(aug1, aug2, label)
                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                average_loss = total_loss / total_num
                val_bar.set_description(f'Val Epoch: [{epoch}/{self.epochs}] Loss: {average_loss:.4f}')
        return total_loss / total_num

    def step(self, aug1, aug2, label):
        aug1 = aug1.cuda()
        aug2 = aug2.cuda()
        _, out1 = self.model(aug1)
        _, out2 = self.model(aug2)
        return self.loss(out1, out2,label)
