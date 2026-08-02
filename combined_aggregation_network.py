import torch
import torch.optim as optim
import os
from datetime import datetime
from collections import OrderedDict

from models import *
from loss import *
from tqdm import tqdm
from utils import *

class AggNet(object):
    
    def __init__(self, args, train_loader, val_loader):

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.project_path = args.project_path
        self.record_path = args.record_path
        self.model_name = args.model_name
        self.task = args.task  # 'binary' or 'multiclass'
        self.lr = args.learning_rate
        self.loss_weight = args.loss_weight if args.loss_weight is not None else [1.0, 1.0, 1.0, 1.0, 5.0] 

        # Define save directory
        self.save_dir = os.path.join(self.project_path, self.record_path, f"{self.task}_aggtrain")
        os.makedirs(self.save_dir, exist_ok=True)

        if self.task == "binary":
            binary_model = Res18_Classifier(num_classes=1)
            bin_score_model = Res_Scoring()

            if args.bin_pretrained_path != None:
                binary_model.load_pretrain_weight(args.bin_pretrained_path)
                print("Loaded encoder pretrain weight for binary model")
            
            for param in binary_model.parameters():
                param.requires_grad = False

            if len(args.gpu_ids) > 1:
                binary_model = torch.nn.DataParallel(binary_model, device_ids=args.gpu_ids)
                bin_score_model = torch.nn.DataParallel(bin_score_model, device_ids=args.gpu_ids)
            self.binary_model = binary_model.to('cuda').eval()
            self.model = bin_score_model.to('cuda')
            
        else:
            binary_model = Res18_Classifier(num_classes=1)
            multi_model = Res18_Classifier(num_classes=self.num_classes)
            
            bin_score_model = Res_Scoring()
            multi_score_model = Res_Scoring()

            if args.bin_pretrained_path != None:
                binary_model.load_pretrain_weight(args.bin_pretrained_path)
                print("Loaded encoder pretrain weight for binary model")
            if args.multi_pretrained_path != None:
                multi_model.load_pretrain_weight(args.multi_pretrained_path)
                print("Loaded encoder pretrain weight for multi model")
            if args.bin_score_model_pretrained_path != None:
                bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_path)
                print("Loaded encoder pretrain weight for binary score model")

            for param in multi_model.parameters():
                param.requires_grad = False
            
            for param in binary_model.parameters():
                param.requires_grad = False
            
            for param in bin_score_model.parameters():
                param.requires_grad = False

            if len(args.gpu_ids) > 1:
                binary_model = torch.nn.DataParallel(binary_model, device_ids=args.gpu_ids)
                multi_model = torch.nn.DataParallel(multi_model, device_ids=args.gpu_ids)
                bin_score_model = torch.nn.DataParallel(bin_score_model, device_ids=args.gpu_ids)
                multi_score_model = torch.nn.DataParallel(multi_score_model, device_ids=args.gpu_ids)

            self.binary_model = binary_model.to('cuda').eval()
            self.bin_score_model = bin_score_model.to('cuda').eval()
            self.multi_model = multi_model.to('cuda').eval()
            self.model = multi_score_model.to('cuda')

        self.score_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.score_optimizer, T_max=self.epochs, eta_min=0.000005)
    
        self.loss = nn.BCEWithLogitsLoss()
        self.sminloss_intra = SimMinLoss()
        self.sminloss_inter = SimMinLoss(intra=False)
        self.smaxloss = SimMaxLoss_intraclass()
        self.aggrementloss=AggrementLoss()

        self.log_path = os.path.join(self.save_dir, "log.log")
        
    def run(self):

        with open(self.log_path, "w+") as log_file:
            log_file.writelines(str(datetime.now()) + "\n")

        train_record = {'loss':[], 'clloss': []}
        val_record = {'loss':[], 'clloss': []}
        best_score = 100000

        val_interval = 2
        best_epoch = -1

        for epoch in range(1, self.epochs + 1):
            train_loss, train_clloss_epoch = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['clloss'].append(train_clloss_epoch.copy())  # Store a copy of the current epoch's clloss
 
            #change loss scaling

            if epoch > 10:
                self.scheduler.step()
            
            with open(self.log_path, "a") as log_file:
                clloss_str = ', '.join([f"{k}: {v:.4f}" for k, v in train_clloss_epoch.items()])
                log_file.writelines(
                    f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f}, CL Loss: {clloss_str}\n'
                )
            
            if epoch % val_interval == 0:
                val_loss, val_clloss_epoch = self.val(epoch)
                val_record['loss'].append(val_loss)
                val_record['clloss'].append(val_clloss_epoch.copy())  # Store a copy of the current epoch's clloss

                with open(self.log_path, "a") as log_file:
                    clloss_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_clloss_epoch.items()])
                    log_file.writelines(
                        f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.4f}, CL Loss: {clloss_str}\n"
                    )

                cur_score = val_loss
                if cur_score < best_score:
                    best_score = cur_score
                    with open(self.log_path, "a") as log_file:
                            log_file.write(f">>> Saved best model at epoch {epoch} with val Loss {val_loss:.4f}\n")
                    self.score_model_path = os.path.join(self.save_dir, "score_model.pth")
                    torch.save(self.model.state_dict(), self.score_model_path)
                    best_epoch = epoch

            plot_path = os.path.join(self.save_dir, "loss_curve.png")
            plot_loss(train_record['loss'], val_record['loss'], val_interval=val_interval, save_path=plot_path)

        # Final log message
        with open(self.log_path, "a") as log_file:
            log_file.write(f"Best model saved at epoch {best_epoch} with val loss {best_score:.4f}\n")

    def train(self, epoch):
        self.model.train()

        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        clloss_epoch={}

        with open(self.log_path, "a") as log_file:
            for idx, (case_batch, label_batch) in enumerate(train_bar):

                case_batch = case_batch.cuda()
                label_batch = label_batch.cuda()
                pred_binary, pred_multi, map_collect_binary, map_collect_multi = self.step(case_batch, label_batch)
            
                self.score_optimizer.zero_grad()
                clloss, clloss_collect = self.score_step(case_batch, label_batch, map_collect_binary, map_collect_multi)
                clloss.backward()
                self.score_optimizer.step()

                total_num += case_batch.size(0)
                total_loss += (clloss.item()) * case_batch.size(0)
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
            
                for k, v in {**clloss_collect}.items():
                    clloss_epoch[k] = (clloss_epoch.get(k, 0)) + v.item()* case_batch.size(0)
        
        for k in clloss_epoch.keys():
            clloss_epoch[k] = clloss_epoch[k] / total_num
        
        return total_loss / total_num, clloss_epoch

    def step(self, data_batch, label_batch):
        
        logits_collect_binary, map_collect_binary = self.binary_model(data_batch)
        pred_binary = torch.sigmoid(logits_collect_binary[-1])

        if self.task == "binary":
            return pred_binary.cpu(), None, map_collect_binary, None
        
        else:
            # For multi-class task, we also need the multi-class model
            logits_collect_multi, map_collect_multi = self.multi_model(data_batch.cuda())
            pred_multi = torch.sigmoid(logits_collect_multi[-1])
            return pred_binary.cpu(), pred_multi.cpu(), map_collect_binary, map_collect_multi


    def normalize_map(self, tensor):
        a1,a2,a3,a4=tensor.size()
        tensor = tensor.view(a1, a2, -1)
        tensor_min = (tensor.min(2, keepdim=True)[0])
        tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4)
        return tensor
    
    def score_step(self, data_batch, label_batch, map_collect_binary, map_collect_multi):

        loss = 0
        loss_collect = {}

        if self.task == "binary":
            _, foreground, background, binary_ame_map = self.model(data_batch, map_collect_binary)
            loss_collect["SimMax_Foreground_intra"] = (self.loss_weight[0]*self.smaxloss(foreground))
            loss_collect["SimMax_background_intra"] = (self.loss_weight[1]*self.smaxloss(background))
            loss_collect["SimMin_intra_foreground_background"] = (self.loss_weight[2]*self.sminloss_intra(foreground, background))

        else:
            _, binary_foreground, binary_background, binary_ame_map = self.bin_score_model(data_batch, map_collect_binary)

            binary_ame_map = self.normalize_map(binary_ame_map)
            map_collect_multi = torch.stack(map_collect_multi, dim=0)
            map_collect= binary_ame_map * map_collect_multi
            map_collect = map_collect.unbind(0)
            map_collect=[map_collect[i] for i in range(len(map_collect))]
            _, foreground, background, multi_ame_map = self.model(data_batch, map_collect)
        
            loss_collect["SimMax_Foreground_intra"] = (self.loss_weight[0]*self.smaxloss(foreground))
            loss_collect["SimMax_background_intra"] = (self.loss_weight[1]*self.smaxloss(background))
            loss_collect["SimMin_intra_foreground_background"] = (self.loss_weight[2]*self.sminloss_intra(foreground, background))
            loss_collect["SimMMin_inter_foreground"] = (self.loss_weight[3]*self.sminloss_inter(foreground,foreground))
            loss_collect["Aggrement_loss"] = (self.loss_weight[4]*self.aggrementloss(binary_ame_map,multi_ame_map))

        for k, l in loss_collect.items():
            loss += l

        return loss, loss_collect
    
    def val(self, epoch):
        self.model.eval()
            
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        clloss_epoch = {}

        with torch.no_grad():
            for idx, (case_batch, label_batch) in enumerate(val_bar):
                case_batch = case_batch.cuda()
                label_batch = label_batch.cuda()

                pred_binary, pred_multi, map_collect_binary, map_collect_multi = self.step(case_batch, label_batch)
                clloss, clloss_collect = self.score_step(case_batch, label_batch, map_collect_binary, map_collect_multi)

                total_num += case_batch.size(0)
                total_loss += clloss.item() * case_batch.size(0)
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))

                for k, v in clloss_collect.items():
                    clloss_epoch[k] = clloss_epoch.get(k, 0.0) + v.item() * case_batch.size(0)

        for k in clloss_epoch:
            clloss_epoch[k] /= total_num

        return total_loss / total_num, clloss_epoch
