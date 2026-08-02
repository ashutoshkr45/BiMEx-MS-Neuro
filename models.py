import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from resnet18_self import ResNet18
from collections import OrderedDict

class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        resnet=resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        #encoder
        self.f = nn.Sequential(*list(resnet.children())[:-1])
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))
        
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        
        return feature, out
    
    def load_pretrain_weight(self, pretrain_path):

        print("Model restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)
        state_dict_init = self.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            new_state_dict[k_0] = v
            print(f"Mapping: {k} -> {k_0}")
        self.load_state_dict(new_state_dict, strict=False)

class Res18_Classifier(nn.Module):
    def __init__(self, num_classes=1, pretrain_path=None):
        super(Res18_Classifier, self).__init__()

        self.f = ResNet18(norm_layer=nn.InstanceNorm2d)  # returns 4 intermediate features
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ic1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.ic2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.ic3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.ic4 = nn.Conv2d(512, num_classes, kernel_size=1)

        if pretrain_path is not None:
            self.load_pretrain_weight(pretrain_path)
        else:
            print("Model from scratch")

    def forward(self, x, return_maps=True):
        batch_size, _, H, W = x.shape

        l1, l2, l3, l4 = self.f(x)

        l1_map = self.ic1(l1)
        l2_map = self.ic2(l2)
        l3_map = self.ic3(l3)
        l4_map = self.ic4(l4)

        # Global average pooled logits (for classification losses)
        l1_logits = torch.flatten(self.gap(l1_map), start_dim=1)
        l2_logits = torch.flatten(self.gap(l2_map), start_dim=1)
        l3_logits = torch.flatten(self.gap(l3_map), start_dim=1)
        l4_logits = torch.flatten(self.gap(l4_map), start_dim=1)

        logits_collect = [l1_logits, l2_logits, l3_logits, l4_logits]

        if return_maps:
            # Resize to input resolution (for CAMs or external models)
            re_l1 = F.interpolate(l1_map, size=(H, W), mode='bilinear', align_corners=False).detach()
            re_l2 = F.interpolate(l2_map, size=(H, W), mode='bilinear', align_corners=False).detach()
            re_l3 = F.interpolate(l3_map, size=(H, W), mode='bilinear', align_corners=False).detach()
            re_l4 = F.interpolate(l4_map, size=(H, W), mode='bilinear', align_corners=False).detach()
            map_collect = [re_l1, re_l2, re_l3, re_l4]
        else:
            map_collect = None

        return logits_collect, map_collect

    def normalize(self, tensor):
        a1, a2, a3, a4 = tensor.size()
        tensor = tensor.view(a1, a2, -1)
        min_val = tensor.min(dim=2, keepdim=True)[0]
        max_val = tensor.max(dim=2, keepdim=True)[0]
        norm = (tensor - min_val) / (max_val - min_val + 1e-5)
        return norm.view(a1, a2, a3, a4)

    def load_pretrain_weight(self, pretrain_path):
        print("Model restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)
        state_dict_init = self.state_dict()

        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            new_state_dict[k_0] = v
            print(f"Mapping: {k} -> {k_0}")

        self.load_state_dict(new_state_dict, strict=False)

    def load_encoder_pretrain_weight(self, pretrain_path):
        print("Encoder restore from", pretrain_path)
        state_dict_weights = torch.load(pretrain_path)
        state_dict_init = self.state_dict()

        new_state_dict = OrderedDict()
        for (k, v), (k_0, _) in zip(state_dict_weights.items(), state_dict_init.items()):
            if k.startswith("f."):
                new_state_dict[k_0] = v
                print(f"Mapping encoder: {k} -> {k_0}")

        self.load_state_dict(new_state_dict, strict=False)


class Res_Scoring(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res_Scoring, self).__init__()
        
        #convert 3 channel image to 1 channel but keep the same size
        self.proj = nn.Conv2d(3, 1, 1)
        
        #3d attention
        self.att = nn.Sequential(
            nn.Conv3d(4, 32, 3, padding=1, bias=False),
            #padding=1 to keep the same size
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 4, 3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU()   
        )

    def forward(self, input, map_collect):
        # Stack input CAMs: shape [batch, 4, num_classes, h, w]
        mask = torch.stack(map_collect, dim=1)

        # Normalize CAMs across spatial dimensions
        norm_mask = self.normalize(mask)
        
        # Convert input RGB image to 1 channel features: [batch, 1, h, w]
        input_gray = self.proj(input)
        

        # Multiply grayscale input with normalized CAMs: [batch, 4, num_classes, h, w]
        masked_input = input_gray.unsqueeze(1) * norm_mask
        
        # Pass through 3D attention/aggregation module
        map_att = self.att(masked_input)
        
        # Permute for softmax across layers: [batch, 4, num_classes, h, w] -> [batch, num_classes, 4, h, w] 
        map_att = map_att.permute(0, 2, 1, 3, 4)
        map_weight = F.softmax(map_att, dim=2)
        map_weight = map_weight.permute(0, 2, 1, 3, 4) #[batch, num_classes, 4, h, w] -> [batch, 4, num_classes, h, w]

        # Weighted sum of maps using attention: [batch, num_classes, h, w]
        final_map = torch.sum(mask * map_weight, dim=1)

        # Compute foreground and background features
        foreground = input_gray * final_map
        foreground = torch.flatten(foreground, start_dim=2)
        background = input_gray * (1 - final_map)
        background = torch.flatten(background, start_dim=2)
        
        #average map
        map_collect.append(final_map)
        all_map = torch.stack(map_collect, dim=1)
        average_map = torch.mean(all_map, dim=1, keepdim=True)
        
        return average_map, foreground, background, final_map

    def normalize(self, tensor):
        a1, a2, a3, a4, a5= tensor.size()
        tensor = tensor.view(a1, a2, a3, -1)
        tensor_min = (tensor.min(2, keepdim=True)[0])
        tensor_max = (tensor.max(2, keepdim=True)[0])
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4, a5)
        return tensor

    def load_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")

    def load_encoder_pretrain_weight(self, pretrain_path):
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()

            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                if "f" in k:
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")
