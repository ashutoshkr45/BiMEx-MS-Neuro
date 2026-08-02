import torch
import torch.nn as nn
import torch.nn.functional as F
import network.resnet38d

class Net(network.resnet38d.Net):
    def __init__(self, num_classes=2):
        super().__init__()

        # Classification head: 4096 -> num_classes
        self.fc8_cls = nn.Conv2d(4096, num_classes, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_cls.weight)
        torch.nn.init.zeros_(self.fc8_cls.bias)

        self.from_scratch_layers = [self.fc8_cls]

    def forward(self, x):
        x = super().forward(x)                          # [B, 4096, H', W']
        x = F.relu(x, inplace=False)
        logits = self.fc8_cls(x)                        # [B, num_classes, H', W']
        logits = F.adaptive_avg_pool2d(logits, (1, 1))  # [B, num_classes, 1, 1]
        logits = logits.view(logits.size(0), -1)        # [B, num_classes]
        return logits

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param
                