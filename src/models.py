"""Neural architectures used by the final analysis."""
import torch
from torch import nn

class QuestionnaireTower(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_inputs,48),nn.ReLU(),nn.Dropout(.2),nn.Linear(48,16),nn.Sigmoid())
    def forward(self,x): return self.net(x)

class TinyVisualResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU())
        self.cam_conv=nn.Conv2d(48,16,1)
        self.gap=nn.AdaptiveAvgPool2d(1)
    def forward(self,image,return_cam=False):
        maps=self.cam_conv(self.features(image)); residual=torch.tanh(self.gap(maps).flatten(1))
        return (residual,maps) if return_cam else residual

def trainable_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable_parameters(TinyVisualResidual()) == 19648
assert trainable_parameters(QuestionnaireTower(31)) == 2320
assert trainable_parameters(QuestionnaireTower(32)) == 2368
