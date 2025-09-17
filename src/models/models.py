import torch.nn as nn
import torch
import yaml 
import os
from . import modules

def get_block(*args):
    match args:
        case [key, values]:
            # nn Modules
            if key in ('ConvTranspose2d', 'Conv2d'):
                return getattr(nn, key)(*values)
            # activation
            elif key.lower() in ('sigmoid', 'tanh', 'relu', 'silu', 'selu'):
                return getattr(nn, key)()
            # self-defined Block
            else:
                try:
                    return getattr(modules, key)(*values)
                except Exception as e:
                    raise e
                
class BaseModel(nn.Module):
    def __init__(self, cfg_file:yaml, key):
        super().__init__()
        self.cfg_file = os.path.join('src/cfg', cfg_file)
        self.key = key
        self.model = self._get_model()
        self.yaml_file = None

        self.initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _get_model(self):
        with open(self.cfg_file, 'r') as file:
            self.yaml_file = yaml.safe_load(file)
        module_list = nn.ModuleList([get_block(*values) for values in self.yaml_file[self.key]])
        return nn.Sequential(*module_list)
    
    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
class Discriminator(BaseModel):
    def __init__(self, cfg_file:yaml):
        super().__init__(cfg_file, 'discriminator')
    

class Generator(BaseModel):
    def __init__(self, cfg_file:yaml):
        super().__init__(cfg_file, 'generator')
    
    def forward(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        x = torch.randn(32, 100, 1, 1).to(device=device)
        return super().forward(x)