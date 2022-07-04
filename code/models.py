from enum import Enum
from os import path

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 10024),
            nn.Sigmoid(),
            nn.Linear(10024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x).reshape(batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 22),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze(1).squeeze(1).squeeze(1)


class ModelType(Enum):
    Generator = Generator
    Discriminator = Discriminator


class ModelManager:
    @staticmethod
    def get_untrained(model_type: ModelType) -> nn.Module:
        return model_type.value()
    
    @staticmethod
    def get_trained(model_type: ModelType) -> nn.Module:
        try:
            return torch.load(path.join('models', f'{model_type.name}.model'))
        except:
            raise ValueError(f'Model with name {model_type.name} is not trained.')
    
    @staticmethod
    def save_model(model_type: ModelType, model: nn.Module):
        torch.save(model, path.join('models', f'{model_type.name}.model'))
