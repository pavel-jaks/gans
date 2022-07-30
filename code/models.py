from enum import Enum
from os import path

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x).reshape(batch_size, 1, 28, 28)


class GeneratorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1, 256, 9, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 512, 5, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, 5, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 1, 3, bias=False)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = x.reshape(batch_size, 1, 10, 10)
        return self.main(out)


class GeneratorConvDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4),
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 128 * 25 * 25),
            nn.ReLU()
        )
        self.norm = nn.Sequential(
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.linear(x)
        return self.conv(self.norm(out.reshape(batch_size, 128, 25, 25)))


class GeneratorDenseConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(1, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(128 * 13 * 13, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = x
        out = self.conv(out.reshape(batch_size, 1, 10, 10))
        return self.linear(out.reshape(batch_size, 128 * 13 * 13)).reshape(batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x.reshape(batch_size, 28 * 28))


class WassersteinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x.reshape(batch_size, 28 * 28))


class DiscriminatorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 256, 7),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 128, 5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, 9),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze(1).squeeze(1)


class WassersteinCriticConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 7),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, 9),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 4)
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze(1).squeeze(1)


class DiscriminatorConvDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 24 * 24, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 10),
            nn.LeakyReLU(0.1, True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv(x).reshape(batch, 64 * 24 * 24)
        return self.linear(out)


class CifarGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 8192),
            nn.LeakyReLU(0.01, True),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.01, True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.01, True),
            nn.Linear(4096, 3 * 32 * 32),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x).reshape(batch_size, 3, 32, 32)


class CifarDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(3 * 32 * 32, 8192),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x.reshape(batch_size, 3 * 32 * 32))


class CifarGeneratorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1, 1024, 5, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(1024, 512, 5,  bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(512, 256, 5, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(256, 3, 11,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.main(x.reshape(batch_size, 1, 10, 10))


class CifarDiscriminatorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, False),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, 4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),
            nn.Conv2d(64, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 1, 5),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).squeeze(1).squeeze(1)


class ModelType(Enum):
    Discriminator = Discriminator
    Generator = Generator
    DiscriminatorConv = DiscriminatorConv
    GeneratorConv = GeneratorConv
    DiscriminatorConvDense = DiscriminatorConvDense
    GeneratorConvDense = GeneratorConvDense
    CifarDiscriminator = CifarDiscriminator
    CifarGenerator = CifarGenerator
    CifarDiscriminatorConv = CifarDiscriminatorConv
    CifarGeneratorConv = CifarGeneratorConv
    GeneratorDenseConv = GeneratorDenseConv
    WassersteinCritic = WassersteinCritic
    WassersteinCriticConv = WassersteinCriticConv


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
