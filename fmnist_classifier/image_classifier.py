import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import numpy as np
import random

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def make_network(num_channels: int, num_classes: int):
    net = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),

        nn.LazyLinear(out_features=120),
        nn.ReLU(),
        nn.LazyLinear(out_features=60),
        nn.ReLU(),
        nn.LazyLinear(out_features=num_classes)
        )
    return net


class ImageClassifier:
    def  __init__(self, num_classes:int=10, num_channels:int =1,   
                  batch_size: int = 32, lr:float = 3e-4,
                  device:torch.device = torch.device('cpu'),
                  opt: callable =optim.Adam, 
                  criterion: callable = nn.CrossEntropyLoss
                  ) -> None:

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lr = lr

        self.device = device
        self.net = make_network(num_channels, num_classes).to(self.device)

        self.opt = opt(self.net.parameters(),lr=self.lr)
        self.criterion = criterion()


    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(0)
        
    def validate_acc(self, val_ds):
        self.net.eval()

        loader = DataLoader(val_ds, batch_size=len(val_ds))

        X, y = next(iter(loader))

        pred = torch.softmax(self.net(X), dim=-1)

        self.net.train()

        return (pred.argmax(dim=-1) == y).sum() / len(val_ds)
            

    def train_once(self, trnds: Dataset):
        self.net.train()

        loader = DataLoader(trnds, batch_size=self.batch_size, shuffle=True)
        losses = []

        for X, y in tqdm(loader):

            output = self.net(X)

            loss = self.criterion(output, y)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

        logger.info(f'loss mean: {np.mean(losses):.3f}')

    def save(self, path):
        D = {
            'opt': self.opt.state_dict(),
            'model': self.net.state_dict(),
            'num_classes':  self.num_classes
        }
        torch.save(D, path)
        logger.info(f'saved model under path: [{path}]')
        
    @staticmethod
    def from_path(path):
        D = torch.load(path)

        cls = ImageClassifier(D['num_classes'])
        cls.net.load_state_dict(D['model'])
        cls.opt.load_state_dict(D['opt'])
        
        return cls


    def load_model(self, path):
        self.net.load_state_dict(torch.load(path)['model'])

    def predict(self, x: torch.Tensor):
        self.net.eval()
        output = self.net(x)
        return torch.softmax(output, dim=-1).argmax(dim=-1)