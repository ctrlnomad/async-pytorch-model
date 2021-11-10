import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import numpy as np

from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def make_network(num_classes: int):
    net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=12*4*4, out_features=120),
        nn.ReLU(),
        nn.Linear(in_features=120, out_features=60),
        nn.ReLU(),
        nn.Linear(in_features=60, out_features=num_classes),
        nn.ReLU()
    )
    return net


class ImageClassifier:
    def  __init__(self, num_classes,  batch_size: int = 32, gpu: bool = False,
                  opt: callable =optim.Adam, 
                  criterion: callable = nn.CrossEntropyLoss) -> None:

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.net = make_network(num_classes)
        self.gpu = gpu

        if self.gpu:
            self.net = self.net.cuda()

        self.opt = opt(self.net.parameters())
        self.criterion = criterion()


    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, num_epochs):
        self.net.train()

        for e in range(num_epochs):
            self.train_once()

    def validate_acc(self, val_ds):
        self.net.eval()

        loader = DataLoader(val_ds, batch_size=len(val_ds))

        X, y = next(iter(loader))

        pred = torch.softmax(self.net(X), dim=-1)

        self.net.train()

        return (pred.argmax(dim=-1) == y).sum() / len(val_ds)
            

    def train_once(self, trnds: Dataset):

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