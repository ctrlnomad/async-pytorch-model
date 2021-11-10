import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from argparse import ArgumentParser
from classifier import ImageClassifier


import logging 
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

train_ds = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = True,
            download = True,
            transform = T.Compose([
                T.ToTensor()                                 
            ])
        )

valid_ds = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = False,
            download = True,
            transform = T.Compose([
                T.ToTensor()                                 
            ])
        )

def parse_args():
    parser = ArgumentParser()

    parser.add_argument()
    parser.add_argument()

    return parser.parse_args()
 

if __name__ == '__main__':
    # args = parse_args()
    NUM_EPOCHS = 10
    BATCH_SIZE = 100 
    SEED = 2021

    num_classes = len(train_ds.classes)
    clf = ImageClassifier(num_classes, batch_size=BATCH_SIZE)
    clf.seed(SEED)

    for e in range(NUM_EPOCHS):

        clf.train_once(train_ds)
        acc = clf.validate_acc(valid_ds)

        logger.info(f'[{e}] accuracy: {acc:.3f}')

    clf.save('./fmnist_network.pth')