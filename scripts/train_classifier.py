import torchvision
import torchvision.transforms as T

from argparse import ArgumentParser
from fmnist_classifier import ImageClassifier


import logging 
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

MAX_PIXEL_VALUE = 255

train_ds = torchvision.datasets.FashionMNIST(
            root = './data',
            train = True,
            download = True,
            transform = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: x / MAX_PIXEL_VALUE)                            
            ])
        )

valid_ds = torchvision.datasets.FashionMNIST(
            root = './data',
            train = False,
            download = True,
            transform = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: x / MAX_PIXEL_VALUE)                
            ])
        )

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--save_path', default='./fmnist_network.pth')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--seed', default=2021, type=int)

    return parser.parse_args()
 

if __name__ == '__main__':
    args = parse_args()

    num_classes = len(train_ds.classes)
    clf = ImageClassifier(num_classes, batch_size=args.batch_size, lr=args.lr)
    clf.seed(args.seed)

    for e in range(args.num_epochs):

        clf.train_once(train_ds)
        acc = clf.validate_acc(valid_ds)

        logger.info(f'[{e}] accuracy: {acc:.3f}')

    clf.save(args.save_path)