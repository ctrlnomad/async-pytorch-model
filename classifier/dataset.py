import torch
import torchvision
import torchvision.transforms as T

train_set = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = True,
            download = True,
            transform = T.Compose([
                T.ToTensor()                                 
            ])
        )

valid_set = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = False,
            download = True,
            transform = T.Compose([
                T.ToTensor()                                 
            ])
        )


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, i):
        pass

    def __len__(self):
        pass
