import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, data_dir='./data'):
    # 정규화 전 transform
    transform_no_norm = transforms.Compose([
        transforms.ToTensor()
    ])

    # 정규화 후 transform
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # 정규화 없이
    train_no_norm = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_no_norm)
    test_no_norm  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_no_norm)

    # 정규화 있음
    train_norm = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_norm)
    test_norm  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_norm)

    loaders = {
        'no_norm': {
            'train': DataLoader(train_no_norm, batch_size=batch_size, shuffle=True,  num_workers=2),
            'test':  DataLoader(test_no_norm,  batch_size=batch_size, shuffle=False, num_workers=2)
        },
        'norm': {
            'train': DataLoader(train_norm, batch_size=batch_size, shuffle=True,  num_workers=2),
            'test':  DataLoader(test_norm,  batch_size=batch_size, shuffle=False, num_workers=2)
        }
    }
    return loaders

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')