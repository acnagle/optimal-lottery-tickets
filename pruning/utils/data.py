import os
import torch
from torchvision import datasets, transforms

class MNIST:
    def __init__(self, args, **kwargs):
        super(MNIST, self).__init__()

        self.INPUT_SIZE = 784
        self.NUM_CLASSES = 10

        data_dir = os.path.join(args.data, "cifar10")
        
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_dir,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

class CIFAR10:
    def __init__(self, args, **kwargs):
        super(CIFAR10, self).__init__()

        self.INPUT_SIZE = 1024
        self.NUM_CLASSES = 10

        data_dir = os.path.join(args.data, "cifar10")

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447],
            std=[0.247, 0.243, 0.262]
        )

        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize
                    ]
                )
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
