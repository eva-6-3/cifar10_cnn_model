from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10_dataset():
    """
    a utility class to get data loader for CIFAR10.
    transforms are hardcoded for now.
    
    USAGE:
        train_loader = CIFAR10_dataset(
            train=True, cude=torch.cuda.is_available()
        ).get_loader()
    """
    def __init__(
        self,
        train, cuda,
        root='./data',
    ):
        self.train = train
        self.cuda = cuda
        self.root = root
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.train:
            self.transforms = self.train_transforms
        else:
            self.transforms = self.test_transforms
            
        self.shuffle = True if self.train else False
        self.classes = (
            'plane', 'car', 'bird', 
            'cat', 'deer', 'dog', 'frog', 
            'horse', 'ship', 'truck'
        )
            
    def get_loader(self):
        data = datasets.CIFAR10(
            self.root,
            train=self.train,
            transform=self.transforms,
            download=True
        )
        dataloader_args = dict(
            shuffle=self.shuffle, 
            batch_size=128, 
            num_workers=2, 
            pin_memory=True
        ) if self.cuda else dict(
            shuffle=self.shuffle, 
            batch_size=64
        )
        data_loader = DataLoader(data, **dataloader_args)
        print(f"""
        [INFO] {'train' if self.train else 'test'} dataset of size {len(data)} loaded...
        """)
        return data_loader
