from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


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
        root='./data'
    ):
        self.train = train
        self.cuda = cuda
        self.root = root
        
        self.mean = (0.491, 0.482, 0.447)
        self.std = (0.247, 0.243, 0.262)
        
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
               shift_limit=0.0625, scale_limit=0.1, 
                rotate_limit=45, interpolation=1, 
                border_mode=4, p=0.2
            ),
            A.CoarseDropout(
                max_holes=2, max_height=8, 
                max_width=8, p=0.1
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.ToGray(p=0.1),
            A.Normalize(
                mean=self.mean, 
                std=self.std,
                always_apply=True
            ),
            ToTensorV2()
        ])
        self.test_transforms = A.Compose([
            A.Normalize(
                mean=self.mean, 
                std=self.std,
                always_apply=True
            ),
            ToTensorV2()
        ])
        
        if self.train:
            self.transforms = self.train_transforms
        else:
            self.transforms = self.test_transforms
            
        self.shuffle = True if self.train else False
        self.classes = None
        
        
    def get_data(self):
        data = datasets.CIFAR10(
            self.root,
            train=self.train,
            transform=lambda img:self.transforms(image=np.array(img))["image"],
            download=True
        )
        self.classes = data.classes
        return data
            
    def get_loader(self):
        data = self.get_data()

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
        print(
            f"""[INFO] {'train' if self.train else 'test'} dataset of size {len(data)} loaded..."""
        )
        return data_loader
        