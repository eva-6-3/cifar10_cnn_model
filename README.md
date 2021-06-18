# cifar10 cnn Advanced Concepts

# Group: EVA6 - Group 3
1. Muhsin Abdul Mohammed - muhsin@omnilytics.co 
2. Nilanjana Dev Nath - nilanjana.dvnath@gmail.com
3. Pramod Ramachandra Bhagwat - pramod@mistralsolutions.com
4. Udaya Kumar NAndhanuru - udaya.k@mistralsolutions.com
------

# Data Exploration
CIFAR-10 contains 1000 images per class for test, and 5000 images per class for train.<br>
The classes on CIFAR-10 are Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.<br>
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/train_class_distribution.png" alt="train_class_distribution" width="500"/>

### mean and std for dataset
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/mean_std_dataset.png" alt="mean_std_dataset" width="500"/>

### Some sample images from train set -- 
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/train_rand_images_1.png" alt="train_rand_images_1" width="400"/>
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/train_rand_images_2.png" alt="train_rand_images_2" width="400"/>

### Some sample images from test set -- 
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/test_rand_images_1.png" alt="test_rand_images_1" width="400"/>
<img src="https://github.com/askmuhsin/cifar10_cnn_model/blob/main/resources/test_rand_images_2.png" alt="test_rand_images_2" width="400"/>

# Model Architecture
Our architecture is C1C2C3C40 without using any pooling operations.<br>
Instead of pooling we used a combination of dilated convolutions and strided covolutions. <br>
In C3 we have also used depthwise seperable convolutions instead of normal covolution layers.<br>
The total number of parameters for our model is 96,596. <br>
We have not used any Dense Layer, Instead we targeted GAP to get the output classes dim.

```python
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = self.dilated_conv_1(x)

        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.dilated_conv_2(x)
        
        x = self.sep_conv_1(x)
        x = self.sep_conv_2(x)
        x = self.strided_conv_1(x)
        
        x = self.convblock_5(x)
        x = self.convblock_6(x)

        x = self.gap(x)
        x = x.view(-1, 10)
```

# Augmentation
```python
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
               shift_limit=0.0625, scale_limit=0.1, 
                rotate_limit=45, interpolation=1, 
                border_mode=4, p=0.5
            ),
            A.CoarseDropout(
                max_holes=2, max_height=8, 
                max_width=8, p=0.3
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
```

# Training



# Goals 
- [X] Model is trained on GPU
- [X] change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- [X] total RF must be more than 44. _(Bonux points if RF > 52)_
- [X] one of the layers must use Depthwise Separable Convolution. _(Bonus points for two layers)_
- [X] one of the layers must use Dilated Convolution
- [X] use GAP (compulsory):- add FC after GAP to target #of classes (optional) _(if optional achieved Bonus points)_
- [X] use albumentation library and apply:
  - [X] horizontal flip
  - [X] shiftScaleRotate
  - [x] coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - [X] grayscale _(For Bonus points)_
- [ ] achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.  _(Bonus for 87% acc, and <100k params)_
- [X] upload to Github
