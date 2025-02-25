
from typing import Dict, Any

from pathlib import Path

import torch
import torch.utils.data 
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


# This is the path to the folder containing the images
DEFAULT_IMAGE_FOLDER_PATH = Path('example_dataset/')

# This script is set up to crop first, then resize
DEFAULT_CENTRE_CROP_SIZE = 300
DEFAULT_RESIZED_IMAGE_SIZE = 100


def get_transforms(grayscale: bool = False, crop_size: int = DEFAULT_CENTRE_CROP_SIZE, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
    """
    This function returns the transforms that are applied to the images when they are loaded.
    """
    # Set up the transforms on load of the data
    if grayscale:
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(size=400),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.8),
                transforms.CenterCrop(crop_size),
                transforms.Resize(resize_size),
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
        # In this case, as we aren't doing any kind of random augmentation we 
        # can use the same transforms for the test data as the train data
        test_transforms = train_transforms
    else:
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(size=400),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)], p=0.8),
                transforms.CenterCrop(crop_size),
                transforms.Resize(resize_size),
                transforms.ToTensor()
            ]
        )
    # The test transforms are the same as the train transforms but without the random augmentations
    if grayscale:
        test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize(resize_size),
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
    else:
        test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize(resize_size),
                transforms.ToTensor()
            ]
        )
    return train_transforms, test_transforms


def load_image_targets_from_csv(csv_path: Path, header: bool = True) -> Dict[str, Any]:
    """
    This function loads the image targets from a csv file. It assumes that the csv file
    has a header row and that the first column contains the image path and all the subsequent
    columns contain the target values which are bundled together into a numpy array.
    """
    image_targets = {}
    with csv_path.open('r') as f:
        lines = f.readlines()
        start_line = 0
        # If there is a header, skip the first line
        if header:
            header_line = lines[0].strip().split(',')
            print(f'Header line of csv {csv_path} : {header_line}')
            start_line = 1
        for line in lines[start_line:]:
            line = line.strip().split(',')
            image_path = line[0]
            # ONLY FL not U
            image_targets[image_path] = np.array([float(line[1]),float(line[2])], dtype=np.float32)
    return image_targets


class RegressionImageFolder(datasets.ImageFolder):
    """
    The regression image folder is a subclass of the ImageFolder class and is designed for 
    image regression tasks rather than image classification tasks. It takes in a dictionary
    that maps image paths to their target values.
    """
    def __init__(
        self, root: str, image_targets: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(root, *args, **kwargs)
        paths, _ = zip(*self.imgs)
        self.targets = []
        self.samples = []
        for path in paths:
            if str(path) in image_targets:
                self.targets.append(image_targets[str(path)])
                self.samples.append((path, image_targets[str(path)]))

class RegressionTaskData:
    """
    This class is a wrapper for the data that is used in the regression task. It contains
    the train and test loaders.
    """
    def __init__(
        self,
        grayscale: bool = False,
        image_folder_path: Path = DEFAULT_IMAGE_FOLDER_PATH,
        crop_size: int = DEFAULT_CENTRE_CROP_SIZE,
        resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE,
    ) -> None:
        self.grayscale = grayscale
        self.image_folder_path = image_folder_path
        self.train_transforms, self.test_transforms = get_transforms(grayscale, crop_size, resize_size)
        self._trainloader = None
        self._testloader = None
        self.crop_size = crop_size
        self.resize_size = resize_size

    @property
    def output_image_size(self):
        return (1 if self.grayscale else 3, self.resize_size, self.resize_size)
    
    @property
    def trainloader(self):
        if self._trainloader is None:
            self._trainloader = self.make_trainloader()
        return self._trainloader
    
    @property
    def testloader(self):
        if self._testloader is None:
            self._testloader = self.make_testloader()
        return self._testloader

    def make_dataloader(
            self,
            image_folder_path: Path,
            image_targets: Dict[str, Any],
            randomise: bool = True,
            transform: transforms.Compose = None
    ) -> torch.utils.data.DataLoader:
        """
        Builds a data loader that just will just run the model on the images in the image_folder_path
        """
        if transform is None:
            data = RegressionImageFolder(
                str(image_folder_path), 
                image_targets=image_targets,
                transform=self.test_transforms
            )
        else:
            data = RegressionImageFolder(
                str(image_folder_path), 
                image_targets=image_targets,
                transform=transform
            )
        # This constructs the dataloader that actually determins how images will be loaded in batches
        dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=randomise, num_workers=4)
        return dataloader

    def make_trainloader(
            self, 
        ) -> torch.utils.data.DataLoader:
        """
        Builds the train data loader
        """
        train_targets = load_image_targets_from_csv(self.image_folder_path / 'train.csv')
        return self.make_dataloader(self.image_folder_path / 'train', train_targets, transform=self.train_transforms)

    def make_testloader(
            self, 
        ) -> torch.utils.data.DataLoader:
        """
        Builds the test data loader
        """
        test_targets = load_image_targets_from_csv(self.image_folder_path / 'test.csv')
        return self.make_dataloader(self.image_folder_path / 'test', test_targets, transform=self.test_transforms)

    def visualise_image(self):
        """
        This function visualises a single image from the train set
        """
        images, targets = next(iter(self.trainloader))
        print(targets[0].shape)
        print(images[0].shape)
        if self.grayscale:
            plt.imshow(images[0][0, :, :], cmap='gray')
        else:
            plt.imshow(images[0].permute(1, 2, 0))
        plt.show()


if __name__ == '__main__':
    data = RegressionTaskData(grayscale=True, resize_size=100)
    data.visualise_image()
