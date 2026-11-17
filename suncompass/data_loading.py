
import csv
import random
from typing import Dict, Any

from pathlib import Path
from .preprocessing import Letterbox

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


def get_legacy_transforms(grayscale: bool = False, crop_size: int = DEFAULT_CENTRE_CROP_SIZE, resize_size: int = DEFAULT_RESIZED_IMAGE_SIZE):
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


def get_transforms(grayscale=False, crop_size=DEFAULT_CENTRE_CROP_SIZE,
                   resize_size=DEFAULT_RESIZED_IMAGE_SIZE, *, preprocessing="legacy", augment=True):
    """New RGB training/evaluation share letterboxing; only color augmentation differs."""
    if preprocessing == "legacy":
        return get_legacy_transforms(grayscale, crop_size, resize_size)
    if preprocessing != "letterbox" or grayscale:
        raise ValueError("Use RGB letterbox preprocessing, or explicitly select legacy")
    evaluation = Letterbox(resize_size)
    training = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(
            brightness=.1, contrast=.1, saturation=.1, hue=.05)], p=.8),
        Letterbox(resize_size),
    ]) if augment else Letterbox(resize_size)
    return training, evaluation


def load_image_targets_from_csv(csv_path: Path, header: bool = True) -> Dict[str, Any]:
    """Exactly image_path,sun_f,sun_l; paths are relative to the CSV directory.

    Absolute paths from historical exports remain supported. No extra columns,
    non-finite labels, duplicates or silently truncated targets are accepted.
    """
    csv_path = Path(csv_path).resolve()
    targets = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        if header:
            names = next(rows, None)
            if names not in (["image_path", "sun_f", "sun_l"], ["image_name", "sun_f", "sun_l"]):
                raise ValueError(f"{csv_path}: expected image_path,sun_f,sun_l header")
        for number, row in enumerate(rows, 2 if header else 1):
            if len(row) != 3 or not row[0].strip():
                raise ValueError(f"{csv_path}:{number}: expected one path and exactly two targets")
            path = Path(row[0])
            path = (csv_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
            values = np.array([float(row[1]), float(row[2])], dtype=np.float32)
            norm = float(np.linalg.norm(values))
            if not np.isfinite(values).all() or not 1e-8 < norm <= 1.00001:
                raise ValueError(f"{csv_path}:{number}: invalid forward/left unit-vector projection")
            if str(path) in targets:
                raise ValueError(f"{csv_path}:{number}: duplicate image path")
            targets[str(path)] = values
    if not targets:
        raise ValueError(f"{csv_path}: no targets")
    return targets


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class RegressionImageFolder(datasets.ImageFolder):
    """
    The regression image folder is a subclass of the ImageFolder class and is designed for 
    image regression tasks rather than image classification tasks. It takes in a dictionary
    that maps image paths to their target values.
    """
    def __init__(
        self, root: str, image_targets: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        image_targets = {str(Path(path).resolve()): value for path, value in image_targets.items()}
        super().__init__(root, *args, **kwargs)
        samples = []
        for path, _ in self.imgs:
            key = str(Path(path).resolve())
            if key not in image_targets:
                raise ValueError(f"Image has no target: {path}")
            samples.append((path, image_targets[key]))
        available = {str(Path(path).resolve()) for path, _ in self.imgs}
        missing = set(image_targets) - available
        if missing:
            raise ValueError(f"Targets have no matching image: {sorted(missing)[:3]}")
        self.samples = self.imgs = samples
        self.targets = [value for _, value in samples]

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
        *, batch_size=32, num_workers=4, seed=0, augment=True, preprocessing="legacy",
    ) -> None:
        self.grayscale = grayscale
        self.image_folder_path = Path(image_folder_path).resolve()
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.preprocessing = preprocessing
        if batch_size < 1 or num_workers < 0:
            raise ValueError("Invalid batch size or worker count")
        self.train_transforms, self.test_transforms = get_transforms(grayscale, crop_size, resize_size, preprocessing=preprocessing, augment=augment)
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
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=randomise, num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed), worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(), **({"prefetch_factor": 1} if self.num_workers else {}))
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
        return self.make_dataloader(self.image_folder_path / 'test', test_targets, randomise=False, transform=self.test_transforms)

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
