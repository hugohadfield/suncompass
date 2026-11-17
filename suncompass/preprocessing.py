"""Shared geometry/normalization contract for new training and inference."""
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

IMAGE_SIZE = 1024
PADDING_RGB = (124, 116, 104)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def letterbox_image(image, size=IMAGE_SIZE):
    """Fit all decoded pixels on a square canvas; odd padding goes right/bottom.

    Input is PIL, RGB uint8 HWC numpy, or unnormalized CHW torch in [0,1].
    No EXIF rotation, cropping, flipping, or solar-label transformation occurs.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("Letterbox size must be a positive integer")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB uint8 HWC image")
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected unnormalized RGB CHW tensor")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Raw image tensors must be finite and in [0,1]; pass preprocessed=True for normalized inputs")
        image = to_pil_image(image.detach().cpu())
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image, RGB numpy array or CHW tensor")
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Empty image")
    scale = min(size / width, size / height)
    resized = (max(1, min(size, round(width * scale))),
               max(1, min(size, round(height * scale))))
    canvas = Image.new("RGB", (size, size), PADDING_RGB)
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR),
                 ((size - resized[0]) // 2, (size - resized[1]) // 2))
    return canvas


class Letterbox:
    """Picklable DataLoader transform: RGB -> normalized float32 CHW tensor."""
    def __init__(self, size=IMAGE_SIZE):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("Letterbox size must be a positive integer")
        self.size = size

    def __call__(self, image):
        canvas = letterbox_image(image, self.size)
        tensor = torch.from_numpy(np.array(canvas)).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        return (tensor - mean) / std


def contract(size=IMAGE_SIZE):
    return dict(method="letterbox", size=[size, size], padding_rgb=list(PADDING_RGB),
                interpolation="Pillow bilinear", mean=list(MEAN), std=list(STD),
                color="RGB", target_order=["sun_f", "sun_l"], exif_transpose=False)
