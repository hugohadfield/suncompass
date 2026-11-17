from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from .draw_arrow import draw_all_on_image_fl
from .pretrained_resnet import ResNetRegression
from .data_loading import get_transforms, DEFAULT_CENTRE_CROP_SIZE


class SunCompass:
    """
    This is the main class for the suncompass package. It loads the pretrained model and
    can be used to predict the direction of the sun from an image.
    """
    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self._model = None
        self._transforms = None

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def transforms(self):
        if self._transforms is None:
            _, self._transforms = get_transforms(False, DEFAULT_CENTRE_CROP_SIZE, 224)
        return self._transforms

    def load_model(self):
        """
        Load the pretrained model.
        """
        print("SunCompass: Loading model...")
        this_dir = Path(__file__).parent
        model_name = self.weights_path or f"{this_dir}/3_224_224_resnet_baseline_5_all.pth"
        model = ResNetRegression.load(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'SunCompass:  model using device: {device}')
        model.to(device)
        self.device = device
        self._model = model

    def set_eval(self, dropout: bool = False):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        if dropout:
            self.enable_dropout()
        else:
            self.disable_dropout()

    def enable_dropout(self):
        """
        Enable dropout for uncertainty estimation.
        """
        def enable_dropout(mod: nn.Module):
            if isinstance(mod, nn.Dropout):
                mod.train()
        self.model.apply(enable_dropout)
    
    def disable_dropout(self):
        """
        Disable dropout for deterministic predictions.
        """
        def disable_dropout(mod: nn.Module):
            if isinstance(mod, nn.Dropout):
                mod.eval()
        self.model.apply(disable_dropout)

    def predict(self, image: np.ndarray):
        """
        Predict the direction of the sun from an image.
        """
        if self.model is None:
            self.load_model()
        image_transformed = self.apply_transforms(image)
        outputs = self.model(image_transformed.to(self.device))
        outputs_np = outputs.detach().cpu().numpy().flatten()
        return outputs_np
    
    def scan_predict(self, image: np.ndarray):
        """
        Predict the direction of the sun from an image.
        Scan over the image left to right and top to bottom.
        """
        sub_images = []
        for i in range(0, image.shape[1]-DEFAULT_CENTRE_CROP_SIZE, 20):
            for j in range(0, image.shape[0]-DEFAULT_CENTRE_CROP_SIZE, 20):
                sub_image = image[j:j+DEFAULT_CENTRE_CROP_SIZE, i:i+DEFAULT_CENTRE_CROP_SIZE]
                sub_images.append(sub_image)
        predictions = []
        for sub_image in sub_images:
            prediction = self.predict(sub_image)
            predictions.append(prediction)
        preds = np.array(predictions)

        ## Uncomment to show the sub images
        # import matplotlib.pyplot as plt
        # for i in range(0, len(sub_images)):
        #     plt.figure()
        #     plt.imshow(sub_images[i])
        #     plt.title(f"Prediction: {preds[i]}")
        #     plt.show()

        return np.median(preds, axis=0)
    
    def apply_transforms(self, image: np.ndarray):
        # first make it a PIL image
        image_pil = Image.fromarray(image)
        image_torch = self.transforms(image_pil)
        image_torch = image_torch.unsqueeze(0)
        return image_torch

    def apply_letterbox_transforms(self, image: np.ndarray):
        """
        Letterbox transform: fit entire RGB image into 224x224 with aspect ratio preserved,
        using bilinear resize, integer rounded dimensions clamped 1..224, centered padding
        with RGB (124,116,104), then convert to float32 tensor [0,1] and apply ImageNet normalization.
        """
        # Convert to PIL
        pil_img = Image.fromarray(image)

        # Compute target size preserving aspect ratio
        orig_w, orig_h = pil_img.size
        scale = min(224 / orig_w, 224 / orig_h)
        new_w = max(1, min(224, round(orig_w * scale)))
        new_h = max(1, min(224, round(orig_h * scale)))

        # Resize with bilinear
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create padded canvas
        padded = Image.new('RGB', (224, 224), (124, 116, 104))
        paste_x = (224 - new_w) // 2
        paste_y = (224 - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        # Convert to tensor [0,1] and normalize
        tensor = torch.from_numpy(np.array(padded)).float().permute(2, 0, 1) / 255.0

        # Apply ImageNet normalization exactly once
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor

    def letterbox_predict(self, image: np.ndarray):
        """
        Predict using full-field-of-view letterbox inference.
        """
        if self.model is None:
            self.load_model()

        # Apply letterbox transforms
        image_torch = self.apply_letterbox_transforms(image)

        # Run model with preprocessed=True to skip internal transforms
        outputs = self.model(image_torch.to(self.device), preprocessed=True)
        outputs_np = outputs.detach().cpu().numpy().flatten()
        return outputs_np

    def apply_single_max_transforms(self, image: np.ndarray):
        """
        Single Crop Max transform: largest centered square crop of side min(width,height),
        origin floor((width-side)/2), floor((height-side)/2); crop BEFORE resizing with Pillow bilinear to 224x224.
        Convert RGB uint8 to float32 tensor [0,1] and apply ImageNet normalization exactly once.
        """
        # Convert to PIL
        pil_img = Image.fromarray(image)

        # Compute largest centered square crop
        orig_w, orig_h = pil_img.size
        side = min(orig_w, orig_h)
        left = (orig_w - side) // 2
        top = (orig_h - side) // 2
        right = left + side
        bottom = top + side

        # Crop to centered square
        cropped = pil_img.crop((left, top, right, bottom))

        # Resize to 224x224 with bilinear
        resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to tensor [0,1] and normalize
        tensor = torch.from_numpy(np.array(resized)).float().permute(2, 0, 1) / 255.0

        # Apply ImageNet normalization exactly once
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor

    def single_max_predict(self, image: np.ndarray):
        """
        Predict using single-max inference: largest centered square crop, then resize to 224x224.
        """
        if self.model is None:
            self.load_model()

        # Apply single-max transforms
        image_torch = self.apply_single_max_transforms(image)

        # Run model with preprocessed=True to skip internal transforms
        outputs = self.model(image_torch.to(self.device), preprocessed=True)
        outputs_np = outputs.detach().cpu().numpy().flatten()
        return outputs_np

    def __call__(self, image):
        return self.predict(image)
    
    def __repr__(self):
        return f"SunCompass(model={self.model})"
    
    def __str__(self):
        return repr(self)
    
    def predict_and_draw(self, image: np.ndarray):
        """
        Predict the direction of the sun from an image and draw the sun compass on the image.
        """
        w = image.shape[1]
        h = image.shape[0]
        K = np.array(
            [
                [w, 0.0, w//2],
                [0.0, w, h//2],
                [0.0, 0.0, 1.0]
            ]
        )
        res = self.scan_predict(image)
        f_m = res[0]
        l_m = res[1]
        theta_rad = np.arctan2(l_m, f_m)
        img_with_suncompass = draw_all_on_image_fl(image, K, f_m, l_m)
        return img_with_suncompass, theta_rad
    
    def get_input_image(self, image: np.ndarray):
        """
        Get the input image to the model.
        """
        return self.apply_transforms(image).cpu().squeeze().permute(1, 2, 0).numpy()