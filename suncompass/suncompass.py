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
    def __init__(self):
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
        model_name = f"{this_dir}/3_224_224_resnet_baseline_5_all.pth"
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