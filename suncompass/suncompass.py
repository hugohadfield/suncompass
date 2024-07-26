from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .pretrained_resnet import ResNetRegression


class SunCompass:
    """
    This is the main class for the suncompass package. It loads the pretrained model and
    can be used to predict the direction of the sun from an image.
    """
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

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

    def set_eval(self, dropout: bool = True):
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
        outputs_np = outputs.detach().cpu().numpy()
        return outputs_np
    
    def apply_transforms(self, image: np.ndarray):
        image_torch = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        return image_torch
    
    def __call__(self, image):
        return self.predict(image)
    
    def __repr__(self):
        return f"SunCompass(model={self.model})"
    
    def __str__(self):
        return repr(self)
    