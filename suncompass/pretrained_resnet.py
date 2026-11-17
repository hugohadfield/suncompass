import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .data_loading import RegressionTaskData
from .preprocessing import IMAGE_SIZE, Letterbox


def calculate_angles_between_sun_vectors(sun_f_a: float, sun_l_a: float, sun_f_b: float, sun_l_b: float) -> float:
    """
    This function calculates the angle between two sun vectors
    """
    norm_a = np.sqrt(sun_f_a*sun_f_a + sun_l_a*sun_l_a  + 1e-6) 
    norm_b = np.sqrt(sun_f_b*sun_f_b + sun_l_b*sun_l_b + 1e-6)
    dot_prod = sun_f_a*sun_f_b/(norm_a*norm_b) + sun_l_a*sun_l_b/(norm_a*norm_b)
    return np.rad2deg(np.arccos(dot_prod))


class ResNetRegression(nn.Module):
    """
    This class defines a regression model based on a pre-trained ResNet model.
    """
    def __init__(self, output_size: int = 3, input_size: Tuple[int, int] = (224, 224), fine_tune: bool = True, preprocessing="legacy"):
        super(ResNetRegression, self).__init__()
        # Load a pre-trained ResNet model
        weights=models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)
        
        # Freeze the layers except the last 3
        if fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer2.parameters():
                param.requires_grad = False
            for param in self.resnet.layer1.parameters():
                param.requires_grad = False

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Optionally, add a dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Replace the last fully connected layer to output the desired number of regression values
        num_ftrs = self.resnet.fc.out_features
        self.fc_end = nn.Linear(in_features=num_ftrs, out_features=output_size)

        # Store the input size for resizing
        if len(input_size) != 2 or input_size[0] != input_size[1] or input_size[0] < 1:
            raise ValueError("Expected positive square input_size")
        if preprocessing not in ("letterbox", "legacy"):
            raise ValueError("Unknown preprocessing mode")
        self.input_size = tuple(input_size)
        self.preprocessing = preprocessing
        
        # Define a resize transform
        self.preprocess_transform = weights.transforms() if preprocessing == "legacy" else Letterbox(input_size[0])

    def preprocess(self, img):
        """
        Preprocess the input image tensor to the required format.
        """
        img = self.preprocess_transform(img)
        return img

    def forward(self, x, preprocessed=False):
        """
        Passes the data through the ResNet model.
        If preprocessed=True, x is assumed to be already normalized and ready for inference.
        Otherwise, apply the standard preprocessing (resize and normalization).
        """
        if not preprocessed:
            # Resize the input image
            x = torch.stack([self.preprocess(img) for img in x])

        x = x.to(next(self.parameters()).device)  # Ensure the input is on the same device as the model parameters
        x = self.resnet(x)

        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc_end(x)
        x = nn.functional.sigmoid(x)
        x = 2.0*x - 1.0  # Scale the output to be between -1 and 1
        return x
    
    def save(self, filename: str):
        torch.save(self.state_dict(), filename)

    @staticmethod
    def load(filename: str, output_size: int = 2, input_size: Tuple[int, int] = (224, 224), fine_tune: bool = True, preprocessing="legacy"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetRegression(output_size=output_size, input_size=input_size, fine_tune=fine_tune, preprocessing=preprocessing)
        model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
        return model


def train_network(device, n_epochs=10, image_size=(IMAGE_SIZE, IMAGE_SIZE), *,
                  image_folder_path="example_dataset", batch_size=32, num_workers=4,
                  amp=True, seed=0, max_steps=None, weights=None, learning_rate=.001,
                  log_dir=None):
    """Train on shared, once-normalized letterboxes and forward/left targets."""
    import random
    device = torch.device(device)
    if n_epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("Epoch and step counts must be positive")
    if image_size[0] != image_size[1]:
        raise ValueError("Image size must be square")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    regression_task = RegressionTaskData(image_folder_path=image_folder_path,
        resize_size=image_size[0], batch_size=batch_size, num_workers=num_workers, seed=seed,
        preprocessing="letterbox")
    model = ResNetRegression(output_size=2, input_size=image_size, fine_tune=True, preprocessing="letterbox")
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(device).train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=use_amp)
    writer = SummaryWriter(log_dir=log_dir)
    steps = 0
    try:
        for epoch in range(n_epochs):
            for inputs, targets in regression_task.trainloader:
                if targets.ndim != 2 or targets.shape[1] != 2:
                    raise ValueError("Expected [N,2] sun_f,sun_l targets")
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    # Loader already letterboxed and normalized; never process twice.
                    outputs = model(inputs, preprocessed=True)
                    loss = nn.functional.mse_loss(outputs, targets)
                if outputs.shape != targets.shape or not torch.isfinite(loss):
                    raise RuntimeError("Invalid/non-finite training output or loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                writer.add_scalar("Train Loss", loss.item(), steps)
                if steps % 10 == 0 or steps == 1:
                    print(f"epoch={epoch+1} step={steps} loss={loss.item():.6f}", flush=True)
                if max_steps is not None and steps >= max_steps:
                    return model
    finally:
        writer.close()
    return model


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from .amos import verify_export
    from .preprocessing import contract
    parser = argparse.ArgumentParser(description="Train SunCompass on a verified AMOS export")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Optional initial SunCompass state dictionary")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Square letterbox size in pixels")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, help="Bounded pipeline smoke run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.image_size < 1:
        parser.error("--image-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Checkpoint output already exists; choose a new run path")
    receipt = verify_export(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = train_network(args.device, args.epochs, image_size=(args.image_size, args.image_size),
        image_folder_path=args.data,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=not args.no_amp,
        seed=args.seed, max_steps=args.max_steps, weights=args.weights,
        learning_rate=args.learning_rate, log_dir=str(args.output.parent / (args.output.stem + "-tensorboard")))
    model.save(str(args.output))
    metadata = dict(preprocessing=contract(args.image_size), target_order=["sun_f", "sun_l"],
        dataset_import_fingerprint=receipt["import_fingerprint"],
        checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        config={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
