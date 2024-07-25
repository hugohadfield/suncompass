import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data_loading import RegressionTaskData


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
    def __init__(self, output_size: int = 3, input_size: Tuple[int, int] = (224, 224), fine_tune: bool = True):
        super(ResNetRegression, self).__init__()
        # Load a pre-trained ResNet model
        weights=models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights)
        
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
        self.input_size = input_size
        
        # Define a resize transform
        self.preprocess_transform = weights.transforms()

    def preprocess(self, img):
        """
        Preprocess the input image tensor to the required format.
        """
        img = self.preprocess_transform(img)
        return img

    def forward(self, x):
        """
        Passes the data through the ResNet model.
        """
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
    def load(filename: str, output_size: int = 2, input_size: Tuple[int, int] = (224, 224), fine_tune: bool = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetRegression(output_size=output_size, input_size=input_size, fine_tune=fine_tune)
        model.load_state_dict(torch.load(filename, map_location=device))
        return model


def train_network(device, n_epochs: int = 10, image_size: Tuple[int, int] = (128, 128)):
    """
    This trains the network for a set number of epochs.
    """
    assert image_size[0] == image_size[1], 'Image size must be square'
    resize_size = image_size[0]
    regression_task = RegressionTaskData(grayscale=False, resize_size=resize_size)

    # Define the model, loss function, and optimizer
    model = ResNetRegression(output_size=2, input_size=image_size, fine_tune=True)
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Train the model
    writer = SummaryWriter()
    for epoch in range(n_epochs):
        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Loss', loss.item(), epoch * len(regression_task.trainloader) + i)

            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            output_angles = [calculate_angles_between_sun_vectors(out[0], out[1], t[0], t[1]) for out, t in zip(outputs_np, targets_np)]
            angle_error = np.sum(output_angles) / len(output_angles)

            # Print training statistics
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], Loss: {loss.item():.4f} Angle Error: {angle_error:.4f} degrees')
    writer.close()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_network(device, n_epochs=5)

    # Save the model
    image_size_load = (3, 224, 224)
    filename = f'{image_size_load[0]}_{image_size_load[1]}_{image_size_load[2]}_resnet.pth'
    model.save(filename)

    # Load the model
    model = ResNetRegression.load(filename)
    model.eval()
    print(model)
    print('Model loaded successfully')
