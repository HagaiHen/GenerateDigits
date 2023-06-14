from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from config import hidden_dim_1, hidden_dim_2, hidden_dim_3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the layers of the generator model
        self.model = nn.Sequential(
            nn.Linear(100, hidden_dim_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_3, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        # Forward pass through the generator model
        output = self.model(x)
        
        # Reshape the output to match the image size (28x28) with a single channel
        output = output.view(x.size(0), 1, 28, 28)
        
        return output