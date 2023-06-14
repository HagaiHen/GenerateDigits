from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from config import hidden_dim_1, hidden_dim_2, hidden_dim_3

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the layers of the discriminator model
        self.model = nn.Sequential(
            nn.Linear(784, hidden_dim_3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_3, hidden_dim_2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape the input image tensor to a flattened vector
        x = x.view(x.size(0), 784)
        
        # Forward pass through the discriminator model
        output = self.model(x)
        
        return output