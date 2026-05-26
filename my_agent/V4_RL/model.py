import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(),

            nn.Linear(256, 6)

        )

    def forward(self, x):

        x = self.cnn(x)

        x = self.fc(x)

        return x