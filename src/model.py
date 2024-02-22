"""
This file contains the model definition for the simple image classifier.
"""

import torch
from torch import nn


class SimpleImageClassifier(nn.Module):
    """
    Simple image classifier model
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize the model

        Parameters
        ----------
        num_classes : int, optional
            Number of classes, by default 2
        """
        assert num_classes > 0, "Number of classes must be greater than 0"
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.ReLU(),  # Use ReLU activation function
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Add max pooling for downsampling
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),  # Add dropout layer to prevent overfitting
            nn.Linear(in_features=512 * 4 * 4, out_features=num_classes),
        )

        # Initialize the weights of the model
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x : [batch_size, 3, 224, 224]
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, x : [batch_size, 3, 224, 224] (3: RGB channels, 224: image size)

        Returns
        -------
        torch.Tensor
            Output tensor, x : [batch_size, 2]
        """
        x = self.conv_layer_1(x)  # x : [batch_size, 64, 56, 56]
        x = self.conv_layer_2(x)  # x : [batch_size, 128, 28, 28]
        x = self.conv_layer_3(x)  # x : [batch_size, 256, 14, 14]
        x = self.conv_layer_4(x)  # x : [batch_size, 512, 7, 7]
        x = self.conv_layer_5(x)  # x : [batch_size, 512, 4, 4]
        x = self.classifier(x)  # x : [batch_size, 512*4*4] -> [batch_size, 2]
        return x

    def _init_weights(self, module: nn.Module):
        """
        Initialize the weights of the model

        Parameters
        ----------
        module : nn.Module
            Module of the model
        """
        # Initialize the weights of the convolutional layers
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        # Initialize the weights of the batch normalization layers
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        # Initialize the weights of the linear layers
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
