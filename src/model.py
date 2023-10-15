import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 3)-> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
        
class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,  n_classes: int, p: float)-> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=p),
            nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, n_classes), 
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
                 
        channels = [3,]+[2**(4+i) for i in range(7)]
        
        self.model = nn.Sequential()
        for i in range(7):
                 self.model.add_module(f"ConvBlock_{i}",
                                       ConvBlock(channels[i],channels[i+1])
                                      )
        self.model.add_module("Head",
                              Head(channels[-1],channels[-2],num_classes,dropout)
                             )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
