import torch
import torch.nn as nn
import torch.nn.functional as F 

import torchvision 
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
    """
    A basic residual block for a neural net

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels): 
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)  # Add the input (residual) to the output and apply ReLU
        

class TinyResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)


    def forward(self, x):
        out = self.initial(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        return self.fc(out)
    


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5, ))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


    model = TinyResNet()
    x = torch.randn(1, 3, 32, 32)  # 1 image, 3 channels, 32 x 32 pixels
    output = model(x)
    print(output.shape)  # Should be [1, num_classes]