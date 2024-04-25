import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingFeatureMap(nn.Module):

    def __init__(self, kernel_size=3):
        super(AutoEncoder_2D, self).__init__()
        # input size = (batch_size, 1, 2, len_sample=11000)

        # encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)

        # decoder
        self.inv_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, padding=1)
        self.up_bn1 = torch.nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.inv_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=1)
        self.up_bn2 = torch.nn.BatchNorm2d(64)
        self.up2 = nn.Upsample(scale_factor=2)
        self.inv_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=1)
        self.up_bn2 = torch.nn.BatchNorm2d(32)
        self.up3 = nn.Upsample(scale_factor=2)
        self.inv_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, padding=1)

    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return x

    def decoder(self, x):
        x = self.up1(F.relu(self.up_bn1(self.inv_conv1(x))))
        x = self.up2(F.relu(self.up_bn2(self.inv_conv2(x))))
        x = self.up3(F.relu(self.up_bn3(self.inv_conv3(x))))
        x = self.inv_conv4(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
