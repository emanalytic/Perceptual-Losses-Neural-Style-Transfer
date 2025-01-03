import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        ### ---- Initial convolution layers --- ###
        self.conv_pad = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(32, affine=True)

        ### --- Strided convolutions with Instance Normalization --- ###
        self.conv2_pad = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

        self.conv3_pad = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        ### --- Upsampling --- ###
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)

        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(self.conv_pad(x))))
        x = self.relu(self.in2(self.conv2(self.conv2_pad(x))))
        x = self.relu(self.in3(self.conv3(self.conv3_pad(x))))
        x = self.res_blocks(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = torch.tanh(self.conv4(x))
        return x
