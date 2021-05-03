import torch
import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DoubleDepthwiseConv(nn.Module):
    """Applies conv 3x3 twice

    [
        DepthwiseConv(in_channels, out_channels),
        LeakyReLU(0.2),
        DepthwiseConv(out_channels, out_channels),
        BatchNorm2d(out_channels),
        LeakyReLU(0.2)
    ]"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            DepthwiseConv(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            DepthwiseConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.layers(x)


class DownUNet(nn.Module):
    """Downsampling layer in UNet

    [
        MaxPool2d(2),
        DoubleDepthwiseConv()
    ]"""

    def __init__(self, in_channels, out_channels):
        """Initialize the layer

        Args:
            in_channels  - number of input channels
            out_channels - number of output channels"""
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDepthwiseConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        """Run the layer"""
        return self.down(x)

class UpUNet(nn.Module):
    """Upsampling layer in UNet

    [
        Upsample(mode="bilinear"),
        cat(x,x_mirror),
        DoubleDepthwiseConv(in_channels_0, in_channels_1, out_channels)
    ]"""

    def __init__(self, in_channels_0, in_channels_1, out_channels):
        """Initialize the layer

        Args:
            in_channels  - number of input channels
            out_channels - number of output channels"""
        super().__init__()
        self.conv = DoubleDepthwiseConv(in_channels_0+in_channels_1, out_channels)

    def forward(self, x, x_mirror):
        """Run the layer"""
        x = nn.functional.interpolate(x, size=[x_mirror.size(2), x_mirror.size(3)], mode='bilinear', align_corners=True)
        x = torch.cat([x, x_mirror], dim=1)
        x = self.conv(x)
        return x

class ConvUNet(nn.Module):
    """Implementation of a convolutional UNet

    Returns:
        Net(x)"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 first_layer_channels,
                 max_layer_channels,
                 depth,
                 initial_and_final_kernel_size=3,
                 use_dropout_depths=None):
        """Initialize the net

        Args:
            in_channels                   - number of input channels
            out_channels                  - number of output channels
            first_layer_channels          - number of channels after the first convolution
            max_layer_channels            - maximum number of output channels in a single layer
            depth                         - number of downsampling and upsampling
            initial_and_final_kernel_size - the kernel size of initial and final convolution
                                            choices = [1, 3]
            use_dropout_depths            - a list of integers representing depths,
                                            where dropout should be applied when upscaling"""

        super().__init__()
        if initial_and_final_kernel_size not in [1, 3]:
            raise ValueError(f"Expected 'initial_and_final_kernel_size' to be one of [1, 3].")
        if use_dropout_depths is None:
            use_dropout_depths = []
        
        for item in use_dropout_depths:
            if item > depth:
                raise ValueError(f"Invalid value in 'use_dropout_depths', value {item} is greater than max depth.")

        if initial_and_final_kernel_size == 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, first_layer_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2)
            )
        else:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, first_layer_channels, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2)
            )

        layer_in_sizes = [min(max_layer_channels, 2 ** i * first_layer_channels) for i in range(depth)]
        layer_in_sizes.append(layer_in_sizes[-1])

        down_stack = []
        for i in range(depth):
            down_stack.append(DownUNet(layer_in_sizes[i], layer_in_sizes[i + 1]))

        up_stack = [UpUNet(layer_in_sizes[-1], layer_in_sizes[-2], layer_in_sizes[-2])]
        for i in range(depth - 1, 1, -1):
            up_stack.append(UpUNet(layer_in_sizes[i], layer_in_sizes[i - 1], layer_in_sizes[i - 1]))


        self.down_stack = nn.ModuleList(down_stack)
        self.up_stack = nn.ModuleList(up_stack)

        if initial_and_final_kernel_size == 3:
            self.final = nn.Sequential(
                nn.Conv2d(2*first_layer_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(2*first_layer_channels, out_channels, kernel_size=1, stride=1)
            )

    def forward(self, x):
        """Run the net"""
        x_list = [self.initial(x)]

        for i in range(len(self.down_stack)):
            x_list.append(self.down_stack[i](x_list[i]))

        x_out = x_list[-1]
        for i in range(len(self.up_stack)):
            x_out = self.up_stack[i](x_out, x_list[-2 - i])
        return self.final(x_out)