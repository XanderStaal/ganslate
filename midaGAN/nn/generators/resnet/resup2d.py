from torch import nn
from midaGAN.nn.utils import get_norm_layer_2d, is_bias_before_norm

# Config imports
from dataclasses import dataclass
from midaGAN import configs


@dataclass
class ResUp2DConfig(configs.base.BaseGeneratorConfig):
    name: str = 'ResUp2D'
    n_residual_blocks: int = 5


class ResUp2D(nn.Module):

    def __init__(self, in_channels, out_channels, norm_type, n_residual_blocks=5):
        super().__init__()

        norm_layer = get_norm_layer_2d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        # Initial convolution block
        model = []

        # Residual blocks
        in_features=256
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, norm_type)]

        self.encoder = nn.ModuleList(model)

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features,
                                   out_features,
                                   3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7, bias=use_bias), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_features, norm_type):
        super().__init__()
        norm_layer = get_norm_layer_2d(norm_type)
        use_bias = is_bias_before_norm(norm_type)

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, bias=use_bias),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, bias=use_bias),
            norm_layer(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
