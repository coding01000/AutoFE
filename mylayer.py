import torch
from torch import nn


class EncoderConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(2, 2)),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

    def forward(self, x):
        return self.layer(x)


class CAE_32(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
    #Encoder
        # 64x64x64
        self.e_conv_1 = EncoderConv2d(3, 64)
        # 128x32x32
        self.e_conv_2 = EncoderConv2d(64, 128)
        # 128x32x32
        self.e_block_1 = ResBlock()
        # 128x32x32
        self.e_block_2 = ResBlock()
        # 128x32x32
        self.e_block_3 = ResBlock()
        # in_channelsx32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )
    #Decoder
        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )
        # 128x64x64
        self.d_block_1 = ResBlock()
        # 128x64x64
        self.d_block_2 = ResBlock()
        # 128x64x64
        self.d_block_3 = ResBlock()
        # 32x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.PixelShuffle(2)
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layer(x)