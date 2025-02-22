import torch
from torch import nn


def EncoderConv2d(in_channels, out_channels):

    return nn.Sequential(
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(2, 2)),
                nn.PReLU()
            )


def ResBlock():
    return nn.Sequential(
                nn.ZeroPad2d((1, 1, 1, 1)),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
                nn.PReLU(),

                nn.ZeroPad2d((1, 1, 1, 1)),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            )


class CAE_32(nn.Module):

    def __init__(self, in_channels):
        super(CAE_32, self).__init__()
    # Encoder
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
            nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )
    # Decoder
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
            nn.PReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.PixelShuffle(1),
            nn.Tanh()
        )

    def forward(self, x):
        self.encode(x)
        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec

    def encode(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)


class CAE_16(nn.Module):

    def __init__(self, in_channels):
        super(CAE_16, self).__init__()

        self.encoded = None

    # ENCODER
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
        # 16x16x16
        self.e_conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=(5, 5), stride=(2, 2)),
            nn.Tanh()
        )

    # DECODER
        # 128x32x32
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )
        # 128x32x32
        self.d_block_1 = ResBlock()
        # 128x32x32
        self.d_block_2 = ResBlock()
        # 128x32x32
        self.d_block_3 = ResBlock()
        # 256x64x64
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.PixelShuffle(2)
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
            nn.PixelShuffle(1),
            nn.Tanh()
        )

    def forward(self, x):
        self.encode(x)
        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec

    def encode(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
