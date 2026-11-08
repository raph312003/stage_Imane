import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        instance_norm: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()

        norm1 = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d

        self.boubleConv2D = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
            norm1(out_channels),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1),
            norm1(out_channels),
            nn.LeakyReLU(0.2, inplace= True)
        )

    def forward(self, x):
        return self.boubleConv2D(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        current_filter: int = 32,
        instance_norm: str = False,
        kernel_size: int = 3,
        num_pool: int = 3
    ):
        super().__init__()

        self.num_pool = num_pool
        self.encoder_block = nn.ModuleList()
        self.maxpool = nn.ModuleList()
        

        for i in range(self.num_pool +1):
            self.encoder_block.append(
                DoubleConv(
                    in_channels,
                    current_filter,
                    instance_norm,
                    kernel_size
                )
            )

            in_channels = current_filter

            if i < self.num_pool:
                self.maxpool.append(nn.MaxPool2d(2))
                current_filter*=2
        
        self.decoder_block = nn.ModuleList()
        self.upsample = nn.ModuleList()
        decoder_channel = in_channels//2

        for i in range(self.num_pool):
            # print("in_channels",in_channels)
            # print("decoder_channel",decoder_channel)

            self.upsample.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=decoder_channel,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder_block.append(
                DoubleConv(
                    in_channels=decoder_channel * 2,
                    out_channels=decoder_channel,
                    kernel_size=kernel_size,
                )
            )
            in_channels = decoder_channel
            decoder_channel//=2


        self.finalConv = nn.Conv2d(in_channels,1,kernel_size//2)

    def forward(self,x):

        skip_connections = []

        for i in range(self.num_pool +1):
            x = self.encoder_block[i](x)

            if i < self.num_pool:
                skip_connections.append(x)
                x = self.maxpool[i](x)
                

        skip_connections = list(reversed(skip_connections))
        for i in range(self.num_pool):
            x = self.upsample[i](x)

            skip = skip_connections[i]
            x = torch.cat([x, skip], dim=1)

            x = self.decoder_block[i](x)
            

        return self.finalConv(x)

model = Unet()
y = model(torch.randn(4,3,64,64))
# print(y.shape)