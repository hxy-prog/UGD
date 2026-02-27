import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation=nn.LeakyReLU(0.01, inplace=True)):
        super(Bottleneck, self).__init__()
        width = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = activation
        
        if downsample is None and (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(3, 256)
        self.encoder = nn.Sequential(
            Bottleneck(256, 128),
            Bottleneck(128, 32),
            Bottleneck(32, 4, stride=1),
            nn.AdaptiveAvgPool2d((32, 32)) # 缩减尺寸到 128x128
        )
        self.fc_mu = nn.Conv2d(4, 4, kernel_size=1)  # 1x1 卷积
        self.fc_log_var = nn.Conv2d(4, 4, kernel_size=1)  # 1x1 卷积
    def forward(self,x):
        feat=self.encoder(self.embedding(x).squeeze(1).permute(0, 3, 1, 2))
        u=self.fc_mu(feat)
        s=self.fc_log_var(feat)
        return u,s
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 128, kernel_size=4, stride=4),  # Upsample到128x128
            Bottleneck(128, 128),
            Bottleneck(128, 4),
            nn.Conv2d(4, 2, kernel_size=1)  # 调整通道到2
        )
    def forward(self,x):
        return self.decoder(x)
class EmbeddingEncoderDecoder(nn.Module):
    def __init__(self):
        super(EmbeddingEncoderDecoder, self).__init__()
        self.noise_std=1*10
        self.bit_scale = 0.01
        self.encoder = Encoder()
        self.decoder =Decoder()

    def forward(self, x):
        u,std= self.encoder(x)
        std = torch.exp(0.5 * std)
        eps = torch.randn_like(std)
        z = u + eps * std
        # x = (torch.sigmoid(x) * 2 - 1) * self.bit_scale
        
        # np.savetxt('xt.txt',z[0][1].cpu().numpy())
        
        # z=torch.randn_like(z).to(x.device)
        
        x = self.decoder(z)
        return x,u,std
