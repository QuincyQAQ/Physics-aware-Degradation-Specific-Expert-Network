import torch
import torch.nn as nn

try:
    from .arch_util import LayerNorm2d
except:
    from arch_util import LayerNorm2d

# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):
        x = inp           # size [B, C, H, W]

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sg(x)    # size [B, C, H, W]
        x = x * self.sca(x) # size [B, C, H, W]
        x = self.conv3(x) # size [B, C, H, W]

        x = self.dropout1(x)

        y = inp + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        x = self.dropout2(x)

        x = y + x * self.gamma


        return x 

class EfficientClassificationHead(nn.Module):
    
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.conv_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),  # Channel reduction
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2))
        
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid())
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv_bottleneck(x)
        attention_mask = self.attention(x)
        x = x * attention_mask  # Spatial attention
        return self.classifier(x)
