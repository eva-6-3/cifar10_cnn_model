import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, 
        kernel_size=1, stride=1, 
        padding=0, dilation=1, 
        bias=False
    ):
        super().__init__()
        self.sep_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 
                kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.sep_conv(x)
        return x

    
class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        k=3, s=1, p=1, 
        dilation=1, groups=1, bias=False,
        dropout_value=dropout_value,
        regularizers=True,
    ):
        super().__init__()
        layers = []
        layers.extend([
            nn.Conv2d(
                in_channels, out_channels, 
                k, s, p, 
                dilation, groups, bias,
            )
        ])
        if regularizers:
            layers.extend([
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_value),
            ])
        self.custom_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.custom_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 BLOCK
        self.convblock_0 = ConvBNAct(3, 16)
        self.convblock_1 = ConvBNAct(16, 32)
        self.convblock_2 = ConvBNAct(32, 32)
        self.dilated_conv_1 = ConvBNAct(32, 32, k=3, s=2, dilation=2)
        
        # C2 BLOCK
        self.convblock_3 = ConvBNAct(32, 32)
        self.convblock_4 = ConvBNAct(32, 52)
        self.dilated_conv_2 = ConvBNAct(52, 64, k=3, s=2, dilation=2)
        
        # C3 BLOCK
        self.sep_conv_1 = ConvBNAct(64, 64)
        self.convblock_7 = ConvBNAct(64, 64)
        self.strided_conv_1 = ConvBNAct(64, 64, k=1, s=2)
        
        # C4 BLOCK
        self.convblock_5 = ConvBNAct(64, 64)
        self.convblock_6 = ConvBNAct(64, 10, regularizers=False)
        
        # OUTPUT BLOCK
        self.gap = nn.AvgPool2d(kernel_size=5)
    
    def forward(self, x):
        x = self.convblock_0(x)
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = self.dilated_conv_1(x)

        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.dilated_conv_2(x)
        
        x = self.sep_conv_1(x)
        x = self.convblock_7(x)
        x = self.strided_conv_1(x)
        
        x = self.convblock_5(x)
        x = self.convblock_6(x)

        x = self.gap(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
