import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(
        self,
        dropout_value=0.05,
        BN=True,
        LN=False,
        GN=False, GN_groups=2,
    ):
        super().__init__()
        # Regularizers
        self.BN = BN
        self.LN = LN
        self.GN = GN
        self.GN_groups = GN_groups
        self.dropout_value = dropout_value
        
        
        # C1C2C3C40
        # C1 BLOCK
        self.convblock_1 = self.build_conv_block(3, 32)
        self.convblock_2 = self.build_conv_block(32, 64)
        self.strided_conv_1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=2, padding=1),
            nn.ReLU()
        )
        
        # C2 BLOCK
        self.convblock_3 = self.build_conv_block(32, 32)
        self.convblock_4 = self.build_conv_block(32, 64)
        self.strided_conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=2, padding=0),
            nn.ReLU()
        )
        
        # C3 BLOCK
        self.convblock_5 = self.build_conv_block(32, 32)
        self.convblock_6 = self.build_conv_block(32, 64)
        self.strided_conv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=2, padding=1),
            nn.ReLU()
        )
        
        # C4 BLOCK
        self.convblock_7 = self.build_conv_block(32, 32)
        self.convblock_8 = self.build_conv_block(32, 10)
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
    
    def build_conv_block(
        self,
        in_channel, out_channel,
        kernel_size=(3, 3),
        padding=1,
    ):
        elements = []
        conv_layer = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )
        activation_layer = nn.ReLU()
        elements.extend([conv_layer, activation_layer])
        
        regularizers = []
        if self.dropout_value:
            regularizers.append(nn.Dropout(self.dropout_value))
        if self.BN:
            regularizers.append(nn.BatchNorm2d(out_channel))
        if self.LN:
            regularizers.append(nn.GroupNorm(1, out_channel))
        if self.GN:
            regularizers.append(nn.GroupNorm(self.GN_groups, out_channel))
        elements.extend(regularizers)
        
        return nn.Sequential(*elements)

    def forward(self, x):
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        x = self.strided_conv_1(x)

        x = self.convblock_3(x)
        x = self.convblock_4(x)
        x = self.strided_conv_2(x)
        
        x = self.convblock_5(x)
        x = self.convblock_6(x)
        x = self.strided_conv_3(x)
        
        x = self.convblock_7(x)
        x = self.convblock_8(x)

        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)
    


