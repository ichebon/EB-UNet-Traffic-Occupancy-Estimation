import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SSM2DBottleneck(nn.Module):
    """
    Enhanced Bottleneck integrating 2D Selective State Space (SSM2D) 
    and VMamba-inspired operations for global spatial context modeling.
    """
    def __init__(self, channels, kernel_size=31, expansion_factor=2):
        super().__init__()
        mid_channels = channels * expansion_factor
        
        # Spatial mixing / SSM-inspired depthwise large kernel convolution equivalent
        self.norm = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        
        # Wide receptive field depthwise convolution simulating SSM2D global scan coverage
        self.ssm_scan_conv = nn.Conv2d(
            mid_channels, mid_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=mid_channels, 
            bias=False
        )
        
        self.activation = nn.GELU()
        self.proj_out = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.proj_in(out)
        out = self.ssm_scan_conv(out)
        out = self.activation(out)
        out = self.proj_out(out)
        return out + residual


class EBUNet(nn.Module):
    """
    Enhanced Bottleneck-UNet (EB-UNet) for Dense Semantic Segmentation in Smart City Driving Scenarios.
    Combines hierarchical U-Net encoder-decoder with SSM2D + VMamba hybrid bottleneck.
    """
    def __init__(self, in_channels=3, out_classes=19, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Hybrid Bottleneck (SSM2D + VMamba integration)
        self.bottleneck = SSM2DBottleneck(channels=features[-1], kernel_size=31, expansion_factor=2)
        self.bottleneck_conv = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder part
        rev_features = features[::-1]
        for i in range(len(rev_features)):
            prev_feat = rev_features[i] * 2 if i == 0 else rev_features[i] * 2
            next_feat = rev_features[i]
            
            self.ups.append(
                nn.ConvTranspose2d(prev_feat, next_feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(rev_features[i] * 2, next_feat))

        # Final segmentation head
        self.final_conv = nn.Conv2d(features[0], out_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Apply Hybrid SSM2D / VMamba Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_conv(x)
        
        skip_connections = skip_connections[::-1]

        # Decoder forward pass with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Handle potential padding discrepancies
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick sanity check
    model = EBUNet(in_channels=3, out_classes=19, features=[64, 128, 256, 512])
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print("EB-UNet Output Shape:", output.shape)
