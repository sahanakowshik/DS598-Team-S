
""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, global_length, bilinear=True, multiplier=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.global_length = global_length

        # downsampling
        self.inc = DoubleConv(n_channels, 64 * multiplier)
        self.down1 = Down(64 * multiplier, 128 * multiplier)
        self.down2 = Down(128 * multiplier, 256 * multiplier)
        factor = 2 if bilinear else 1
        self.down3 = Down(256 * multiplier, (512 * multiplier) // factor)

        # adding cbam during the downsampling of the unet
        self.cbam1 = CBAM(channel=64 * multiplier)
        self.cbam2 = CBAM(channel=128 * multiplier)
        self.cbam3 = CBAM(channel=256 * multiplier)
        
        # adding cbam during the upsampling of the unet
        self.cbam_up2 = CBAM(channel=(256 * multiplier) // factor)
        self.cbam_up3 = CBAM(channel=(128 * multiplier) // factor)
        self.cbam_up4 = CBAM(channel=(64 * multiplier))

        # upsampling
        self.up2 = Up((512 * multiplier)+global_length, (256 * multiplier) // factor, bilinear)
        self.up3 = Up(256 * multiplier, (128 * multiplier) // factor, bilinear)
        self.up4 = Up(128 * multiplier, 64 * multiplier, bilinear)
        self.outc = OutConv(64 * multiplier, n_classes)
        
        # dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout_decoder1 = nn.Dropout(0.3)
        self.dropout_decoder2 = nn.Dropout(0.3)
        self.dropout_decoder3 = nn.Dropout(0.3)


    def forward(self, x, x_global):
        x1 = self.inc(x)    #output : 32*32*64
        x1 = self.cbam1(x1) + x1
        x1 = self.dropout1(x1) 

        x2 = self.down1(x1)     #output : 16*16*128
        x2 = self.cbam2(x2) + x2
        x2 = self.dropout2(x2)

        x3 = self.down2(x2)     #output : 8*8*256
        x3 = self.cbam3(x3) + x3
        x3 = self.dropout3(x3)

        x4 = self.down3(x3)     #output : 4*4*512
        x4 = torch.cat([x4, x_global], dim=1)
        
        # Decoder with Cross Attention
        out = self.up2(x4, x3)
        out = self.cbam_up2(out) + out  # Apply CBAM after up-sampling
        out = self.dropout_decoder1(out)

        out = self.up3(out, x2)
        out = self.cbam_up3(out) + out  # Apply CBAM after up-sampling
        out = self.dropout_decoder2(out)

        out = self.up4(out, x1)
        out = self.cbam_up4(out) + out  # Apply CBAM after up-sampling
        out = self.dropout_decoder3(out)

        logits = self.outc(out)
        return logits
