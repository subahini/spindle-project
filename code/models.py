#sample output
"""
this file contains all the model used in the thesis .
- 1D models: input [B,C,T] -> logits [B,T]
- 2D models: input [B,1,C,T] -> logits [B,T]
"""

import torch
from torch import nn





# ----- CNN2d-------------------------------------------

class CNN2D_Sample(nn.Module):
    """
    2D CNN (expects [B,1,C,T]) → [B,T] by collapsing channel axis and keeping time.
    """
    def __init__(self, in_channels=1, width=32):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.pool_h_to_1 = nn.AdaptiveAvgPool2d((1, None))   # Make the "height" (C) dimension become 1, keep time (T) unchanged
        self.proj = nn.Conv2d(width, 1, kernel_size=1)  # -> [B,1,1,T]

    def forward(self, x):           # [B,1,C,T]
        z = self.feat(x)            # [B,W,C,T]
        z = self.pool_h_to_1(z)     # [B,W,1,T]
        z = self.proj(z)            # [B,1,1,T]
        return z.squeeze(2).squeeze(1)  # [B,T]




# ----- UNet1D ---------------------------------------------------

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, k, padding=p), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet1D_Sample(nn.Module):
    def __init__(self, in_channels=16, base=32):
        super().__init__()
        #encoder
        self.enc1 = ConvBlock1D(in_channels, base)     #[B, 32, 400]
        self.pool1 = nn.MaxPool1d(2)  # 400/2 =32, 200
        self.enc2 = ConvBlock1D(base, base*2)  #64 ,200
        self.pool2 = nn.MaxPool1d(2)  #64 100

        self.enc3 = ConvBlock1D(base*2, base*4)   #128 ,100
        #decoder
        self.up2 = nn.ConvTranspose1d(base*4, base*2, 2, stride=2)  #64 200
        self.dec2 = ConvBlock1D(base*4, base*2) # -> B, 64, 200
        self.up1 = nn.ConvTranspose1d(base*2, base, 2, stride=2) # -> [B, 32, 400]
        self.dec1 = ConvBlock1D(base*2, base) # B, 32, 400

        self.head = nn.Conv1d(base, 1, kernel_size=1)  # -> [B,1,400]

    def forward(self, x):  # [B,C,T]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2);  d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1).squeeze(1)  # [B,T]



# ----- TCN1D  ---------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x[:, :, :-self.n].contiguous() if self.n > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dil, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dil), Chomp1d(pad),
            nn.ReLU(inplace=True), nn.BatchNorm1d(n_out), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dil), Chomp1d(pad),
            nn.ReLU(inplace=True), nn.BatchNorm1d(n_out), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
    def forward(self, x):
        out = self.net(x); res = self.down(x); return torch.relu(out + res)

class TCN1D_Sample(nn.Module):
    def __init__(self, in_channels=16, channels=(32,64,64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []; ch = in_channels
        for i, out in enumerate(channels):
            dil = 2 ** i
            layers.append(TemporalBlock(ch, out, kernel_size, dil, dropout))
            ch = out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(ch, 1, kernel_size=1)
    def forward(self, x):           # [B,C,T]
        z = self.tcn(x)
        return self.head(z).squeeze(1)   # [B,T]
