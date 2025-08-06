""" this i s  a model file contain ...Model : CNN Architectures for EEG Spindle Detection"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate p_t (probability of true class)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine all components
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SpindleCNN(nn.Module):

    def __init__(self, n_channels=16, dropout_rate=0.4):
        super().__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1)
        )


        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1)
        )

        # Global average pooling - reduces overfitting
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
# second Model :
class UNet1D(nn.Module):
    def __init__(self, in_channels=16, out_channels=1, init_features=64):  #
        super(UNet1D, self).__init__()

        features = init_features
        self.encoder1 = UNet1D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = UNet1D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = UNet1D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = UNet1D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet1D._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet1D._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet1D._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = x.squeeze(1)                                  #                               error in size shape
        enc1 = self.encoder1(x)      # [B, 64, T]
        enc2 = self.encoder2(self.pool1(enc1))  # [B, 128, T/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [B, 256, T/4]
        enc4 = self.encoder4(self.pool3(enc3))  # [B, 512, T/8]

        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, 1024, T/16]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        #return self.conv(dec1) m    size error
        x = self.conv(dec1)  # shape: [B, 1, T]
        x = torch.mean(x, dim=2)  # global average pooling over time → shape: [B, 1]
        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
        )

