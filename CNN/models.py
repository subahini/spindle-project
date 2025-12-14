"""
Advanced CNN Architectures for EEG Spindle Detection
Optimized for severe class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance

    Explanation:
    - Reduces loss for well-classified examples (easy examples)
    - Focuses training on hard-to-classify examples
    - Alpha balances positive/negative classes
    - Gamma controls how much to down-weight easy examples
    """

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
    """
    Improved 2D CNN for EEG spindle detection
    Optimized for class imbalance with better regularization
    """

    def __init__(self, n_channels=16, dropout_rate=0.4):
        super().__init__()

        # First convolutional block - extract basic patterns
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.1)
        )

        # Second convolutional block - extract complex patterns
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1)
        )

        # Third convolutional block - high-level features
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
