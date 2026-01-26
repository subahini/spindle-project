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


class SpindleCNN1D_RNN(nn.Module):
    """
    1D CNN + RNN for EEG spindle detection

    Architecture:
    - 1D CNN extracts local temporal features from each channel
    - RNN captures long-term temporal dependencies
    - Attention mechanism focuses on important time segments
    - Designed specifically for class imbalance
    """

    def __init__(self, n_channels=16, rnn_type='GRU', hidden_size=64, dropout_rate=0.5):
        super().__init__()

        # 1D CNN for each channel separately
        self.channel_conv = nn.ModuleList([
            nn.Sequential(
                # First conv layer - local patterns
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(0.2),

                # Second conv layer - complex patterns
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout1d(0.2),

                # Third conv layer - high-level features
                nn.Conv1d(32, 48, kernel_size=3, padding=1),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(20)  # Fixed output length
            ) for _ in range(n_channels)
        ])

        # RNN for temporal modeling
        rnn_input_size = 48 * n_channels  # Features from all channels

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, 1,
                               batch_first=True, dropout=0.3, bidirectional=True)
        else:  # GRU
            self.rnn = nn.GRU(rnn_input_size, hidden_size, 1,
                              batch_first=True, dropout=0.3, bidirectional=True)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.squeeze(1)  # Remove channel dim: (batch, channels, time)

        # Extract features from each channel
        channel_features = []
        for i, conv in enumerate(self.channel_conv):
            channel_data = x[:, i:i + 1, :]  # Single channel
            features = conv(channel_data)  # (batch, 48, 20)
            channel_features.append(features)

        # Combine all channel features
        combined_features = torch.cat(channel_features, dim=1)  # (batch, 48*16, 20)

        # Transpose for RNN: (batch, 20, 48*16)
        rnn_input = combined_features.transpose(1, 2)

        # RNN processing
        rnn_output, _ = self.rnn(rnn_input)  # (batch, 20, hidden_size*2)

        # Attention mechanism
        attention_weights = self.attention(rnn_output)  # (batch, 20, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention
        attended_output = torch.sum(rnn_output * attention_weights, dim=1)  # (batch, hidden_size*2)

        # Classification
        output = self.classifier(attended_output)
        return output


class TemporalBlock(nn.Module):
    """Temporal block for TCN with dilated convolutions"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.normal_(m.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove rightmost elements to ensure causality"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SpindleTCN(nn.Module):
    """
    Temporal Convolutional Network for EEG spindle detection

    Features:
    - Dilated convolutions capture multi-scale temporal patterns
    - Residual connections for better gradient flow
    - Causal convolutions (no future information)
    - Excellent for long sequences and class imbalance
    """

    def __init__(self, n_channels=16, num_channels=[64, 128, 256], kernel_size=3, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_conv = nn.Conv1d(n_channels, num_channels[0], kernel_size=1)

        # TCN layers with exponential dilation
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 1, 2, 4, 8, ...
            in_channels = num_channels[i - 1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]

            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.squeeze(1)  # (batch, channels, time)
        x = self.input_conv(x)
        x = self.network(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for Transformer"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(context)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SpindleTransformer(nn.Module):
    """
    EEG-Transformer for spindle detection

    Features:
    - Positional encoding for temporal information
    - Multi-head attention captures channel relationships
    - Deep transformer blocks for complex patterns
    - Global pooling for final classification
    """

    def __init__(self, n_channels=16, n_timepoints=400, d_model=128,
                 n_heads=8, n_layers=4, d_ff=512, dropout=0.2):
        super().__init__()

        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(n_timepoints, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.squeeze(1).transpose(1, 2)  # (batch, time, channels)

        # Input embedding
        x = self.input_embedding(x)

        # Add positional encoding
        seq_len = x.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        x = self.dropout(x)

        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        output = self.classifier(x)
        return output