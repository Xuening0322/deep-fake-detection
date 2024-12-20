import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 512,
        output_time_steps: int = 4
    ):
        super().__init__()
        
        # Modified convolutional layers for shorter sequences
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, 512, kernel_size=3, stride=2, padding=1),  # Layer 1
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),         # Layer 2
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),         # Layer 3
        ])
        
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(512, embedding_dim)
        
        # Adaptive pooling to ensure exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_time_steps)
        
    def forward(self, x):
        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)
            x = F.gelu(x)
            
        # Ensure exact temporal dimension
        x = self.adaptive_pool(x)  # [batch, channels, output_time_steps]
            
        # Apply layer norm
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.layer_norm(x)
        
        # Project to embedding dimension
        x = self.projection(x)
        x = self.dropout(x)
        
        # Return to channel-first format
        x = x.transpose(1, 2)  # [batch, channels, time]
        
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        output_time_steps: int = 4
    ):
        super().__init__()
        
        # First process mel spectrogram with frequency convolutions
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 512, kernel_size=(3,3), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        
        # Modified wav2vec style temporal processing
        self.feature_encoder = Wav2Vec2FeatureEncoder(
            in_channels=512,
            embedding_dim=embedding_dim,
            output_time_steps=output_time_steps
        )
        
    def forward(self, x):
        # Input x shape: [batch, channels, freq, time]
        
        # Process frequency dimension
        x = self.freq_conv(x)  # [batch, 512, freq//8, time]
        
        # Combine frequency dims by averaging
        x = x.mean(dim=2)  # [batch, 512, time]
        
        # Apply wav2vec feature encoder
        x = self.feature_encoder(x)
        
        return x


if __name__ == "__main__":
    # Test with spectrogram input
    model = AudioEncoder()
    x = torch.randn(32, 1, 80, 14)  # [batch, channel, freq_bins, time_steps]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")