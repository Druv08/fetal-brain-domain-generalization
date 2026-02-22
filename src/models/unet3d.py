"""
3D U-Net Model for Volumetric Segmentation.

This module implements a clean, readable 3D U-Net architecture
for multi-class segmentation of medical images.

Architecture:
- Encoder: Repeated (Conv3D -> BN -> ReLU) blocks with max pooling
- Bottleneck: Central processing block
- Decoder: Transposed convolutions with skip connections
- Output: 1x1x1 convolution for class predictions

Author: Research Team
Date: 2026
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Basic 3D convolutional block with BatchNorm and ReLU.
    
    Architecture:
        Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        padding: Padding size (default: 1)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock3D(nn.Module):
    """
    Encoder block: ConvBlock followed by MaxPool for downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            pooled: Downsampled feature map
            skip: Feature map before pooling (for skip connection)
        """
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock3D(nn.Module):
    """
    Decoder block: Upsample, concatenate skip connection, ConvBlock.
    
    Args:
        in_channels: Number of input channels (from previous layer)
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=2, stride=2
        )
        
        # After concatenation: in_channels + skip_channels
        self.conv_block = ConvBlock3D(
            in_channels + skip_channels,
            out_channels,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
            
        Returns:
            Upsampled and processed feature map
        """
        # Upsample
        x = self.upsample(x)
        
        # Handle size mismatch (due to odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv_block(x)
        
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for Volumetric Segmentation.
    
    A fully convolutional network with encoder-decoder architecture
    and skip connections. Designed for multi-class segmentation of
    3D medical images.
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        out_channels: Number of output classes
        base_features: Number of features in the first encoder layer (default: 32)
        num_levels: Number of encoder/decoder levels (default: 4)
        dropout: Dropout probability (default: 0.1)
        
    Input:
        Tensor of shape (B, C, D, H, W)
        
    Output:
        Tensor of shape (B, num_classes, D, H, W) with logits
        
    Example:
        >>> model = UNet3D(in_channels=1, out_channels=8)
        >>> x = torch.randn(1, 1, 128, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([1, 8, 128, 128, 128])
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        base_features: int = 32,
        num_levels: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.num_levels = num_levels
        
        # Calculate feature sizes for each level
        # e.g., base=32, levels=4: [32, 64, 128, 256]
        self.feature_sizes = [base_features * (2 ** i) for i in range(num_levels)]
        
        # Build encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        
        for i, features in enumerate(self.feature_sizes):
            encoder = EncoderBlock3D(prev_channels, features, dropout=dropout)
            self.encoders.append(encoder)
            prev_channels = features
        
        # Bottleneck
        bottleneck_features = self.feature_sizes[-1] * 2  # e.g., 512
        self.bottleneck = ConvBlock3D(
            self.feature_sizes[-1],
            bottleneck_features,
            dropout=dropout
        )
        
        # Build decoder
        self.decoders = nn.ModuleList()
        prev_channels = bottleneck_features
        
        for i in range(num_levels - 1, -1, -1):
            skip_channels = self.feature_sizes[i]
            out_features = self.feature_sizes[i]
            
            decoder = DecoderBlock3D(
                prev_channels,
                skip_channels,
                out_features,
                dropout=dropout
            )
            self.decoders.append(decoder)
            prev_channels = out_features
        
        # Final 1x1x1 convolution for class predictions
        self.final_conv = nn.Conv3d(
            self.feature_sizes[0],
            out_channels,
            kernel_size=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path (reverse order of skip connections)
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)
        
        # Final classification layer
        x = self.final_conv(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with softmax probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (B, num_classes, D, H, W)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions and return class indices.
        
        Args:
            x: Input tensor
            
        Returns:
            Class predictions of shape (B, D, H, W)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def build_unet3d(config: dict) -> UNet3D:
    """
    Build a 3D U-Net from configuration dictionary.
    
    Supports two config formats:
    1. Legacy: base_features + num_levels (auto-generates feature sizes)
    2. New: features list (explicit feature sizes per level)
    
    Args:
        config: Configuration dictionary with 'model' key
        
    Returns:
        Configured UNet3D model
        
    Example:
        >>> config = {
        ...     'model': {
        ...         'in_channels': 1,
        ...         'out_channels': 7,
        ...         'features': [32, 64, 128, 256],
        ...         'dropout': 0.1
        ...     }
        ... }
        >>> model = build_unet3d(config)
    """
    model_config = config.get('model', {})
    
    # Get in/out channels
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', model_config.get('num_classes', 8))
    dropout = model_config.get('dropout', 0.1)
    
    # Check for explicit features list (new format)
    if 'features' in model_config:
        features = model_config['features']
        num_levels = len(features)
        base_features = features[0] if features else 32
    else:
        # Legacy format
        base_features = model_config.get('base_features', 32)
        num_levels = model_config.get('num_levels', 4)
    
    return UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_features=base_features,
        num_levels=num_levels,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the model
    print("Testing 3D U-Net model...")
    
    # Create model
    model = UNet3D(
        in_channels=1,
        out_channels=8,
        base_features=32,
        num_levels=4,
        dropout=0.1
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with different input size
    x2 = torch.randn(1, 1, 128, 128, 128)
    with torch.no_grad():
        output2 = model(x2)
    print(f"\nInput shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")
    
    # Test prediction methods
    with torch.no_grad():
        probs = model.predict(x)
        classes = model.predict_classes(x)
    
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Classes shape: {classes.shape}")
    print(f"Unique classes: {torch.unique(classes).tolist()}")
    
    print("\nAll tests passed!")
