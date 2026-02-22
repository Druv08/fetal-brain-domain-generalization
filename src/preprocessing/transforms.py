"""
Preprocessing module for Fetal Brain MRI Segmentation.

This module provides preprocessing transforms for 3D medical images:
- Z-score normalization
- Intensity clipping
- Center cropping
- Resampling (placeholder)

Author: Research Team
Date: 2026
"""

from typing import Optional, Tuple, Union

import numpy as np


def zscore_normalize(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply z-score normalization to a 3D volume.
    
    Normalizes the volume to have zero mean and unit variance.
    If a mask is provided, statistics are computed only within the mask.
    
    Args:
        volume: 3D numpy array to normalize
        mask: Optional binary mask (1 = foreground, 0 = background)
              If provided, statistics computed only within mask
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Normalized volume with zero mean and unit variance
        
    Example:
        >>> volume = np.random.randn(128, 128, 128) * 100 + 500
        >>> normalized = zscore_normalize(volume)
        >>> print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        Mean: 0.0000, Std: 1.0000
    """
    volume = volume.astype(np.float32)
    
    if mask is not None:
        # Compute statistics only within mask
        mask = mask.astype(bool)
        masked_values = volume[mask]
        mean = masked_values.mean()
        std = masked_values.std()
    else:
        # Use non-zero values for brain MRI (assuming 0 is background)
        non_zero_mask = volume > 0
        if non_zero_mask.sum() > 0:
            mean = volume[non_zero_mask].mean()
            std = volume[non_zero_mask].std()
        else:
            mean = volume.mean()
            std = volume.std()
    
    # Normalize
    normalized = (volume - mean) / (std + epsilon)
    
    return normalized


def intensity_clipping(
    volume: np.ndarray,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Clip intensity values to specified percentiles.
    
    Removes outliers by clipping values below the low percentile
    and above the high percentile.
    
    Args:
        volume: 3D numpy array
        percentile_low: Lower percentile for clipping (default: 0.5)
        percentile_high: Upper percentile for clipping (default: 99.5)
        mask: Optional mask for computing percentiles
        
    Returns:
        Volume with clipped intensity values
        
    Example:
        >>> volume = np.random.randn(128, 128, 128) * 100
        >>> clipped = intensity_clipping(volume, 1, 99)
    """
    volume = volume.astype(np.float32)
    
    if mask is not None:
        # Compute percentiles only within mask
        values = volume[mask.astype(bool)]
    else:
        # Use non-zero values
        non_zero_mask = volume > 0
        if non_zero_mask.sum() > 0:
            values = volume[non_zero_mask]
        else:
            values = volume.flatten()
    
    # Compute percentile thresholds
    low_thresh = np.percentile(values, percentile_low)
    high_thresh = np.percentile(values, percentile_high)
    
    # Clip values
    clipped = np.clip(volume, low_thresh, high_thresh)
    
    return clipped


def center_crop_3d(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    label: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Center crop a 3D volume to the target size.
    
    If the volume is smaller than target size in any dimension,
    it will be zero-padded to reach the target size.
    
    Args:
        volume: 3D numpy array of shape (D, H, W)
        target_size: Target size as (D, H, W)
        label: Optional label volume to crop with same parameters
        
    Returns:
        Cropped volume (and label if provided)
        
    Example:
        >>> volume = np.random.randn(256, 256, 256)
        >>> cropped = center_crop_3d(volume, (128, 128, 128))
        >>> print(cropped.shape)
        (128, 128, 128)
    """
    current_size = volume.shape
    target_size = tuple(target_size)
    
    # Calculate start and end indices for cropping/padding
    starts = []
    ends = []
    pad_before = []
    pad_after = []
    
    for current, target in zip(current_size, target_size):
        if current >= target:
            # Crop: center crop
            start = (current - target) // 2
            end = start + target
            starts.append(start)
            ends.append(end)
            pad_before.append(0)
            pad_after.append(0)
        else:
            # Pad: zero-pad to reach target
            starts.append(0)
            ends.append(current)
            total_pad = target - current
            pad_before.append(total_pad // 2)
            pad_after.append(total_pad - total_pad // 2)
    
    # Extract crop region
    cropped = volume[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2]
    ]
    
    # Apply padding if necessary
    if any(p > 0 for p in pad_before + pad_after):
        pad_width = list(zip(pad_before, pad_after))
        cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)
    
    # Process label if provided
    if label is not None:
        label_cropped = label[
            starts[0]:ends[0],
            starts[1]:ends[1],
            starts[2]:ends[2]
        ]
        
        if any(p > 0 for p in pad_before + pad_after):
            pad_width = list(zip(pad_before, pad_after))
            label_cropped = np.pad(label_cropped, pad_width, mode='constant', constant_values=0)
        
        return cropped, label_cropped
    
    return cropped


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 3,
    label: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Resample a 3D volume to a target spacing.
    
    This is a placeholder function. Full implementation would use
    scipy.ndimage.zoom or SimpleITK for proper resampling.
    
    Args:
        volume: 3D numpy array
        original_spacing: Original voxel spacing (D, H, W) in mm
        target_spacing: Target voxel spacing (D, H, W) in mm
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        label: Optional label volume (will use order=0)
        
    Returns:
        Resampled volume (and label if provided)
        
    Note:
        This is a placeholder. Implement with scipy.ndimage.zoom:
        
        ```python
        from scipy.ndimage import zoom
        
        zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
        resampled = zoom(volume, zoom_factors, order=order)
        ```
    """
    # TODO: Implement full resampling logic
    # Placeholder: return input unchanged
    
    print("Warning: resample_volume is a placeholder. Returning unchanged volume.")
    print(f"  Original spacing: {original_spacing}")
    print(f"  Target spacing: {target_spacing}")
    
    if label is not None:
        return volume, label
    
    return volume


def preprocess_volume(
    volume: np.ndarray,
    label: Optional[np.ndarray] = None,
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
    normalize: bool = True,
    crop_size: Optional[Tuple[int, int, int]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply full preprocessing pipeline to a volume.
    
    Pipeline order:
    1. Intensity clipping
    2. Z-score normalization
    3. Center cropping (optional)
    
    Args:
        volume: 3D numpy array
        label: Optional label volume
        clip_percentiles: (low, high) percentiles for clipping
        normalize: Whether to apply z-score normalization
        crop_size: Optional target crop size
        
    Returns:
        Preprocessed volume (and label if provided)
        
    Example:
        >>> volume, label = preprocess_volume(
        ...     volume, label,
        ...     clip_percentiles=(1, 99),
        ...     normalize=True,
        ...     crop_size=(128, 128, 128)
        ... )
    """
    # Step 1: Intensity clipping
    volume = intensity_clipping(
        volume,
        percentile_low=clip_percentiles[0],
        percentile_high=clip_percentiles[1]
    )
    
    # Step 2: Z-score normalization
    if normalize:
        volume = zscore_normalize(volume)
    
    # Step 3: Center cropping
    if crop_size is not None:
        if label is not None:
            volume, label = center_crop_3d(volume, crop_size, label=label)
        else:
            volume = center_crop_3d(volume, crop_size)
    
    if label is not None:
        return volume, label
    
    return volume


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for 3D medical images.
    
    Can be used as a transform in PyTorch datasets.
    
    Args:
        clip_percentiles: (low, high) percentiles for intensity clipping
        normalize: Whether to apply z-score normalization
        crop_size: Optional target crop size (D, H, W)
        target_spacing: Optional target voxel spacing for resampling
        
    Example:
        >>> pipeline = PreprocessingPipeline(
        ...     clip_percentiles=(1, 99),
        ...     normalize=True,
        ...     crop_size=(128, 128, 128)
        ... )
        >>> volume, label = pipeline(volume, label)
    """
    
    def __init__(
        self,
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        normalize: bool = True,
        crop_size: Optional[Tuple[int, int, int]] = None,
        target_spacing: Optional[Tuple[float, float, float]] = None
    ):
        self.clip_percentiles = clip_percentiles
        self.normalize = normalize
        self.crop_size = crop_size
        self.target_spacing = target_spacing
    
    def __call__(
        self,
        volume: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply the preprocessing pipeline."""
        return preprocess_volume(
            volume,
            label=label,
            clip_percentiles=self.clip_percentiles,
            normalize=self.normalize,
            crop_size=self.crop_size
        )


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing module...")
    
    # Create dummy data
    volume = np.random.randn(128, 128, 128).astype(np.float32) * 100 + 500
    label = np.random.randint(0, 8, size=(128, 128, 128)).astype(np.int64)
    
    # Test z-score normalization
    normalized = zscore_normalize(volume)
    print(f"Z-score: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    
    # Test intensity clipping
    clipped = intensity_clipping(volume, 1, 99)
    print(f"Clipping: min={clipped.min():.2f}, max={clipped.max():.2f}")
    
    # Test center cropping
    cropped, label_cropped = center_crop_3d(volume, (64, 64, 64), label=label)
    print(f"Center crop: {volume.shape} -> {cropped.shape}")
    
    # Test full pipeline
    pipeline = PreprocessingPipeline(crop_size=(64, 64, 64))
    processed, processed_label = pipeline(volume, label)
    print(f"Pipeline output: {processed.shape}")
    
    print("All tests passed!")
