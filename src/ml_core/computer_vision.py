"""
Computer Vision utility functions for image processing and deep learning.

This module provides helper functions for:
- Image loading and preprocessing
- Data augmentation with transforms.v2
- Model loading and inference
- Visualization utilities
- Detection and segmentation helpers
"""

from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision import datasets
from PIL import Image


# ==============================================================================
# Image Loading and Preprocessing
# ==============================================================================

def load_image(
    path: Union[str, Path],
    mode: str = 'RGB'
) -> Image.Image:
    """
    Load an image from disk.
    
    Args:
        path: Path to image file
        mode: Image mode ('RGB', 'L' for grayscale, 'RGBA')
    
    Returns:
        PIL Image
    
    Examples:
        >>> img = load_image('photo.jpg')
        >>> img = load_image('mask.png', mode='L')  # Grayscale
    """
    img = Image.open(path)
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def image_to_tensor(
    image: Image.Image,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert PIL image to tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [0, 1] range
    
    Returns:
        Tensor of shape (C, H, W)
    
    Examples:
        >>> tensor = image_to_tensor(pil_image)
        >>> tensor.shape
        torch.Size([3, 224, 224])
    """
    tensor = v2.functional.to_image(image)  # Returns uint8 tensor
    if normalize:
        tensor = v2.functional.to_dtype(tensor, dtype=torch.float32, scale=True)
    return tensor


def tensor_to_image(
    tensor: torch.Tensor
) -> Image.Image:
    """
    Convert tensor to PIL image.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (H, W)
    
    Returns:
        PIL Image
    
    Examples:
        >>> pil_img = tensor_to_image(tensor)
    """
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    
    if tensor.is_floating_point():
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    return Image.fromarray(tensor.cpu().numpy())


def get_imagenet_stats() -> Tuple[List[float], List[float]]:
    """
    Get ImageNet normalization statistics.
    
    Returns:
        Tuple of (mean, std) lists for RGB channels
    
    Examples:
        >>> mean, std = get_imagenet_stats()
        >>> transforms.Normalize(mean, std)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


# ==============================================================================
# Data Augmentation Transforms
# ==============================================================================

def get_train_transforms(
    size: int = 224,
    augmentation: str = 'standard'
) -> v2.Compose:
    """
    Get training transforms with data augmentation.
    
    Args:
        size: Output image size
        augmentation: Augmentation level ('none', 'standard', 'strong', 'autoaugment')
    
    Returns:
        Composed transforms
    
    Examples:
        >>> train_transform = get_train_transforms(224, 'strong')
        >>> augmented = train_transform(image)
    """
    mean, std = get_imagenet_stats()
    
    if augmentation == 'none':
        return v2.Compose([
            v2.Resize(size),
            v2.CenterCrop(size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
    
    elif augmentation == 'standard':
        return v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
    
    elif augmentation == 'strong':
        return v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
    
    elif augmentation == 'autoaugment':
        return v2.Compose([
            v2.RandomResizedCrop(size),
            v2.RandomHorizontalFlip(),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
    
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def get_val_transforms(size: int = 224) -> v2.Compose:
    """
    Get validation/inference transforms (no augmentation).
    
    Args:
        size: Output image size
    
    Returns:
        Composed transforms
    
    Examples:
        >>> val_transform = get_val_transforms(224)
        >>> normalized = val_transform(image)
    """
    mean, std = get_imagenet_stats()
    
    return v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


# ==============================================================================
# Model Utilities
# ==============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
    
    Returns:
        Number of parameters
    
    Examples:
        >>> total = count_parameters(model, trainable_only=False)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"{trainable/1e6:.1f}M trainable params")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze backbone parameters for transfer learning.
    
    Args:
        model: PyTorch model with backbone attribute
        freeze: Whether to freeze (True) or unfreeze (False)
    
    Examples:
        >>> freeze_backbone(model, freeze=True)  # Feature extraction
        >>> freeze_backbone(model, freeze=False)  # Fine-tuning
    """
    # Common backbone names in torchvision models
    backbone_names = ['backbone', 'features', 'encoder', 'base']
    
    for name in backbone_names:
        if hasattr(model, name):
            backbone = getattr(model, name)
            for param in backbone.parameters():
                param.requires_grad = not freeze
            return
    
    # If no standard backbone found, freeze all but last layer
    params = list(model.parameters())
    for param in params[:-2]:  # Keep last 2 params (weight, bias) trainable
        param.requires_grad = not freeze


def set_trainable_layers(model: nn.Module, num_layers: int) -> None:
    """
    Set the last N layers to be trainable, freeze the rest.
    
    Args:
        model: PyTorch model
        num_layers: Number of layers from end to make trainable
    
    Examples:
        >>> set_trainable_layers(model, 3)  # Train last 3 layers
    """
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False
    
    # Get all modules
    modules = list(model.modules())
    
    # Unfreeze last N modules with parameters
    count = 0
    for module in reversed(modules):
        if list(module.parameters(recurse=False)):  # Has own parameters
            for param in module.parameters(recurse=False):
                param.requires_grad = True
            count += 1
            if count >= num_layers:
                break


# ==============================================================================
# Detection and Segmentation Helpers
# ==============================================================================

def compute_iou(
    box1: Union[List[float], torch.Tensor],
    box2: Union[List[float], torch.Tensor]
) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Boxes in format [x1, y1, x2, y2] (top-left and bottom-right corners).
    
    Args:
        box1: First bounding box
        box2: Second bounding box
    
    Returns:
        IoU score (0 to 1)
    
    Examples:
        >>> iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        >>> print(f"IoU: {iou:.3f}")  # ~0.143
    """
    if isinstance(box1, torch.Tensor):
        box1 = box1.tolist()
    if isinstance(box2, torch.Tensor):
        box2 = box2.tolist()
    
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_dice(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor
) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        Dice coefficient (0 to 1)
    
    Examples:
        >>> dice = compute_dice(prediction, ground_truth)
        >>> print(f"Dice: {dice:.3f}")
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    intersection = (pred_mask & gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2 * intersection / total).item()


def compute_mask_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor
) -> float:
    """
    Compute IoU for segmentation masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        IoU score (0 to 1)
    
    Examples:
        >>> iou = compute_mask_iou(prediction, ground_truth)
        >>> print(f"Mask IoU: {iou:.3f}")
    """
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


# ==============================================================================
# Dataset Utilities
# ==============================================================================

class ImageFolderDataset(Dataset):
    """
    Simple image folder dataset with transform support.
    
    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                ...
    
    Examples:
        >>> dataset = ImageFolderDataset('data/train', transform=train_transform)
        >>> image, label = dataset[0]
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[v2.Compose] = None
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory containing class folders
            transform: Optional transforms to apply
        """
        self.root = Path(root)
        self.transform = transform
        
        # Get class names
        self.classes = sorted([
            d.name for d in self.root.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.samples = []
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = load_image(path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    
    Examples:
        >>> train_loader, val_loader = create_data_loaders(train_ds, val_ds, batch_size=64)
        >>> for images, labels in train_loader:
        ...     # Training loop
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


# ==============================================================================
# Visualization Helpers
# ==============================================================================

def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    
    Returns:
        Denormalized tensor in [0, 1] range
    
    Examples:
        >>> denorm = denormalize_image(normalized_tensor)
        >>> plt.imshow(denorm.permute(1, 2, 0))
    """
    if mean is None or std is None:
        mean, std = get_imagenet_stats()
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    denorm = tensor * std + mean
    return denorm.clamp(0, 1)


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2
) -> torch.Tensor:
    """
    Create a grid of images.
    
    Args:
        images: Batch of images (N, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
    
    Returns:
        Grid tensor (C, H', W')
    
    Examples:
        >>> grid = make_grid(batch_images, nrow=4)
        >>> plt.imshow(grid.permute(1, 2, 0))
    """
    from torchvision.utils import make_grid as tv_make_grid
    return tv_make_grid(images, nrow=nrow, padding=padding, normalize=False)
