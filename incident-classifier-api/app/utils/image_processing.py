"""
Image Processing Utilities
"""

import torch
from torchvision import transforms
from PIL import Image
import io
from typing import Union
import logging

logger = logging.getLogger(__name__)


def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.229]
        )
    ])


def process_image(image_bytes: bytes) -> torch.Tensor:
    """
    Process uploaded image bytes into tensor
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        image = image.convert('RGB')
        
        # Apply transforms
        transform = get_image_transforms()
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError(f"Invalid image format: {e}")


def validate_image(image_bytes: bytes, max_size: int) -> bool:
    """
    Validate image size and format
    
    Args:
        image_bytes: Raw image bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if valid
    """
    # Check size
    if len(image_bytes) > max_size:
        raise ValueError(f"Image size exceeds maximum allowed ({max_size} bytes)")
    
    # Try to open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        return True
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")
