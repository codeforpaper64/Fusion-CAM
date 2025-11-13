import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List
import os


def preprocess_image(image_path: str, 
                    size: int = 224,
                    device: torch.device = None) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to input image
        size: Target size for resizing
        device: Target device for tensor
    
    Returns:
        input_tensor: Preprocessed tensor for model
        display_img: Original image for visualization
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model preprocessing (with normalization)
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Display preprocessing (without normalization)
    display_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    display_img = display_transform(img).numpy().transpose(1, 2, 0)
    display_img = np.clip(display_img, 0, 1)  # Ensure [0, 1] range
    
    return input_tensor, display_img


def get_image_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to image folder
        extensions: List of allowed extensions
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(folder_path, file))
    
    return sorted(image_files)  # Sort for consistent ordering