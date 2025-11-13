import torch

# Default configurations
DEFAULT_CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'image_size': 224,
    'architecture': 'resnet50',
    'num_classes': 1000
}

# Supported architectures
SUPPORTED_ARCHITECTURES = {
    'resnet50': 'torchvision',
    'vgg16': 'torchvision', 
    'mobilenet_v2': 'torchvision',
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]