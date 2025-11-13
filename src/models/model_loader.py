import torch
import torch.nn as nn
import torchvision.models as models
import requests
import json
from typing import Optional, Dict, Any


class ModelLoader:
    """Flexible model loader for both pretrained and custom models"""
    
    @staticmethod
    def load_model(model_path: Optional[str] = None, 
                   architecture: str = 'resnet50',
                   num_classes: int = 1000,
                   device: torch.device = None) -> nn.Module:
        """
        Load model from path or use pretrained
        
        Args:
            model_path: Path to model weights (None for pretrained)
            architecture: Model architecture
            num_classes: Number of output classes
            device: Target device
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            model = ModelLoader._load_custom_model(architecture, model_path, num_classes)
            print(f"✅ Loaded custom model from {model_path}")
        else:
            model = ModelLoader._load_pretrained_model(architecture)
            print(f"✅ Loaded pretrained {architecture}")
            
        model.to(device)
        model.eval()
        return model
    
    @staticmethod
    def _load_pretrained_model(architecture: str) -> nn.Module:
        """Load pretrained ImageNet model"""
        arch_lower = architecture.lower()
        
        if arch_lower == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif arch_lower == 'vgg16':
            return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif arch_lower == 'mobilenet_v2':
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif arch_lower == 'efficientnet_b0':
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    @staticmethod
    def _load_custom_model(architecture: str, model_path: str, num_classes: int) -> nn.Module:
        """Load custom trained model"""
        arch_lower = architecture.lower()
        
        if arch_lower == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch_lower == 'vgg16':
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif arch_lower == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif arch_lower == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture for custom model: {architecture}")
        
        # Load weights with flexible key handling
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
            
        return model
    
    @staticmethod
    def get_imagenet_classes() -> Dict[int, str]:
        """Get ImageNet class mapping"""
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        try:
            response = requests.get(url)
            imagenet_class_index = response.json()
            return {int(k): v[1] for k, v in imagenet_class_index.items()}
        except:
            # Fallback to basic class mapping
            return {i: f"class_{i}" for i in range(1000)}
    
    @staticmethod
    def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
        """Get appropriate target layer for CAM methods"""
        if layer_name:
            # Find layer by name
            for name, module in model.named_modules():
                if name == layer_name:
                    return module
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Auto-detect based on architecture
        if hasattr(model, 'features'):
            return model.features[-1]  # VGG, MobileNet
        elif hasattr(model, 'layer4'):
            return model.layer4[-1]    # ResNet
        elif hasattr(model, 'features') and hasattr(model.features, '8'):  # EfficientNet
            return model.features[8]
        else:
            # Fallback to last convolutional layer
            last_conv = None
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
            raise ValueError("Could not auto-detect target layer")