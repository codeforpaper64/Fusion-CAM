import torch
import torch.nn.functional as F
from .base_cam import BaseCAM

class GradCAM(BaseCAM):
    """Grad-CAM: Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        """Compute weights using global average pooling"""
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        return cam