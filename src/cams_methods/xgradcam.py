import torch
from .base_cam import BaseCAM

class XGradCAM(BaseCAM):
    """XGrad-CAM implementation"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        """XGrad-CAM weight computation"""
        activation = activations[0].squeeze(0)      # [C, H, W]
        gradient = gradients[0].squeeze(0)           # [C, H, W]

        # XGrad-CAM: ReLU(sum(A * dY/dA))
        weights = (activation * gradient).sum(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activation).sum(dim=0)  # [H, W]
        return cam.unsqueeze(0).unsqueeze(0)