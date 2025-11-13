import torch
import torch.nn.functional as F
from .base_cam import BaseCAM

class GradCAMPP(BaseCAM):
    """Grad-CAM++ implementation"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        """Grad-CAM++ specific weight computation"""
        grad = gradients[0]  # [B, C, H, W]
        activation = activations[0]  # [B, C, H, W]

        # Grad-CAM++ specific weighting
        grad = grad.squeeze(0)
        activation = activation.squeeze(0)

        grad_squared = grad ** 2
        grad_cubed = grad_squared * grad

        # α coefficients
        sum_activ = activation.sum(dim=(1, 2), keepdim=True) + 1e-8
        alpha_num = grad_squared
        alpha_denom = 2 * grad_squared + sum_activ * grad_cubed
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom

        relu_grad = F.relu(grad)
        weights = (alpha * relu_grad).sum(dim=(1, 2))

        cam = (weights[:, None, None] * activation).sum(dim=0)
        return cam.unsqueeze(0).unsqueeze(0)