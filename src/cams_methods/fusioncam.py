import torch
import torch.nn.functional as F
import numpy as np
from .base_cam import BaseCAM
from .gradcam import GradCAM
from .scorecam import ScoreCAM


class FusionCAM(BaseCAM):
    """Fusion-CAM: Advanced fusion of Denoised Grad-CAM and Score-CAM"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
        self.gradcam = GradCAM(model, device)
        self.scorecam = ScoreCAM(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        pass
    
    def _generate_gradcam_denoised(self, input_tensor, target_layer=None, target_class=None, theta=10):
        """Generate denoised Grad-CAM using the GradCAM class"""
        # Use the GradCAM class to generate base heatmap
        gradcam, pred_class = self.gradcam.generate_cam(input_tensor, target_layer, target_class)
        
        # Apply denoising
        if theta > 0:
            threshold = np.percentile(gradcam, theta)
            zero_mask = gradcam < threshold
            gradcam[zero_mask] = 0
        
        return gradcam, pred_class
        
    def _normalize_weights(self, a: float, b: float, eps: float = 1e-8):
        """Normalize two scalar weights so they sum to 1, safely."""
        total = a + b + eps
        return a / total, b / total

    def generate_cam(self, input_tensor, target_layer=None, target_class=None, theta=10, **kwargs):
        """Generate Fusion-CAM heatmap"""
        # Generate component CAMs
        L_denoising, _ = self._generate_gradcam_denoised(input_tensor, target_layer, target_class, theta)
        L_region, _ = self.scorecam.generate_cam(input_tensor, target_layer, target_class)
        
        # Use BaseCAM normalization methods
        L_denoising = self.normalize_heatmap(L_denoising)
        L_region = self.normalize_heatmap(L_region)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item() if target_class is None else target_class
        
        # Create black image for baseline
        Ib = torch.zeros_like(input_tensor).to(self.device)
        
        # Apply masks to images using BaseCAM method
        masked_denoising, _ = self.apply_mask_to_image(L_denoising, input_tensor)
        masked_region, _ = self.apply_mask_to_image(L_region, input_tensor)
        
        # Run inference on masked images
        with torch.no_grad():
            f_black = torch.softmax(self.model(Ib), dim=1)
            f_denoised = torch.softmax(self.model(masked_denoising), dim=1)
            f_region = torch.softmax(self.model(masked_region), dim=1)
        
        # --- Compute base weights ---
        β_black = f_black[0, pred_class]
        β_denoising = (f_denoised[0, pred_class] - β_black).item()
        β_region = (f_region[0, pred_class] - β_black).item()

        # --- Normalize and combine denoising + region ---
        β_denoising, β_region = self._normalize_weights(β_denoising, β_region)
        L_de_region = β_denoising * L_denoising + β_region * L_region
        L_de_region = self.normalize_heatmap(L_de_region)

        # --- Compute union weights ---
        masked_de_region, _ = self.apply_mask_to_image(L_de_region, input_tensor)
        with torch.no_grad():
            f_de_region = torch.softmax(self.model(masked_de_region), dim=1)
        β_de_region = (f_de_region[0, pred_class] - β_black).item()

        # --- Normalize fusion weights ---
        β_de_region, β_region = self._normalize_weights(β_de_region, β_region)

        # --- Soft fusion ---
        L1 = L_de_region * β_de_region
        L2 = L_region * β_region

        diff = np.abs(L1 - L2)
        sim = np.clip(1 - diff, 0, 1)

        fusion_cam = sim * np.maximum(L1, L2) + (1 - sim) * 0.5 * (L1 + L2)
        fusion_cam = self.normalize_heatmap(fusion_cam)

        return fusion_cam, pred_class
