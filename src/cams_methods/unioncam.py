import torch
import torch.nn.functional as F
import numpy as np
from .base_cam import BaseCAM
from .scorecam import ScoreCAM


class UnionCAM(BaseCAM):
    """Union-CAM: Combines Denoised CAM and Score-CAM with union strategy"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
        self.scorecam = ScoreCAM(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        pass
        
    
    def _generate_denoised_cam(self, input_tensor, target_layer=None, target_class=None, theta=10):
        """
        Generate denoised CAM internally (not as separate class)
        """
        self.activations.clear()
        self.gradients.clear()
        
        # Use the provided target_layer directly (no _find_target_layer call)
        self._register_hooks(target_layer)
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            score = output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Retrieve feature maps and gradients
            A = self.activations[0][0]  # [C, H, W]
            W = self.gradients[0][0].clone()  # [C, H, W]
            
            # Channel-wise denoising
            for c in range(W.shape[0]):
                channel = W[c]
                channel_np = channel.detach().cpu().numpy()
                
                # Get threshold for this channel
                threshold_c = np.percentile(channel_np, theta)
                
                # Apply denoising
                mask = channel < threshold_c
                W[c][mask] = 0
                
            relu_W = F.relu(W)
            
            # α_ij normalization
            indicator = (W > 0).float()
            sum_weights = torch.sum(W * indicator, dim=(1, 2), keepdim=True) + 1e-8
            alpha = indicator / sum_weights
            
            # Apply denoising equation
            L_denoising = torch.sum(alpha * relu_W * A, dim=0)
            
            # Use BaseCAM methods for upsampling and normalization
            L_denoising = self.upsample_heatmap(L_denoising.unsqueeze(0).unsqueeze(0), input_tensor.shape[2:])
            L_denoising = self.normalize_heatmap(L_denoising)
            
            return L_denoising, target_class
            
        finally:
            self._remove_hooks()
    
    def generate_cam(self, input_tensor, target_layer=None, target_class=None, theta=10, **kwargs):
        """
        Generate Union-CAM by combining Denoised CAM and Score-CAM
        """
        # Generate component CAMs
        L_denoising, _ = self._generate_denoised_cam(input_tensor, target_layer, target_class, theta)
        L_region, _ = self.scorecam.generate_cam(input_tensor, target_layer, target_class)
        
        # Use BaseCAM normalization methods
        L_denoising = self.normalize_heatmap(L_denoising)
        L_region = self.normalize_heatmap(L_region)
        
        # Get prediction from original input
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item() if target_class is None else target_class
        
        # Create black image for baseline
        Ib = torch.zeros_like(input_tensor).to(self.device)
        
        # Apply masks to images using BaseCAM method
        masked_denoising, _ = self.apply_mask_to_image(L_denoising, input_tensor)
        masked_region, _ = self.apply_mask_to_image(L_region, input_tensor)
        
        # Run inference on masked images to get confidence scores
        with torch.no_grad():
            f_black = torch.softmax(self.model(Ib), dim=1)
            f_denoised = torch.softmax(self.model(masked_denoising), dim=1)
            f_region = torch.softmax(self.model(masked_region), dim=1)
        
        # Calculate confidence scores
        β_black = f_black[0, pred_class]
        β_denoising = (f_denoised[0, pred_class] - β_black).item()
        β_region = (f_region[0, pred_class] - β_black).item()
        
        
        # Weighted combination of CAMs
        L_de_region = β_denoising * L_denoising + β_region * L_region
        L_de_region = self.normalize_heatmap(L_de_region)
        
        # Get β for fused CAM
        masked_de_region, _ = self.apply_mask_to_image(L_de_region, input_tensor)
        with torch.no_grad():
            f_de_region = torch.softmax(self.model(masked_de_region), dim=1)
        β_de_region = (f_de_region[0, pred_class] - β_black).item()
        
        
        # Final selection with weight multiplication
        if β_de_region > β_region:
            union_cam = L_de_region * β_de_region
            selected_method = "De-Region"
        else:
            union_cam = L_region * β_region
            selected_method = "Region"
        
        # Print selection info (optional)
        print(f"Union-CAM selected: {selected_method} "
              f"(β_de_region: {β_de_region:.3f}, β_region: {β_region:.3f})")
        
        # Renormalize final map using BaseCAM
        union_cam = self.normalize_heatmap(union_cam)
        
        return union_cam, pred_class