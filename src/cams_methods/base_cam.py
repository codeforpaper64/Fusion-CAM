import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
import warnings


class BaseCAM(ABC):
    """Base class for all CAM methods with enhanced functionality"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.activations = []
        self.gradients = []
        self._hooks = []
        
    def _register_hooks(self, target_layer):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])
            
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        self._hooks.extend([forward_handle, backward_handle])
        
    def _remove_hooks(self):
        """Safely remove all hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def normalize_heatmap(self, heatmap):
        """Normalize heatmap to [0, 1] range"""
        if isinstance(heatmap, torch.Tensor):
            # FIX: Use detach() to remove gradients
            heatmap = heatmap.detach().cpu().numpy()
            
        heatmap = heatmap - np.min(heatmap)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap
    
    def upsample_heatmap(self, heatmap, target_size):
        """Upsample heatmap using bilinear interpolation"""
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap).float()
            
        # Ensure proper dimensions [1, 1, H, W]
        if heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        elif heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(0)
            
        # Upsample using bilinear interpolation
        heatmap = F.interpolate(
            heatmap, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # FIX: Use detach() to remove gradients before numpy conversion
        return heatmap.squeeze().detach().cpu().numpy()
    
    def apply_mask_to_image(self, heatmap, input_tensor):
        """Apply heatmap mask to input tensor"""
        if isinstance(heatmap, np.ndarray):
            heatmap_tensor = torch.from_numpy(heatmap).float().to(self.device)
        else:
            heatmap_tensor = heatmap.to(self.device)
            
        # Ensure proper dimensions
        if heatmap_tensor.dim() == 2:
            heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)
        elif heatmap_tensor.dim() == 3:
            heatmap_tensor = heatmap_tensor.unsqueeze(0)
        
        # Resize if needed
        if heatmap_tensor.shape[-2:] != input_tensor.shape[-2:]:
            heatmap_tensor = F.interpolate(
                heatmap_tensor,
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        masked_input = input_tensor * heatmap_tensor
        return masked_input, heatmap_tensor.squeeze().cpu().numpy()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.activations.clear()
        self.gradients.clear()
        self._remove_hooks()
    
    @abstractmethod
    def compute_cam_weights(self, activations, gradients):
        """Compute CAM weights - to be implemented by subclasses"""
        pass
    
    def generate_cam(self, input_tensor, target_layer=None, target_class=None, **kwargs):
        """Generate CAM with enhanced error handling"""
        try:
            self.activations.clear()
            self.gradients.clear()
            
            # Register hooks
            self._register_hooks(target_layer)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            # Determine target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            score = output[0, target_class]
            score.backward(retain_graph=True)
            
            # Compute CAM
            if not self.activations or not self.gradients:
                raise RuntimeError("No activations or gradients captured")
                
            cam = self.compute_cam_weights(self.activations[0], self.gradients[0])
            cam = F.relu(cam)
            
            # Post-processing
            cam = self.upsample_heatmap(cam, input_tensor.shape[2:])
            cam = self.normalize_heatmap(cam)
            
            return cam, target_class
            
        except Exception as e:
            self._cleanup()
            raise e
        finally:
            self._cleanup()
    
    def __call__(self, input_tensor, target_layer=None, target_class=None, **kwargs):
        return self.generate_cam(input_tensor, target_layer, target_class, **kwargs)