import torch
import torch.nn.functional as F
import numpy as np
from kornia.filters import gaussian_blur2d
from .base_cam import BaseCAM

class GroupCAM(BaseCAM):
    """GroupCAM implementation"""
    
    def __init__(self, model, device=None):
        super().__init__(model, device)
    
    def compute_cam_weights(self, activations, gradients):
        """GroupCAM has complex implementation, override generate_cam"""
        pass
    
    def generate_cam(self, input_tensor, target_layer=None, target_class=None, groups=32, **kwargs):
        """GroupCAM specific implementation"""
        activations = []
        gradients = []
    
        def save_gradients(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())
    
        def activation_hook(module, input, output):
            activations.append(output.detach())
    
        # Register hooks
        if target_layer is None:
            target_layer = self._find_target_layer(self.model)
        forward_hook = target_layer.register_forward_hook(activation_hook)
        backward_hook = target_layer.register_full_backward_hook(save_gradients)
    
        try:
            # Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item() if target_class is None else target_class
            score = output[0, pred_class]
    
            # Backward pass
            score.backward()
    
            # Get activations and gradients
            grads = gradients[0]     # [1, C, H, W]
            acts = activations[0]    # [1, C, H, W]
            b, c, h, w = acts.shape
    
            if c % groups != 0:
                raise ValueError(f"Number of channels ({c}) must be divisible by number of groups ({groups}).")
    
            # Compute weighted activations
            alpha = grads.view(b, c, -1).mean(2)  # [1, C]
            weights = alpha.view(b, c, 1, 1)
            weighted_acts = weights * acts       # [1, C, H, W]
    
            # Group-wise sum - FIXED: Don't reduce to single channel yet
            group_chunks = torch.chunk(weighted_acts, groups, dim=1)
            saliency_map = torch.cat(group_chunks, dim=0)  # [groups, C//groups, H, W]
            saliency_map = saliency_map.sum(dim=1, keepdim=True)  # [groups, 1, H, W]
            saliency_map = F.relu(saliency_map)
            
            # Upsample each group map individually
            upsampled_maps = []
            for i in range(groups):
                single_map = saliency_map[i]  # [1, H, W]
                upsampled = self.upsample_heatmap(single_map.squeeze(0).cpu().numpy(), input_tensor.shape[2:])
                upsampled_maps.append(torch.from_numpy(upsampled).to(self.device))
            
            saliency_map = torch.stack(upsampled_maps)  # [groups, H, W]
    
            # Normalize each group map
            norm_maps = []
            for i in range(groups):
                single_map = saliency_map[i]  # [H, W]
                norm_map = self.normalize_heatmap(single_map.cpu().numpy())
                norm_maps.append(torch.from_numpy(norm_map).to(self.device))
            
            norm_maps = torch.stack(norm_maps)  # [groups, H, W]
    
            # Baseline on blurred input
            blur = lambda x: gaussian_blur2d(x, (51, 51), (25., 25.))  # Reduced kernel size
    
            with torch.no_grad():
                blurred_input = blur(input_tensor)
    
                # Expand norm maps for blending - FIXED shape
                norm_maps_expanded = norm_maps.unsqueeze(1).expand(-1, input_tensor.size(1), -1, -1)
    
                # Blend input and baseline for each group
                mixed_inputs = []
                for i in range(groups):
                    group_norm = norm_maps_expanded[i].unsqueeze(0)  # [1, C, H, W]
                    mixed = group_norm * input_tensor + (1 - group_norm) * blurred_input
                    mixed_inputs.append(mixed)
                
                mixed_batch = torch.cat(mixed_inputs, dim=0)  # [groups, C, H, W]
    
                # Predict on mixed inputs
                output = self.model(mixed_batch)
                probs = F.softmax(output, dim=1)[:, pred_class]
                base_line = F.softmax(self.model(blurred_input), dim=1)[0, pred_class].detach()
                diff = F.relu(probs - base_line).view(groups, 1, 1)
    
            # Final weighted aggregation
            final_map = (norm_maps * diff).sum(dim=0)  # [H, W]
            final_map = F.relu(final_map).cpu().numpy()
    
            # Normalize final map
            final_map = self.normalize_heatmap(final_map)
    
            return final_map, pred_class
            
        finally:
            forward_hook.remove()
            backward_hook.remove()