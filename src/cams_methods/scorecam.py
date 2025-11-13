import torch
import torch.nn.functional as F
from .base_cam import BaseCAM

class ScoreCAM(BaseCAM):
    """ScoreCAM implementation - overrides generate_cam since it's not gradient-based"""
    
    def compute_cam_weights(self, activations, gradients):
        # Not used for ScoreCAM
        pass
    
    def generate_cam(self, input_tensor, target_layer=None, target_class=None, **kwargs):
        """Override since ScoreCAM doesn't use gradients"""
        activations = []
    
        def activation_hook(module, input, output):
            activations.append(output)
    
        if target_layer is None:
            target_layer = self._find_target_layer(self.model)
        forward_hook = target_layer.register_forward_hook(activation_hook)
    
        try:
            # First pass to get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_class = output.argmax(dim=1).item() if target_class is None else target_class
    
            # Get activations
            _ = self.model(input_tensor)
            activation_maps = activations[0].squeeze(0)
    
            # Use BaseCAM upsampling
            upsampled_maps = []
            for i in range(activation_maps.shape[0]):
                single_map = activation_maps[i].unsqueeze(0).unsqueeze(0)
                upsampled_map = self.upsample_heatmap(single_map, input_tensor.shape[2:])
                upsampled_maps.append(torch.from_numpy(upsampled_map).to(self.device))
            
            upsampled_maps = torch.stack(upsampled_maps)
    
            # Process each channel
            contributions = []
            for i in range(activation_maps.shape[0]):
                act_map = upsampled_maps[i]
                act_map_norm = self.normalize_heatmap(act_map)
                act_map_norm = torch.from_numpy(act_map_norm).to(self.device)
                
                masked_input = input_tensor * act_map_norm.unsqueeze(0)
    
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    score = masked_output[0, pred_class]
    
                contributions.append(score)
    
            # Combine maps
            scores = torch.stack(contributions)
            scores = scores - scores.max()  # numerical stability
            score_weights = F.softmax(scores, dim=0)
    
            weighted_activation = (score_weights.view(-1, 1, 1) * upsampled_maps).sum(dim=0)
    
            scorecam = F.relu(weighted_activation)
            # Use BaseCAM upsampling and normalization
            scorecam = self.upsample_heatmap(scorecam, input_tensor.shape[2:])
            scorecam = self.normalize_heatmap(scorecam)
    
            return scorecam, pred_class
            
        finally:
            forward_hook.remove()