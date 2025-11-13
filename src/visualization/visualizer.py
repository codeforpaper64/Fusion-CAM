import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# Change these imports:
from src.cam_methods import GradCAM, GradCAMPP, XGradCAM, ScoreCAM, GroupCAM, UnionCAM, FusionCAM
from src.utils.image_utils import preprocess_image, get_image_files
from src.models.model_loader import ModelLoader

class XAIVisualizer:
    """Main visualizer class for XAI techniques"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 architecture: str = 'resnet50',
                 num_classes: int = 1000,
                 device: Optional[torch.device] = None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = ModelLoader.load_model(
            model_path=model_path,
            architecture=architecture,
            num_classes=num_classes,
            device=self.device
        )
        
        # Get class names
        self.class_names = ModelLoader.get_imagenet_classes()
        
        # Initialize CAM methods
        self.cam_methods = {
            'gradcam': GradCAM(self.model, self.device),
            'gradcampp': GradCAMPP(self.model, self.device),
            'xgradcam': XGradCAM(self.model, self.device),
            'scorecam': ScoreCAM(self.model, self.device),
            'groupcam': GroupCAM(self.model, self.device),
            'unioncam': UnionCAM(self.model, self.device),
            'fusioncam': FusionCAM(self.model, self.device),
        }
        
        print(f" XAIVisualizer initialized with {architecture}")
        print(f" Available methods: {list(self.cam_methods.keys())}")
    
    def visualize_single(self, 
                        image_path: str, 
                        method: str = 'gradcam',
                        target_class: Optional[int] = None,
                        target_layer: Optional[str] = None,
                        save: bool = False,
                        save_dir: str = 'outputs',
                        **kwargs) -> tuple:
        """
        Visualize CAM for a single image
        
        Args:
            image_path: Path to input image
            method: CAM method to use
            target_class: Specific class to visualize
            target_layer: Specific layer to use
            save: Whether to save the result
            save_dir: Directory to save results
            **kwargs: Additional arguments for CAM methods
        """
        if method not in self.cam_methods:
            raise ValueError(f"Method '{method}' not supported. Available: {list(self.cam_methods.keys())}")
        
        # Preprocess image
        input_tensor, display_img = preprocess_image(image_path, device=self.device)
        
        # Get target layer
        if target_layer is None:
            target_layer = ModelLoader.get_target_layer(self.model)
        
        # Generate CAM
        start_time = time.time()
        cam_method = self.cam_methods[method]
        heatmap, pred_class = cam_method.generate_cam(
            input_tensor, 
            target_layer=target_layer,
            target_class=target_class,
            **kwargs
        )
        exec_time = time.time() - start_time
        
        # Get class name
        pred_name = self.class_names.get(pred_class, f"Class {pred_class}")
        target_name = self.class_names.get(target_class, f"Class {target_class}") if target_class else pred_name
        
        print(f" {method.upper()} completed in {exec_time:.3f}s")
        print(f" Target: {target_name}")
        
        # Create visualization
        self._create_visualization(display_img, heatmap, method, target_name)
        
        # Save if requested
        if save:
            self._save_visualization(image_path, display_img, heatmap, method, target_name, save_dir)
        
        return heatmap, pred_class
    
    def process_folder(self,
                      folder_path: str,
                      method: str = 'gradcam',
                      target_class: Optional[int] = None,
                      save_dir: str = 'outputs',
                      max_images: Optional[int] = None,
                      **kwargs) -> Dict[str, tuple]:
        """
        Process all images in a folder
        
        Args:
            folder_path: Path to image folder
            method: CAM method to use
            target_class: Specific class to visualize
            save_dir: Directory to save results
            max_images: Maximum number of images to process
            **kwargs: Additional arguments for CAM methods
        """
        # Get image files
        image_files = get_image_files(folder_path)
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f" Processing {len(image_files)} images from {folder_path}")
        
        results = {}
        target_layer = ModelLoader.get_target_layer(self.model)
        
        for image_path in tqdm(image_files, desc=f"Processing {method}"):
            try:
                # Preprocess image
                input_tensor, display_img = preprocess_image(image_path, device=self.device)
                
                # Generate CAM
                cam_method = self.cam_methods[method]
                heatmap, pred_class = cam_method.generate_cam(
                    input_tensor,
                    target_layer=target_layer,
                    target_class=target_class,
                    **kwargs
                )
                
                # Save result
                self._save_visualization(image_path, display_img, heatmap, method, 
                                       self.class_names.get(pred_class, f"Class {pred_class}"), 
                                       save_dir)
                
                results[os.path.basename(image_path)] = (heatmap, pred_class)
                
            except Exception as e:
                print(f" Error processing {image_path}: {e}")
                continue
        
        print(f" Processed {len(results)} images successfully")
        return results
    
    def _create_visualization(self, display_img, heatmap, method, class_name):
        """Create visualization with multiple views"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(display_img)
        axes[0].set_title(f'Original Image\n{class_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'{method.upper()} Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(display_img)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title(f'{method.upper()} Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _save_visualization(self, image_path, display_img, heatmap, method, class_name, save_dir):
        """Save visualization to file"""
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_{method}.png")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display_img)
        ax.imshow(heatmap, cmap='jet', alpha=0.6)
        ax.set_title(f'{method.upper()} - {class_name}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def get_available_methods(self) -> List[str]:
        """Get list of available CAM methods"""
        return list(self.cam_methods.keys())

        
    def create_comparison_grid(
        self, image_path, methods=None, target_layer: Optional[str] = None,
        save_path="outputs/comparison_grid.png"
    ):
        """
        Create a 2-row grid:
        - Row 1: Original + Heatmaps of selected techniques
        - Row 2: Original + Overlays of selected techniques
        """
        if methods is None:
            methods = ['gradcam', 'scorecam', 'unioncam', 'fusioncam']
        
        # Preprocess image once
        input_tensor, display_img = preprocess_image(image_path, device=self.device)
        
        # Get prediction and target layer once
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            pred_name = self.class_names.get(pred_class, f"Class {pred_class}")
        
        if target_layer is None:
            target_layer = ModelLoader.get_target_layer(self.model)
        
        n_methods = len(methods)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(2 * (n_methods + 1), 6))
        
        # --- Column 0: Original image (both rows) ---
        for r in range(2):
            axes[r, 0].imshow(display_img)
            axes[r, 0].set_title("Original\nImage", fontsize=12, fontweight='bold')
            axes[r, 0].axis('off')
        
        # --- Process each CAM method ---
        for col, method in enumerate(methods, 1):
            print(f" Processing {method} for comparison grid...")
            try:
                cam_method = self.cam_methods[method]
                heatmap, _ = cam_method.generate_cam(
                    input_tensor,
                    target_layer=target_layer,
                    target_class=pred_class
                )
    
                # Row 1 – Heatmap only
                axes[0, col].imshow(heatmap, cmap='jet')
                axes[0, col].set_title(f'{method.upper()}\nHeatmap', fontsize=12, fontweight='bold')
                axes[0, col].axis('off')
    
                # Row 2 – Overlay
                axes[1, col].imshow(display_img)
                axes[1, col].imshow(heatmap, cmap='jet', alpha=0.6)
                axes[1, col].set_title(f'{method.upper()}\nOverlay', fontsize=12, fontweight='bold')
                axes[1, col].axis('off')
    
            except Exception as e:
                print(f" Error with {method}: {e}")
                for r in range(2):
                    axes[r, col].text(
                        0.5, 0.5, f'{method.upper()}\nError',
                        ha='center', va='center', transform=axes[r, col].transAxes,
                        fontsize=12, color='red'
                    )
                    axes[r, col].axis('off')
    
        plt.suptitle(
            f'XAI Method Comparison\nPredicted: {pred_name}',
            fontsize=12, y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
        print(f" Comparison grid saved to: {save_path}")
        return fig

