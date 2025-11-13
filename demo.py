#!/usr/bin/env python3
"""
Demo script for testing XAI visualization methods on a sample image.

This script:
  1. Generates individual CAM visualizations for multiple explanation methods.
  2. Produces a comparison grid showing all methods side by side.

Outputs are saved in the 'outputs/' directory.
"""

import sys
import os

# Add 'src' to the path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.visualization.visualizer import XAIVisualizer


def main():
    print("XAI Visualization Demo")

    # Initialize visualizer (uses pretrained model by default)
    vis = XAIVisualizer()

    # Path to test image
    image_path = "images/test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place a test image in the 'images/' directory.")
        return

    methods_to_test = [
        "gradcam",
        "gradcampp",
        "xgradcam",
        "scorecam",
        "groupcam",
        "unioncam",
        "fusioncam",
    ]

    print(f"Processing image: {image_path}")
    print(f"Available methods: {vis.get_available_methods()}")
    print("-" * 60)

    # Run and save individual visualizations
    for method in methods_to_test:
        print(f"Running {method.upper()}...")
        try:
            vis.visualize_single(
                image_path=image_path,
                method=method,
                save=True,
                save_dir="outputs",
            )
            print(f"{method.upper()} completed successfully.")
        except Exception as e:
            print(f"Error with {method}: {e}")

    # Generate comparison grid
    print("\nGenerating comparison grid...")
    try:
        vis.create_comparison_grid(
            image_path=image_path,
            methods=methods_to_test,
            save_path="outputs/comparison_grid.png",
            save=True,
        )
        print("Comparison grid saved to 'outputs/comparison_grid.png'.")
    except Exception as e:
        print(f"Error creating comparison grid: {e}")

    print("\nAll results have been saved in the 'outputs/' directory.")


if __name__ == "__main__":
    main()
