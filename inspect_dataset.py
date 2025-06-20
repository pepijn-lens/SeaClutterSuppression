#!/usr/bin/env python3
"""
Small utility script to inspect the structure and keys of a PyTorch dataset file.
"""

import torch
import sys
import os

def inspect_dataset(dataset_path):
    """
    Load and inspect a PyTorch dataset file, printing its structure and keys.
    
    Args:
        dataset_path: Path to the .pt dataset file
    """
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        return
    
    try:
        print(f"📂 Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, map_location='cpu')
        
        print(f"\n📋 Dataset Type: {type(dataset)}")
        
        if isinstance(dataset, dict):
            print(f"\n🔑 Dataset Keys ({len(dataset)} total):")
            for key in dataset.keys():
                value = dataset[key]
                if hasattr(value, 'shape'):
                    print(f"  '{key}': {type(value).__name__} with shape {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"  '{key}': {type(value).__name__} with length {len(value)}")
                else:
                    print(f"  '{key}': {type(value).__name__} - {value}")
        
        elif isinstance(dataset, (list, tuple)):
            print(f"\n📦 Dataset is a {type(dataset).__name__} with {len(dataset)} elements")
            if len(dataset) > 0:
                print(f"   First element type: {type(dataset[0])}")
                if hasattr(dataset[0], 'shape'):
                    print(f"   First element shape: {dataset[0].shape}")
        
        else:
            print(f"\n📦 Dataset is a single {type(dataset).__name__}")
            if hasattr(dataset, 'shape'):
                print(f"   Shape: {dataset.shape}")
        
        # Additional analysis for common dataset structures
        if isinstance(dataset, dict):
            print(f"\n🔍 Detailed Analysis:")
            
            # Check for common keys
            common_keys = ['sequences', 'images', 'masks', 'labels', 'metadata']
            found_keys = [key for key in common_keys if key in dataset]
            
            if found_keys:
                print(f"   Found common keys: {found_keys}")
            
            # Analyze shapes and types
            for key, value in dataset.items():
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    print(f"   {key}: shape {value.shape}, dtype {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
                    
                    # Show some statistics for numerical data
                    if hasattr(value, 'min') and hasattr(value, 'max'):
                        try:
                            print(f"     ↳ Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                            print(f"     ↳ Mean: {value.mean().item():.3f}, Std: {value.std().item():.3f}")
                        except:
                            pass  # Skip if statistics calculation fails
        
        print(f"\n✅ Dataset inspection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

def main():
    """Main function to handle command line arguments or interactive input."""
    
    if len(sys.argv) > 1:
        # Dataset path provided as command line argument
        dataset_path = sys.argv[1]
    else:
        # Interactive mode - list available datasets and ask user
        print("🔍 Dataset Inspector")
        print("=" * 50)
        
        # Look for common dataset locations
        common_paths = [
            "local_data/",
            "data/", 
            "pretrained/",
            "./"
        ]
        
        found_datasets = []
        for path in common_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('.pt'):
                        found_datasets.append(os.path.join(path, file))
        
        if found_datasets:
            print(f"\n📁 Found {len(found_datasets)} dataset files:")
            for i, dataset in enumerate(found_datasets):
                print(f"  {i+1}. {dataset}")
            
            print(f"\nEnter dataset path or number (1-{len(found_datasets)}):")
        else:
            print("\n📁 No .pt files found in common directories.")
            print("Enter the full path to your dataset:")
        
        try:
            user_input = input("> ").strip()
            
            # Check if input is a number (selecting from list)
            if user_input.isdigit() and found_datasets:
                dataset_idx = int(user_input) - 1
                if 0 <= dataset_idx < len(found_datasets):
                    dataset_path = found_datasets[dataset_idx]
                else:
                    print(f"❌ Invalid selection. Please choose 1-{len(found_datasets)}")
                    return
            else:
                # Treat as file path
                dataset_path = user_input
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Inspect the selected dataset
    inspect_dataset(dataset_path)

if __name__ == "__main__":
    main()
