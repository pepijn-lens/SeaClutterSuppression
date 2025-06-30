import numpy as np

# Import the data loading function from your training file
from sea_clutter import create_data_loaders
from .end_to_end_helper import (
    analyze_single_sample, 
    evaluate_spatial_performance,
    print_spatial_performance_report,
    plot_spatial_performance_analysis,
)
import models
import torch

def comprehensive_evaluation(dataset_path, model_path, base_filter_size, save=None, clustering_params=None, marimo_var=False, distance_threshold=5.0):
    """Run a comprehensive evaluation of the end-to-end model using both count-based and spatial evaluation"""
    
    # Create data loaders same as in training
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,
    )
    
    # Use test loader for evaluation
    eval_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    eval_dataset_name = "test" if len(test_loader.dataset) > 0 else "validation"
    
    print(f"Using {eval_dataset_name} dataset for evaluation")
    print(f"Dataset size: {len(eval_loader.dataset)} samples")
    
    # Model setup - get actual channels from a sample
    sample_batch = next(iter(eval_loader))
    sample_image = sample_batch[0][0]  # First image from first batch
    
    # For sequence data with 3 frames, n_channels should be 3
    if len(sample_image.shape) > 1:  # (C, H, W)
        n_channels = sample_image.shape[0]
    else:  # Single channel
        n_channels = 1
    
    print(f"Detected {n_channels} input channels from sample shape: {sample_image.shape}")
        
    # Create model with detected parameters
    model = models.EndToEndTargetDetector(
        unet_weights_path=model_path,
        n_channels=n_channels,
        base_filter_size=base_filter_size,
        clustering_params=clustering_params or {'min_area': 1, 'eps': 1, 'min_samples': 1}
    ).to('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    
    # 1. NEW: Spatial evaluation (recommended)
    print(f"\n1. RUNNING SPATIAL EVALUATION (distance threshold: {distance_threshold} pixels)...")
    spatial_results = evaluate_spatial_performance(model, eval_loader, distance_threshold, eval_dataset_name)
    print_spatial_performance_report(spatial_results)
    
    # Create spatial performance plots
    if save:
        spatial_save_path = f"{save}/spatial_performance"
    else:
        spatial_save_path = None
    
    spatial_plots = plot_spatial_performance_analysis(spatial_results, save_path=spatial_save_path, marimo=marimo_var)
    
        
    # 3. Summary comparison
    print(f"\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {eval_dataset_name} ({len(eval_loader.dataset)} samples)")
    print(f"Distance threshold: {distance_threshold} pixels")
    print(f"\nSPATIAL EVALUATION RESULTS:")
    print(f"  Precision: {spatial_results['precision']:.3f}")
    print(f"  Recall:    {spatial_results['recall']:.3f}")
    print(f"  F1-Score:  {spatial_results['f1_score']:.3f}")
    print(f"  True Positives:  {spatial_results['total_true_positives']}")
    print(f"  False Positives: {spatial_results['total_false_positives']}")
    print(f"  False Negatives: {spatial_results['total_false_negatives']}")
    
    # Return both results for further analysis
    return {
        'spatial_results': spatial_results,
        'spatial_plots': spatial_plots,
    }

def interactive_sample_explorer(dataset_path, model_path, base_filter_size, clustering_params=None, distance_threshold=5.0):
    """
    Interactive method to explore samples from the dataset using the end-to-end target detector with spatial evaluation
    """
    print("Loading dataset and model...")
    
    _, val_loader, test_loader = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=1,  # Load one sample at a time
    )
    
    # Use test loader for exploration, fallback to validation
    data_loader = test_loader if len(test_loader.dataset) > 0 else val_loader
    dataset_name = "test" if len(test_loader.dataset) > 0 else "validation"
    dataset = data_loader.dataset
    
    print(f"Loaded {dataset_name} dataset with {len(dataset)} samples")
    
    # Load model
    sample_batch = next(iter(data_loader))
    sample_image = sample_batch[0][0]
    n_channels = sample_image.shape[0] if len(sample_image.shape) == 3 else 1
    
    model = models.EndToEndTargetDetector(
        unet_weights_path=model_path,
        n_channels=n_channels,
        base_filter_size=base_filter_size,
        clustering_params=clustering_params or {'min_area': 1, 'eps': 1, 'min_samples': 1}
    )
    
    print(f"Model loaded with {n_channels} input channels")
    print(f"Using spatial evaluation with distance threshold: {distance_threshold} pixels")
    print("\n" + "="*60)
    print("INTERACTIVE SAMPLE EXPLORER")
    print("="*60)
    print("Commands:")
    print("  Enter a number (0-{}) to analyze a specific sample".format(len(dataset)-1))
    print("  'random' or 'r' for a random sample")
    print("  'threshold X' to change distance threshold (e.g., 'threshold 10')")
    print("  'quit' or 'q' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input(f"\nEnter sample index (0-{len(dataset)-1}) or command: ").strip().lower()
            
            if user_input in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input in ['random', 'r']:
                sample_idx = np.random.randint(0, len(dataset))
                print(f"Randomly selected sample {sample_idx}")
            
            elif user_input.startswith('threshold'):
                try:
                    new_threshold = float(user_input.split()[1])
                    distance_threshold = new_threshold
                    print(f"Distance threshold updated to {distance_threshold} pixels")
                    continue
                except (IndexError, ValueError):
                    print("Invalid threshold format. Use 'threshold X' where X is a number (e.g., 'threshold 10')")
                    continue
                
            else:
                try:
                    sample_idx = int(user_input)
                    if sample_idx < 0 or sample_idx >= len(dataset):
                        print(f"Invalid index. Please enter a number between 0 and {len(dataset)-1}")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number, 'random', 'threshold X', or 'quit'")
                    continue
            
            # Process the selected sample with spatial evaluation
            analyze_single_sample(model, dataset, sample_idx, distance_threshold)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the end-to-end detector")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained U-Net weights")
    parser.add_argument("--base-filter-size", type=int, default=64, help="Base filter size for the U-Net model")
    parser.add_argument("--save-path", type=str, default="end_to_end_results",
                        help="Directory where evaluation figures will be saved")
    parser.add_argument("--cluster-min-area", type=int, default=1, help="Minimum area for a cluster to be valid")
    parser.add_argument("--cluster-eps", type=float, default=1.0, help="DBSCAN eps parameter")
    parser.add_argument("--cluster-min-samples", type=int, default=1, help="DBSCAN min_samples parameter")
    parser.add_argument("--distance-threshold", type=float, default=1.5, 
                        help="Distance threshold for spatial evaluation (pixels)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive sample explorer after evaluation")

    args = parser.parse_args()

    clustering = {
        'min_area': args.cluster_min_area,
        'eps': args.cluster_eps,
        'min_samples': args.cluster_min_samples,
    }

    # Run comprehensive evaluation with both spatial and count-based methods
    results = comprehensive_evaluation(
        args.dataset,
        args.model,
        save=args.save_path,
        base_filter_size=args.base_filter_size,
        clustering_params=clustering,
        distance_threshold=args.distance_threshold,
    )
    
    if args.interactive:
        print(f"\nLaunching interactive explorer...")
        interactive_sample_explorer(
            args.dataset,
            args.model,
            args.base_filter_size,
            clustering_params=clustering,
            distance_threshold=args.distance_threshold,
        )
