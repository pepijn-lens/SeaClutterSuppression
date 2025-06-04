import numpy as np

# Import the data loading function from your training file
from sea_clutter import create_data_loaders
from .end_to_end_helper import plot_performance_analysis, print_performance_report, evaluate_target_count_performance, show_dataset_stats, analyze_single_sample
import models

def comprehensive_evaluation(dataset_path, model_path, save='multi_frame', clustering_params=None):
    """Run a comprehensive evaluation of the end-to-end model using test data"""
    
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
        clustering_params=clustering_params or {'min_area': 3, 'eps': 1, 'min_samples': 1}
    ).to('mps')  # Move model to MPS
    
    # Evaluate performance on test data
    print("Running comprehensive evaluation on test data...")
    results = evaluate_target_count_performance(model, eval_loader, eval_dataset_name)
    
    # Print report
    print_performance_report(results)
    
    # Create plots
    plot_performance_analysis(results, save_path=f'end_to_end_analysis/{save}')
    
    return results

def interactive_sample_explorer(dataset_path, model_path, clustering_params=None):
    """
    Interactive method to explore samples from the dataset using the end-to-end target detector
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
        clustering_params=clustering_params or {'min_area': 3, 'eps': 1, 'min_samples': 1}
    )
    
    print(f"Model loaded with {n_channels} input channels")
    print("\n" + "="*60)
    print("INTERACTIVE SAMPLE EXPLORER")
    print("="*60)
    print("Commands:")
    print("  Enter a number (0-{}) to analyze a specific sample".format(len(dataset)-1))
    print("  'random' or 'r' for a random sample")
    print("  'stats' or 's' to show dataset statistics")
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
            
            elif user_input in ['stats', 's']:
                show_dataset_stats(dataset)
                continue
                
            else:
                try:
                    sample_idx = int(user_input)
                    if sample_idx < 0 or sample_idx >= len(dataset):
                        print(f"Invalid index. Please enter a number between 0 and {len(dataset)-1}")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number, 'random', 'stats', or 'quit'")
                    continue
            
            # Process the selected sample
            analyze_single_sample(model, dataset, sample_idx)
            
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
    parser.add_argument("--save-path", type=str, default="end_to_end_results",
                        help="Directory where evaluation figures will be saved")
    parser.add_argument("--cluster-min-area", type=int, default=3, help="Minimum area for a cluster to be valid")
    parser.add_argument("--cluster-eps", type=float, default=1.0, help="DBSCAN eps parameter")
    parser.add_argument("--cluster-min-samples", type=int, default=1, help="DBSCAN min_samples parameter")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive sample explorer after evaluation")

    args = parser.parse_args()

    clustering = {
        'min_area': args.cluster_min_area,
        'eps': args.cluster_eps,
        'min_samples': args.cluster_min_samples,
    }

    comprehensive_evaluation(
        args.dataset,
        args.model,
        save=args.save_path,
        clustering_params=clustering,
    )

    if args.interactive:
        interactive_sample_explorer(
            args.dataset,
            args.model,
            clustering_params=clustering,
        )
