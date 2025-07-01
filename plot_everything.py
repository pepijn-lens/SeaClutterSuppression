import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models.Unet import UNet

def plot_all_kernels_comprehensive(model):
    """Plot ALL kernels from ALL layers."""
    os.makedirs('kernels_complete', exist_ok=True)
    
    # First pass: compute global min/max for all kernel weights
    all_weights = []
    layers_to_plot = [
        ('enc1_conv1', model.enc1.double_conv[0]),
        ('enc1_conv2', model.enc1.double_conv[2]),
        ('enc2_conv1', model.enc2.double_conv[0]),
        ('enc2_conv2', model.enc2.double_conv[2]),
        ('enc3_conv1', model.enc3.double_conv[0]),
        ('enc3_conv2', model.enc3.double_conv[2]),
        ('up2', model.up2),
        ('dec2_conv1', model.dec2.double_conv[0]),
        ('dec2_conv2', model.dec2.double_conv[2]),
        ('up1', model.up1),
        ('dec1_conv1', model.dec1.double_conv[0]),
        ('dec1_conv2', model.dec1.double_conv[2]),
        ('out_conv', model.out_conv)
    ]
    
    for layer_name, layer in layers_to_plot:
        weights = layer.weight.data.cpu().numpy()
        all_weights.append(weights.flatten())
    
    all_weights = np.concatenate(all_weights)
    global_vmin = all_weights.min()
    global_vmax = all_weights.max()
    global_vabs = max(abs(global_vmin), abs(global_vmax))
    
    print(f"Kernel weights global range: [{global_vmin:.4f}, {global_vmax:.4f}]")
    print(f"Using symmetric range: [{-global_vabs:.4f}, {global_vabs:.4f}]")
    print(f"Global statistics: mean={all_weights.mean():.6f}, std={all_weights.std():.6f}")
    print(f"Percentage of weights > 0.1: {(np.abs(all_weights) > 0.1).mean()*100:.2f}%")
    print(f"Percentage of weights > 0.01: {(np.abs(all_weights) > 0.01).mean()*100:.2f}%")
    print(f"Percentage of weights near zero (|w| < 0.001): {(np.abs(all_weights) < 0.001).mean()*100:.2f}%")
    
    for layer_name, layer in layers_to_plot:
        weights = layer.weight.data.cpu().numpy()
        n_kernels = weights.shape[0]
        
        # Analyze this layer's weights
        layer_min, layer_max = weights.min(), weights.max()
        layer_mean, layer_std = weights.mean(), weights.std()
        layer_abs_mean = np.abs(weights).mean()
        large_weights_pct = (np.abs(weights) > 0.1).mean() * 100
        
        print(f"Plotting {layer_name}: {n_kernels} kernels, shape {weights.shape}")
        print(f"  Layer stats: range=[{layer_min:.4f}, {layer_max:.4f}], mean={layer_mean:.6f}, std={layer_std:.4f}")
        print(f"  Abs mean: {layer_abs_mean:.6f}, >0.1: {large_weights_pct:.1f}%")
        
        # Determine grid size - aim for roughly square layout
        if n_kernels == 1:
            cols, rows = 1, 1
        elif n_kernels <= 16:
            cols = min(4, n_kernels)
            rows = (n_kernels + cols - 1) // cols
        elif n_kernels <= 64:
            cols = 8
            rows = (n_kernels + cols - 1) // cols
        elif n_kernels <= 256:
            cols = 16
            rows = (n_kernels + cols - 1) // cols
        else:
            cols = 20
            rows = (n_kernels + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        fig.suptitle(f'{layer_name} - ALL {n_kernels} kernels - Shape: {weights.shape}', fontsize=16)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            if i < n_kernels:
                # For multi-channel inputs, average across input channels
                if len(weights.shape) == 4 and weights.shape[1] > 1:
                    kernel = np.mean(weights[i], axis=0)
                else:
                    kernel = weights[i, 0] if len(weights.shape) == 4 else weights[i]
                
                # Use global consistent color scale
                im = ax.imshow(kernel, cmap='RdBu_r', vmin=-global_vabs, vmax=global_vabs)
                ax.set_title(f'K{i}', fontsize=8)
                # Add colorbar to each kernel
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'kernels_complete/{layer_name}_ALL_{n_kernels}_kernels.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved kernels_complete/{layer_name}_ALL_{n_kernels}_kernels.png")

def analyze_kernel_weights(model):
    """Analyze kernel weight distributions in detail."""
    print("\n" + "="*60)
    print("DETAILED KERNEL WEIGHT ANALYSIS")
    print("="*60)
    
    layers_to_analyze = [
        ('enc1_conv1', model.enc1.double_conv[0]),
        ('enc1_conv2', model.enc1.double_conv[2]),
        ('enc2_conv1', model.enc2.double_conv[0]),
        ('enc2_conv2', model.enc2.double_conv[2]),
        ('enc3_conv1', model.enc3.double_conv[0]),
        ('enc3_conv2', model.enc3.double_conv[2]),
        ('out_conv', model.out_conv)
    ]
    
    for layer_name, layer in layers_to_analyze:
        weights = layer.weight.data.cpu().numpy().flatten()
        
        print(f"\n{layer_name}:")
        print(f"  Total parameters: {len(weights):,}")
        print(f"  Range: [{weights.min():.6f}, {weights.max():.6f}]")
        print(f"  Mean: {weights.mean():.6f}, Std: {weights.std():.6f}")
        print(f"  |Mean|: {np.abs(weights).mean():.6f}")
        
        # Distribution analysis
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
        for thresh in thresholds:
            pct = (np.abs(weights) > thresh).mean() * 100
            print(f"  Weights with |w| > {thresh}: {pct:.1f}%")
        
        # Check for dead neurons (all weights near zero)
        if len(weights) > 1000:  # Only for conv layers, not 1x1
            weight_matrix = layer.weight.data.cpu().numpy()
            if len(weight_matrix.shape) == 4:  # Conv layer
                # Check each output channel
                dead_channels = 0
                for i in range(weight_matrix.shape[0]):
                    channel_weights = weight_matrix[i].flatten()
                    if np.abs(channel_weights).max() < 0.001:
                        dead_channels += 1
                print(f"  Dead channels (max |w| < 0.001): {dead_channels}/{weight_matrix.shape[0]} ({dead_channels/weight_matrix.shape[0]*100:.1f}%)")

def plot_input_output(original_sample, normalized_sample, output, ground_truth_mask=None):
    """Plot the input images and final output."""
    os.makedirs('input_output', exist_ok=True)
    
    print("Plotting input and output images...")
    
    # Compute consistent ranges for each data type
    orig_min, orig_max = original_sample.min().item(), original_sample.max().item()
    norm_min, norm_max = normalized_sample.min().item(), normalized_sample.max().item()
    out_min, out_max = output.min().item(), output.max().item()
    
    if ground_truth_mask is not None:
        if ground_truth_mask.dim() == 3 and ground_truth_mask.shape[0] > 1:
            mask_to_analyze = ground_truth_mask[-1]  # Take last channel for analysis
        else:
            mask_to_analyze = ground_truth_mask
        mask_min, mask_max = mask_to_analyze.min().item(), mask_to_analyze.max().item()
    
    print(f"Data ranges - Original: [{orig_min:.3f}, {orig_max:.3f}], Normalized: [{norm_min:.3f}, {norm_max:.3f}], Output: [{out_min:.3f}, {out_max:.3f}]")
    
    # Plot original input frames
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Original Input - 3 Radar Frames', fontsize=16)
    
    for i in range(3):
        im = axes[i].imshow(original_sample[0, i].cpu().numpy(), cmap='viridis', vmin=orig_min, vmax=orig_max)
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('input_output/original_input_frames.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved input_output/original_input_frames.png")
    
    # Plot normalized input frames
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Normalized Input - 3 Radar Frames', fontsize=16)
    
    for i in range(3):
        im = axes[i].imshow(normalized_sample[0, i].cpu().numpy(), cmap='viridis', vmin=norm_min, vmax=norm_max)
        axes[i].set_title(f'Normalized Frame {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('input_output/normalized_input_frames.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved input_output/normalized_input_frames.png")
    
    # Plot final output
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(output[0, 0].cpu().numpy(), cmap='viridis', vmin=-10, vmax=out_max)
    ax.set_title('Final Output - Sea Clutter Suppression Result', fontsize=16)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('input_output/final_output.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved input_output/final_output.png")
    
    # Plot ground truth mask if available
    if ground_truth_mask is not None:
        # Handle multi-channel masks by taking the first channel or average
        if ground_truth_mask.dim() == 3 and ground_truth_mask.shape[0] > 1:
            mask_to_plot = ground_truth_mask[-1]  # Take last channel
        else:
            mask_to_plot = ground_truth_mask
            
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(mask_to_plot.cpu().numpy(), cmap='viridis', vmin=mask_min, vmax=mask_max)
        ax.set_title('Ground Truth Mask', fontsize=16)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('input_output/ground_truth_mask.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  → Saved input_output/ground_truth_mask.png")
    
    # Plot side-by-side comparison
    if ground_truth_mask is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Complete Input → Output Pipeline with Ground Truth', fontsize=20)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Complete Input → Output Pipeline', fontsize=20)
    
    # Top row: Original frames
    axes[0, 0].imshow(original_sample[0, 0].cpu().numpy(), cmap='viridis', vmin=orig_min, vmax=orig_max)
    axes[0, 0].set_title('Original Frame 1', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_sample[0, 1].cpu().numpy(), cmap='viridis', vmin=orig_min, vmax=orig_max)
    axes[0, 1].set_title('Original Frame 2', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(original_sample[0, 2].cpu().numpy(), cmap='viridis', vmin=orig_min, vmax=orig_max)
    axes[0, 2].set_title('Original Frame 3', fontsize=14)
    axes[0, 2].axis('off')
    
    if ground_truth_mask is not None:
        # Handle multi-channel masks
        if ground_truth_mask.dim() == 3 and ground_truth_mask.shape[0] > 1:
            mask_to_plot = ground_truth_mask[-1]  # Take last channel
        else:
            mask_to_plot = ground_truth_mask
            
        axes[0, 3].imshow(mask_to_plot.cpu().numpy(), cmap='viridis', vmin=mask_min, vmax=mask_max)
        axes[0, 3].set_title('Ground Truth Mask', fontsize=14)
        axes[0, 3].axis('off')
    
    # Bottom row: Normalized frames and output
    axes[1, 0].imshow(normalized_sample[0, 0].cpu().numpy(), cmap='viridis', vmin=norm_min, vmax=norm_max)
    axes[1, 0].set_title('Normalized Frame 1', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalized_sample[0, 1].cpu().numpy(), cmap='viridis', vmin=norm_min, vmax=norm_max)
    axes[1, 1].set_title('Normalized Frame 2', fontsize=14)
    axes[1, 1].axis('off')
    
    if ground_truth_mask is not None:
        axes[1, 2].imshow(normalized_sample[0, 2].cpu().numpy(), cmap='viridis', vmin=norm_min, vmax=norm_max)
        axes[1, 2].set_title('Normalized Frame 3', fontsize=14)
        axes[1, 2].axis('off')
        
        im = axes[1, 3].imshow(output[0, 0].cpu().numpy(), cmap='viridis', vmin=-10, vmax=out_max)
        axes[1, 3].set_title('OUTPUT: Clutter Suppressed', fontsize=14, fontweight='bold')
        axes[1, 3].axis('off')
    else:
        im = axes[1, 2].imshow(output[0, 0].cpu().numpy(), cmap='viridis', vmin=-10, vmax=out_max)
        axes[1, 2].set_title('OUTPUT: Clutter Suppressed', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('input_output/complete_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved input_output/complete_pipeline.png")

def plot_all_activations_comprehensive(model, sample, mask=None):
    """Plot ALL channels from ALL activation layers."""
    os.makedirs('activations_complete', exist_ok=True)
    
    # Normalize sample
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    
    normalized_sample = sample.clone()
    for i in range(sample.shape[1]):
        frame = sample[0, i]
        mean = frame.mean()
        std = frame.std()
        if std > 0:
            normalized_sample[0, i] = (frame - mean) / std
        else:
            normalized_sample[0, i] = frame - mean
    
    # Forward pass with intermediates
    with torch.no_grad():
        intermediates = {}
        
        # Encoder
        x1 = model.enc1(normalized_sample)
        intermediates['enc1'] = x1
        
        x2 = model.enc2(model.pool(x1))
        intermediates['enc2'] = x2
        
        x3 = model.enc3(model.pool(x2))
        intermediates['enc3'] = x3
        
        # Decoder
        up2 = model.up2(x3)
        intermediates['up2'] = up2
        
        cat2 = torch.cat([up2, x2], dim=1)
        dec2 = model.dec2(cat2)
        intermediates['dec2'] = dec2
        
        up1 = model.up1(dec2)
        intermediates['up1'] = up1
        
        cat1 = torch.cat([up1, x1], dim=1)
        dec1 = model.dec1(cat1)
        intermediates['dec1'] = dec1
        
        output = model.out_conv(dec1)
        intermediates['output'] = output
    
    # Plot input and output images
    plot_input_output(sample, normalized_sample, output, mask)
    
    # Plot all activations
    stages = ['enc1', 'enc2', 'enc3', 'up2', 'dec2', 'up1', 'dec1', 'output']
    
    # Print activation range info (but don't use for consistent scaling)
    all_activations = []
    for stage in stages:
        activation = intermediates[stage][0]  # Remove batch dimension
        all_activations.append(activation.cpu().numpy().flatten())
    
    all_activations = np.concatenate(all_activations)
    act_min, act_max = all_activations.min(), all_activations.max()
    
    print(f"Activation range: [{act_min:.3f}, {act_max:.3f}]")
    
    for stage in stages:
        activation = intermediates[stage][0]  # Remove batch dimension
        n_channels = activation.shape[0]
        
        print(f"Plotting {stage}: {n_channels} channels, shape {activation.shape}")
        
        # Determine grid size
        if n_channels == 1:
            cols, rows = 1, 1
        elif n_channels <= 16:
            cols = 4
            rows = (n_channels + cols - 1) // cols
        elif n_channels <= 64:
            cols = 8
            rows = (n_channels + cols - 1) // cols
        elif n_channels <= 256:
            cols = 16
            rows = (n_channels + cols - 1) // cols
        else:
            cols = 20
            rows = (n_channels + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        fig.suptitle(f'{stage} - ALL {n_channels} channels - Shape: {activation.shape}', fontsize=16)
        
        # Handle subplot indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            if i < n_channels:
                img = activation[i].cpu().numpy()
                
                # Use auto-scaling for each activation (no log scale, no global vmin/vmax)
                im = ax.imshow(img, cmap='viridis')
                ax.set_title(f'Ch{i}', fontsize=8)
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'activations_complete/{stage}_ALL_{n_channels}_channels.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved activations_complete/{stage}_ALL_{n_channels}_channels.png")

def main():
    print("Loading model and dataset...")
    
    # Load model
    model = UNet(n_channels=3, n_classes=1, base_filters=64)
    checkpoint = torch.load('pretrained/behemoth.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load dataset
    dataset = torch.load('/Users/pepijnlens/Documents/SeaClutterSuppression/local_data/16SNR_clutter.pt', map_location='cpu')
    
    # Get a random sample from the dataset
    import random
    
    if isinstance(dataset, dict):
        # Your dataset has keys: sequences, masks, labels, meta_data
        if 'sequences' in dataset:
            n_samples = len(dataset['sequences'])
            random_idx = 5000
            sample = dataset['sequences'][random_idx]
            mask = dataset['masks'][random_idx] if 'masks' in dataset else None
            label = dataset['labels'][random_idx] if 'labels' in dataset else None
            meta = dataset['meta_data'][random_idx] if 'meta_data' in dataset else None
            
            print(f"Selected random sample {random_idx} out of {n_samples}")
            if mask is not None:
                print(f"Sample has corresponding mask: {mask.shape}")
            if label is not None:
                print(f"Sample label: {label}")
            if meta is not None:
                print(f"Sample metadata: {meta}")
        else:
            # Fallback to previous logic
            sample = list(dataset.values())[0][0]
    else:
        # If dataset is a tensor
        n_samples = len(dataset)
        random_idx = random.randint(0, n_samples - 1)
        sample = dataset[random_idx]
        print(f"Selected random sample {random_idx} out of {n_samples}")
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Sample shape: {sample.shape}")
    
    print("\n" + "="*60)
    print("PLOTTING ALL KERNELS FROM ALL LAYERS")
    print("="*60)
    plot_all_kernels_comprehensive(model)
    
    # Analyze kernel weights in detail
    analyze_kernel_weights(model)
    
    print("\n" + "="*60)
    print("PLOTTING ALL CHANNELS FROM ALL ACTIVATIONS") 
    print("="*60)
    plot_all_activations_comprehensive(model, sample, mask)
    
    print("\n" + "="*60)
    print("COMPLETE! Check these folders:")
    print("  - input_output/         (input frames and final output)")
    print("  - kernels_complete/     (all kernels from all layers)")
    print("  - activations_complete/ (all channels from all activations)")
    print("="*60)

if __name__ == "__main__":
    main()
