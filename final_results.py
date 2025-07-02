import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_final_results():
    """
    Create recall vs SNR plots comparing CFAR and U-Net performance
    across different conditions
    """
    
    # Data for no sea clutter condition
    no_clutter_data = {
        'SNR': [6, 8, 10, 12, 16, 20],
        'CFAR_1e-4_recall': [0.032, 0.094, 0.265, 0.576, 0.989, 0.995],
        'CFAR_1e-4_pfa': [1.3e-4, 1.3e-4, 1.3e-4, 1.3e-4, 1e-4, 1e-4],
        'CFAR_1e-3_recall': [0.112, 0.237, 0.494, 0.785, 0.991, 0.995],
        'CFAR_1e-3_pfa': [1.1e-3, 1.1e-3, 1.1e-3, 1e-3, 1e-3, 1e-3],
        'UNet_1e-1_recall': [0.054, 0.145, 0.381, 0.721, 0.992, 0.992],
        'UNet_1e-1_pfa': [1.9e-4, 1.9e-4, 1.9e-4, 2e-4, 2e-4, 2.4e-4],
        'UNet_1e-6_recall': [0.112, 0.260, 0.53, 0.799, 0.984, 0.985],
        'UNet_1e-6_pfa': [7.4e-4, 7.5e-4, 7.2e-4, 7.4e-4, 7.9e-4, 9.3e-4]
    }
    
    # Data for with sea clutter condition
    with_clutter_data = {
        'SNR': [6, 8, 10, 12, 16, 20],
        'CFAR_1e-4_recall': [0.047, 0.075, 0.191, 0.401, 0.843, 0.940],
        'CFAR_1e-4_pfa': [1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3],
        'CFAR_1e-3_recall': [0.106, 0.175, 0.374, 0.603, 0.892, 0.962],
        'CFAR_1e-3_pfa': [3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3],
        'UNet_1e-1_recall': [0.023, 0.073, 0.244, 0.553, 0.924, 0.972],
        'UNet_1e-1_pfa': [7.8e-5, 7.8e-5, 8.3e-5, 8.4e-5, 8.6e-5, 9.4e-5],
        'UNet_1e-6_recall': [0.070, 0.162, 0.382, 0.688, 0.939, 0.976],
        'UNet_1e-6_pfa': [6.2e-4, 6.2e-4, 6.2e-4, 6.1e-4, 6.4e-4, 7e-4]
    }
    
    # Data for random sea clutter configurations
    random_clutter_data = {
        'CFAR_1e-4_recall': 0.684,
        'CFAR_1e-4_pfa': 1.7e-3,
        'CFAR_1e-3_recall': 0.778,
        'CFAR_1e-3_pfa': 3.5e-3,
        'UNet_1e-1_recall': 0.771,
        'UNet_1e-1_pfa': 9.4e-5,
        'UNet_1e-6_recall': 0.828,
        'UNet_1e-6_pfa': 6.5e-4
    }

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 6))
    
    # Define colors for consistency
    colors = {
        'CFAR_1e-4': '#1f77b4',
        'CFAR_1e-3': '#ff7f0e', 
        'UNet_1e-1': '#2ca02c',
        'UNet_1e-6': '#d62728'
    }
    
    # Plot 1: Recall vs SNR for no clutter
    ax1 = plt.subplot(1, 2, 1)
    
    # No clutter
    ax1.plot(no_clutter_data['SNR'], no_clutter_data['CFAR_1e-4_recall'], 
             'o--', color=colors['CFAR_1e-4'], linewidth=2, markersize=8, 
             label='CFAR (P_FA=1e-4)')
    ax1.plot(no_clutter_data['SNR'], no_clutter_data['CFAR_1e-3_recall'], 
             's--', color=colors['CFAR_1e-3'], linewidth=2, markersize=8, 
             label='CFAR (P_FA=1e-3)')
    ax1.plot(no_clutter_data['SNR'], no_clutter_data['UNet_1e-1_recall'], 
             '^-', color=colors['UNet_1e-1'], linewidth=2, markersize=8, 
             label='U-Net (thresh=1e-1)')
    ax1.plot(no_clutter_data['SNR'], no_clutter_data['UNet_1e-6_recall'], 
             'v-', color=colors['UNet_1e-6'], linewidth=2, markersize=8, 
             label='U-Net (thresh=1e-6)')
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Recall', fontsize=12)
    ax1.set_title('Recall vs SNR (No Sea Clutter)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Recall vs SNR with sea clutter
    ax2 = plt.subplot(1, 2, 2)
    
    ax2.plot(with_clutter_data['SNR'], with_clutter_data['CFAR_1e-4_recall'], 
             'o--', color=colors['CFAR_1e-4'], linewidth=2, markersize=8, 
             label='CFAR (P_FA=1e-4)')
    ax2.plot(with_clutter_data['SNR'], with_clutter_data['CFAR_1e-3_recall'], 
             's--', color=colors['CFAR_1e-3'], linewidth=2, markersize=8, 
             label='CFAR (P_FA=1e-3)')
    ax2.plot(with_clutter_data['SNR'], with_clutter_data['UNet_1e-1_recall'], 
             '^-', color=colors['UNet_1e-1'], linewidth=2, markersize=8, 
             label='U-Net (thresh=1e-1)')
    ax2.plot(with_clutter_data['SNR'], with_clutter_data['UNet_1e-6_recall'], 
             'v-', color=colors['UNet_1e-6'], linewidth=2, markersize=8, 
             label='U-Net (thresh=1e-6)')
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Recall vs SNR (With Sea Clutter)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('/Users/pepijnlens/Documents/SeaClutterSuppression/evaluation_results/recall_vs_snr_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_histograms():
    """Create histogram plots comparing performance across different conditions"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Data preparation
    methods = ['CFAR\n(P_FA=1e-4)', 'CFAR\n(P_FA=1e-3)', 'U-Net\n(thresh=1e-1)', 'U-Net\n(thresh=1e-6)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Average Recall across SNR levels (No Clutter)
    no_clutter_avg_recall = [
        np.mean([0.032, 0.094, 0.265, 0.576, 0.989, 0.995]),  # CFAR 1e-4
        np.mean([0.112, 0.237, 0.494, 0.785, 0.991, 0.995]),  # CFAR 1e-3
        np.mean([0.054, 0.145, 0.381, 0.721, 0.992, 0.992]),  # U-Net 1e-1
        np.mean([0.112, 0.260, 0.53, 0.799, 0.984, 0.985])    # U-Net 1e-6
    ]
    
    bars1 = ax1.bar(methods, no_clutter_avg_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Recall', fontsize=12)
    ax1.set_title('Average Recall (No Sea Clutter)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, no_clutter_avg_recall):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average Recall across SNR levels (With Clutter)
    with_clutter_avg_recall = [
        np.mean([0.047, 0.075, 0.191, 0.401, 0.843, 0.940]),  # CFAR 1e-4
        np.mean([0.106, 0.175, 0.374, 0.603, 0.892, 0.962]),  # CFAR 1e-3
        np.mean([0.023, 0.073, 0.244, 0.553, 0.924, 0.972]),  # U-Net 1e-1
        np.mean([0.070, 0.162, 0.382, 0.688, 0.939, 0.976])   # U-Net 1e-6
    ]
    
    bars2 = ax2.bar(methods, with_clutter_avg_recall, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Average Recall', fontsize=12)
    ax2.set_title('Average Recall (With Sea Clutter)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars2, with_clutter_avg_recall):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. False Alarm Rate Comparison (logarithmic scale)
    no_clutter_avg_pfa = [
        np.mean([1.3e-4, 1.3e-4, 1.3e-4, 1.3e-4, 1e-4, 1e-4]),      # CFAR 1e-4
        np.mean([1.1e-3, 1.1e-3, 1.1e-3, 1e-3, 1e-3, 1e-3]),        # CFAR 1e-3
        np.mean([1.9e-4, 1.9e-4, 1.9e-4, 2e-4, 2e-4, 2.4e-4]),      # U-Net 1e-1
        np.mean([7.4e-4, 7.5e-4, 7.2e-4, 7.4e-4, 7.9e-4, 9.3e-4])   # U-Net 1e-6
    ]
    with_clutter_avg_pfa = [
        np.mean([1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3]),  # CFAR 1e-4
        np.mean([3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3, 3.4e-3]),  # CFAR 1e-3
        np.mean([7.8e-5, 7.8e-5, 8.3e-5, 8.4e-5, 8.6e-5, 9.4e-5]),  # U-Net 1e-1
        np.mean([6.2e-4, 6.2e-4, 6.2e-4, 6.1e-4, 6.4e-4, 7e-4])     # U-Net 1e-6
    ]

    x = np.arange(len(methods))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, no_clutter_avg_pfa, width, label='No Clutter', 
                     color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars3b = ax3.bar(x + width/2, with_clutter_avg_pfa, width, label='With Clutter', 
                     color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax3.set_ylabel('False Alarm Rate (log scale)', fontsize=12)
    ax3.set_title('False Alarm Rate Comparison', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Random Sea Clutter Performance
    random_recall = [0.684, 0.778, 0.771, 0.828]
    random_pfa = [1.7e-3, 3.5e-3, 9.4e-5, 6.5e-4]
    
    # Create scatter plot with recall vs PFA
    scatter = ax4.scatter(random_pfa, random_recall, c=colors, s=200, alpha=0.8, 
                         edgecolors='black', linewidth=2)
    
    # Add method labels
    for i, method in enumerate(['CFAR (1e-4)', 'CFAR (1e-3)', 'U-Net (1e-1)', 'U-Net (1e-6)']):
        ax4.annotate(method, (random_pfa[i], random_recall[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    ax4.set_xlabel('False Alarm Rate (log scale)', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Random Sea Clutter: Recall vs False Alarm Rate', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.6, 0.85)
    
    plt.tight_layout()
    plt.savefig('/Users/pepijnlens/Documents/SeaClutterSuppression/evaluation_results/performance_histograms.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating recall vs SNR plots...")
    plot_final_results()
    print("Recall vs SNR plots saved to evaluation_results/recall_vs_snr_results.png")
    
    print("\nGenerating performance histograms...")
    create_summary_histograms()
    print("Performance histograms saved to evaluation_results/performance_histograms.png")
    
    print("\nAll plots generated successfully!")
