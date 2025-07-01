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
        'UNet_1e-6_pfa': [6.2e-4, 6.2e-4, 6.2e-4, 6.1e-4, 1.2e-5, 7e-4]
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

def create_summary_table():
    """Create a summary table of all results"""
    
    # Create comprehensive summary table
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Table 1: No Sea Clutter
    no_clutter_table_data = [
        ['SNR (dB)', 'CFAR (P_FA=1e-4)', 'CFAR (P_FA=1e-3)', 'U-Net (thresh=1e-1)', 'U-Net (thresh=1e-6)'],
        ['', 'Recall | P_FA', 'Recall | P_FA', 'Recall | P_FA', 'Recall | P_FA'],
        ['6', '0.032 | 1.3e-4', '0.112 | 1.1e-3', '0.054 | 1.9e-4', '0.112 | 7.4e-4'],
        ['8', '0.094 | 1.3e-4', '0.237 | 1.1e-3', '0.145 | 1.9e-4', '0.260 | 7.5e-4'],
        ['10', '0.265 | 1.3e-4', '0.494 | 1.1e-3', '0.381 | 1.9e-4', '0.530 | 7.2e-4'],
        ['12', '0.576 | 1.3e-4', '0.785 | 1.0e-3', '0.721 | 2.0e-4', '0.799 | 7.4e-4'],
        ['16', '0.989 | 1.0e-4', '0.991 | 1.0e-3', '0.992 | 2.0e-4', '0.984 | 7.9e-4'],
        ['20', '0.995 | 1.0e-4', '0.995 | 1.0e-3', '0.992 | 2.4e-4', '0.985 | 9.3e-4']
    ]
    
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=no_clutter_table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    ax1.set_title('Performance Results - No Sea Clutter', fontsize=14, fontweight='bold', pad=20)
    
    # Color header rows
    for i in range(5):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(1, i)].set_facecolor('#E8F5E8')
    
    # Table 2: With Sea Clutter
    with_clutter_table_data = [
        ['SNR (dB)', 'CFAR (P_FA=1e-4)', 'CFAR (P_FA=1e-3)', 'U-Net (thresh=1e-1)', 'U-Net (thresh=1e-6)'],
        ['', 'Recall | P_FA', 'Recall | P_FA', 'Recall | P_FA', 'Recall | P_FA'],
        ['6', '0.047 | 1.6e-3', '0.106 | 3.4e-3', '0.023 | 7.8e-5', '0.070 | 6.2e-4'],
        ['8', '0.075 | 1.6e-3', '0.175 | 3.4e-3', '0.073 | 7.8e-5', '0.162 | 6.2e-4'],
        ['10', '0.191 | 1.6e-3', '0.374 | 3.4e-3', '0.244 | 8.3e-5', '0.382 | 6.2e-4'],
        ['12', '0.401 | 1.6e-3', '0.603 | 3.4e-3', '0.553 | 8.4e-5', '0.688 | 6.1e-4'],
        ['16', '0.843 | 1.6e-3', '0.892 | 3.4e-3', '0.924 | 8.6e-5', '0.939 | 1.2e-5'],
        ['20', '0.940 | 1.6e-3', '0.962 | 3.4e-3', '0.972 | 9.4e-5', '0.976 | 7.0e-4']
    ]
    
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=with_clutter_table_data, cellLoc='center', loc='center',
                      colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    ax2.set_title('Performance Results - With Sea Clutter', fontsize=14, fontweight='bold', pad=20)
    
    # Color header rows
    for i in range(5):
        table2[(0, i)].set_facecolor('#FF9800')
        table2[(1, i)].set_facecolor('#FFF3E0')
    
    # Table 3: Random Sea Clutter
    random_table_data = [
        ['Method', 'CFAR (P_FA=1e-4)', 'CFAR (P_FA=1e-3)', 'U-Net (thresh=1e-1)', 'U-Net (thresh=1e-6)'],
        ['Recall | P_FA', '0.684 | 1.7e-3', '0.778 | 3.5e-3', '0.771 | 9.4e-5', '0.828 | 6.5e-4']
    ]
    
    ax3.axis('tight')
    ax3.axis('off')
    table3 = ax3.table(cellText=random_table_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2)
    ax3.set_title('Performance Results - Random Sea Clutter Configurations', fontsize=14, fontweight='bold', pad=20)
    
    # Color header row
    for i in range(5):
        table3[(0, i)].set_facecolor('#2196F3')
    
    plt.tight_layout()
    plt.savefig('/Users/pepijnlens/Documents/SeaClutterSuppression/evaluation_results/results_summary_table.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating recall vs SNR plots...")
    plot_final_results()
    print("Recall vs SNR plots saved to evaluation_results/recall_vs_snr_results.png")
    
    print("\nGenerating summary table...")
    create_summary_table()
    print("Summary table saved to evaluation_results/results_summary_table.png")
    
    print("\nAll plots generated successfully!")
