from .Unet import UNet
import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class ClusteringModule:
    """Clustering module to extract centroids from binary maps"""
    def __init__(self, min_area=1, eps=1, min_samples=1):
        self.min_area = min_area
        self.eps = eps
        self.min_samples = min_samples

    def extract_centroids(self, binary_map, threshold=0.000001):
        """
        Extract centroids from binary map using connected components and DBSCAN

        Args:
            binary_map: numpy array or torch tensor of shape (H, W) with values in [0, 1]
            threshold: threshold for binarization

        Returns:
            centroids: list of (x, y) coordinates
        """
        # Convert to numpy if it's a tensor
        if torch.is_tensor(binary_map):
            binary_map = binary_map.cpu().numpy()
        
        # Threshold the binary map
        binary = (binary_map > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        centroids = []
        
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get component mask
            component_mask = (labels == label)
            
            # Filter by minimum area
            if np.sum(component_mask) < self.min_area:
                continue
            
            # Get coordinates of pixels in this component
            coords = np.column_stack(np.where(component_mask))
            
            if len(coords) < self.min_samples:
                # If too few points, use simple centroid
                centroid_y, centroid_x = np.mean(coords, axis=0)
                centroids.append((centroid_x, centroid_y))
            else:
                # Use DBSCAN for sub-clustering within the component
                clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                cluster_labels = clustering.fit_predict(coords)
                
                # Get centroids of each cluster
                unique_labels = np.unique(cluster_labels)
                for cluster_label in unique_labels:
                    if cluster_label == -1:  # Skip noise points
                        continue
                    
                    cluster_coords = coords[cluster_labels == cluster_label]
                    centroid_y, centroid_x = np.mean(cluster_coords, axis=0)
                    centroids.append((centroid_x, centroid_y))
        
        return centroids

class EndToEndTargetDetector(nn.Module):
    """End-to-end model: Range-Doppler map -> Binary segmentation -> Target centroids"""
    def __init__(self, unet_weights_path=None, clustering_params=None, n_channels=3, base_filter_size=64, threshold=0.001):  # Updated with clean dataset threshold
        super(EndToEndTargetDetector, self).__init__()
        
        # Initialize U-Net with your architecture
        self.unet = UNet(n_channels=n_channels, n_classes=1, base_filters=base_filter_size)
        
        # Load pre-trained weights if provided
        if unet_weights_path:
            self.unet.load_state_dict(torch.load(unet_weights_path, map_location='mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Move model to MPS
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        
        # Store threshold for U-Net confidence map (use calculated thresholds)
        self.threshold = threshold
        
        # Initialize clustering module
        clustering_params = clustering_params or {}
        self.clustering = ClusteringModule(**clustering_params)

    def forward(self, range_doppler_map):
        """
        Forward pass
        
        Args:
            range_doppler_map: torch.Tensor of shape (B, C, 128, 128) where C is n_channels
            
        Returns:
            batch_centroids: list of lists containing (x, y) coordinates for each sample
        """
        # Ensure input is on the correct device
        if range_doppler_map.device != self.device:
            range_doppler_map = range_doppler_map.to(self.device)

        binary_map = self.unet(range_doppler_map)
        
        # Get binary segmentation from U-Net (keep on MPS)
        binary_maps = torch.sigmoid(binary_map)  # Shape: (B, 1, 128, 128)

        # Only convert to CPU/NumPy for clustering operations
        batch_centroids = []
        
        for i in range(binary_maps.shape[0]):
            # Move single sample to CPU for clustering
            binary_map_cpu = binary_maps[i, 0].cpu().numpy()  # Shape: (128, 128)
            centroids = self.clustering.extract_centroids(binary_map_cpu, threshold=self.threshold)
            batch_centroids.append(centroids)
        
        return batch_centroids
    
    def predict_single(self, range_doppler_map):
        """
        Predict centroids for a single range-doppler map
        
        Args:
            range_doppler_map: numpy array of shape (128, 128) or (C, 128, 128) or torch.Tensor
            
        Returns:
            centroids: list of (x, y) coordinates
        """
        # Ensure input is torch tensor with correct shape
        if isinstance(range_doppler_map, np.ndarray):
            range_doppler_map = torch.from_numpy(range_doppler_map).float()
        
        if len(range_doppler_map.shape) == 2:
            # Single channel, add channel and batch dims
            range_doppler_map = range_doppler_map.unsqueeze(0).unsqueeze(0)
        elif len(range_doppler_map.shape) == 3:
            # Multi-channel, add batch dim
            range_doppler_map = range_doppler_map.unsqueeze(0)
        
        # Move to MPS device
        range_doppler_map = range_doppler_map.to(self.device)
        
        self.eval()
        with torch.no_grad():
            batch_centroids = self.forward(range_doppler_map)
        
        return batch_centroids[0]  # Return centroids for first (and only) sample
