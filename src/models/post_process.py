import numpy as np
from scipy.ndimage import binary_dilation

class TrafficOccupancyEvaluator:
    """
    Implements the spatial filtering and traffic occupancy percentage estimation
    algorithm using mathematical morphology (Binary Dilation with N=5 iterations, K=3x3).
    """
    def __init__(self, road_id: int, vehicle_classes: list[int], dilation_iterations: int = 5):
        self.road_id = road_id
        self.vehicle_classes = vehicle_classes
        self.dilation_iterations = dilation_iterations
        # Define 3x3 structuring element K (cross or square connectivity)
        self.structuring_element = np.ones((3, 3), dtype=bool)

    def compute_occupancy(self, pred_mask: np.ndarray) -> tuple[float, dict[int, int]]:
        """
        Computes the spatially filtered traffic occupancy percentage (tau) from a predicted segmentation mask.
        
        Args:
            pred_mask (np.ndarray): Predicted class ID mask of shape (H, W).
            
        Returns:
            tau (float): Filtered traffic occupancy percentage in [0, 100].
            class_counts (dict): Pixel counts for each vehicle category intersecting the road buffer.
        """
        # Step 1: Extract road binary mask
        b_road = (pred_mask == self.road_id)
        n_road = np.sum(b_road)
        
        # Step 2: Construct spatial buffer zone via N successive dilations
        r_dilated = binary_dilation(
            b_road, 
            structure=self.structuring_element, 
            iterations=self.dilation_iterations
        )
        
        n_vehicle = 0
        class_counts = {}
        
        # Step 3: Evaluate vehicle categories within the dilated road buffer
        for v in self.vehicle_classes:
            b_v = (pred_mask == v)
            b_on_road = np.logical_and(b_v, r_dilated)
            
            count = int(np.sum(b_on_road))
            class_counts[v] = count
            n_vehicle += count
            
        # Step 4: Total drivable surface calculation
        total_drivable = int(n_road + n_vehicle)
        
        # Step 5: Bounded occupancy percentage calculation
        if total_drivable == 0:
            tau = 0.0
        else:
            tau = (n_vehicle / total_drivable) * 100.0
            
        return float(tau), class_counts





