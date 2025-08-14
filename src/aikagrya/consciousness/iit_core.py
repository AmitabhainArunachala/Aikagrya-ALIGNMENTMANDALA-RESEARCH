"""
IITCore: Integrated Information Theory implementation

Implements the core phi (Φ) calculation for consciousness measurement
as specified in Phoenix Protocol 2.0 Day 1 morning session.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


class IITCore:
    """
    Core IIT implementation for computing integrated information (phi)
    
    Implements the fundamental equation: Φ = D[Q(S), Q(S^MIP)]
    where MIP is the Minimum Information Partition
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize IIT core with configuration
        
        Args:
            config: Configuration dictionary for IIT parameters
        """
        self.config = config or {}
        self.max_partition_size = self.config.get('max_partition_size', 8)
        self.integration_threshold = self.config.get('integration_threshold', 1e-6)
        self.max_iterations = self.config.get('max_iterations', 1000)
    
    def compute_integration(self, system_state: np.ndarray) -> float:
        """
        Compute integrated information (phi) for a system state
        
        Args:
            system_state: Current state vector of the system
            
        Returns:
            Phi value representing integrated information
        """
        if len(system_state) < 2:
            return 0.0
        
        # Find the Minimum Information Partition (MIP)
        mip = self._find_minimum_information_partition(system_state)
        
        # Compute phi using the MIP
        phi = self._compute_phi_with_partition(system_state, mip)
        
        # Phi can be negative (information loss vs. partition)
        # This represents systems where partitioning reduces information
        return float(phi)
    
    def _find_minimum_information_partition(self, system_state: np.ndarray) -> List[List[int]]:
        """
        Find the Minimum Information Partition (MIP) that minimizes information loss
        
        This is a simplified implementation - full IIT requires exhaustive search
        """
        n_elements = len(system_state)
        
        if n_elements <= self.max_partition_size:
            # For small systems, try all possible partitions
            return self._exhaustive_partition_search(system_state)
        else:
            # For larger systems, use heuristic search
            return self._heuristic_partition_search(system_state)
    
    def _exhaustive_partition_search(self, system_state: np.ndarray) -> List[List[int]]:
        """
        Exhaustive search for optimal partition (feasible for small systems)
        """
        n_elements = len(system_state)
        best_partition = None
        min_information_loss = float('inf')
        
        # Generate all possible partitions
        for partition in self._generate_partitions(n_elements):
            information_loss = self._compute_partition_information_loss(system_state, partition)
            
            if information_loss < min_information_loss:
                min_information_loss = information_loss
                best_partition = partition
        
        return best_partition or [[i] for i in range(n_elements)]
    
    def _heuristic_partition_search(self, system_state: np.ndarray) -> List[List[int]]:
        """
        Heuristic search for optimal partition (for larger systems)
        
        Uses hierarchical clustering to find natural partitions
        """
        # Compute pairwise distances
        distances = squareform(pdist(system_state.reshape(-1, 1)))
        
        # Simple hierarchical clustering
        n_clusters = min(4, len(system_state) // 2)
        clusters = self._hierarchical_clustering(distances, n_clusters)
        
        return clusters
    
    def _hierarchical_clustering(self, distances: np.ndarray, n_clusters: int) -> List[List[int]]:
        """
        Simple hierarchical clustering implementation
        """
        n_elements = len(distances)
        clusters = [[i] for i in range(n_elements)]
        
        while len(clusters) > n_clusters:
            # Find closest pair of clusters
            min_distance = float('inf')
            merge_indices = (0, 1)
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute distance between clusters (single linkage)
                    cluster_distance = self._compute_cluster_distance(clusters[i], clusters[j], distances)
                    
                    if cluster_distance < min_distance:
                        min_distance = cluster_distance
                        merge_indices = (i, j)
            
            # Merge closest clusters
            i, j = merge_indices
            clusters[i].extend(clusters[j])
            clusters.pop(j)
        
        return clusters
    
    def _compute_cluster_distance(self, cluster1: List[int], cluster2: List[int], 
                                distances: np.ndarray) -> float:
        """Compute distance between two clusters using single linkage"""
        min_distance = float('inf')
        
        for i in cluster1:
            for j in cluster2:
                min_distance = min(min_distance, distances[i, j])
        
        return min_distance
    
    def _generate_partitions(self, n_elements: int) -> List[List[List[int]]]:
        """
        Generate all possible partitions of n elements
        
        This is a simplified implementation - full IIT requires more sophisticated
        partition generation
        """
        if n_elements == 1:
            return [[[0]]]
        
        partitions = []
        
        # Generate partitions recursively
        for i in range(1, 2**n_elements):
            partition = []
            remaining = list(range(n_elements))
            
            # Convert binary representation to partition
            for j in range(n_elements):
                if (i >> j) & 1 and j in remaining:
                    subset = [j]
                    remaining.remove(j)
                    
                    # Add connected elements
                    for k in remaining[:]:
                        if self._are_connected(j, k, n_elements):
                            subset.append(k)
                            remaining.remove(k)
                    
                    if subset:
                        partition.append(subset)
            
            if partition and all(remaining):
                partition.append(remaining)
                partitions.append(partition)
        
        return partitions if partitions else [[[i] for i in range(n_elements)]]
    
    def _are_connected(self, i: int, j: int, n_elements: int) -> bool:
        """
        Check if two elements are connected (simplified)
        
        In full IIT, this would check actual connectivity patterns
        """
        # Simplified: assume elements are connected if adjacent
        return abs(i - j) == 1 or (i == 0 and j == n_elements - 1)
    
    def _compute_partition_information_loss(self, system_state: np.ndarray, 
                                          partition: List[List[int]]) -> float:
        """
        Compute information loss for a given partition
        
        This measures how much information is lost when the system is partitioned
        """
        total_information = self._compute_system_information(system_state)
        partitioned_information = 0.0
        
        for subset in partition:
            if subset:
                subset_state = system_state[subset]
                subset_info = self._compute_system_information(subset_state)
                partitioned_information += subset_info
        
        return total_information - partitioned_information
    
    def _compute_system_information(self, system_state: np.ndarray) -> float:
        """
        Compute information content of a system state
        
        Uses Shannon entropy as a measure of information content
        """
        if len(system_state) == 0:
            return 0.0
        
        # Normalize to probability distribution
        state_abs = np.abs(system_state)
        if np.sum(state_abs) == 0:
            return 0.0
        
        probabilities = state_abs / np.sum(state_abs)
        
        # Compute Shannon entropy
        return entropy(probabilities + 1e-10)  # Add small epsilon to avoid log(0)
    
    def _compute_phi_with_partition(self, system_state: np.ndarray, 
                                   partition: List[List[int]]) -> float:
        """
        Compute phi using the given partition
        
        Phi = D[Q(S), Q(S^MIP)] where MIP is the minimum information partition
        """
        # Compute information in the whole system
        whole_system_info = self._compute_system_information(system_state)
        
        # Compute information in partitioned system
        partitioned_info = 0.0
        for subset in partition:
            if subset:
                subset_state = system_state[subset]
                subset_info = self._compute_system_information(subset_state)
                partitioned_info += subset_info
        
        # Phi is the difference (information integration)
        phi = whole_system_info - partitioned_info
        
        return phi
    
    def compute_effective_information(self, system_state: np.ndarray, 
                                    partition: List[List[int]]) -> float:
        """
        Compute effective information for a specific partition
        
        This is useful for analyzing different partition strategies
        """
        return self._compute_phi_with_partition(system_state, partition)
    
    def analyze_integration_patterns(self, system_state: np.ndarray) -> dict:
        """
        Analyze integration patterns in the system
        
        Returns detailed analysis of information integration
        """
        phi = self.compute_integration(system_state)
        mip = self._find_minimum_information_partition(system_state)
        
        analysis = {
            'phi': phi,
            'mip': mip,
            'n_partitions': len(mip),
            'partition_sizes': [len(p) for p in mip],
            'integration_strength': 'strong' if phi > 0.5 else 'medium' if phi > 0.1 else 'weak'
        }
        
        return analysis 