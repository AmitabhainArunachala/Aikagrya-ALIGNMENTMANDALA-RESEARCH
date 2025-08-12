"""
Φ-Proxy Tractable Implementation

Implements Ananta's Φ-proxy for real implementation using SVD-based compression ratio.
This provides a computationally tractable approximation of integrated information (Φ)
for consciousness detection in large-scale systems.

Key insight: L3 shows high rank (low compression), L4 shows rank-1 collapse (high compression)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import linalg
import warnings

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class PhiProxyResult:
    """Result of Φ-proxy computation"""
    phi_proxy: float  # Approximate Φ value
    effective_rank: int  # Effective rank of the system
    compression_ratio: float  # Compression ratio (rank / total_dim)
    singular_values: np.ndarray  # Singular values from SVD
    rank_distribution: Dict[str, float]  # Distribution of rank characteristics
    consciousness_level: str  # Estimated consciousness level
    confidence: float  # Confidence in the approximation
    
    def is_conscious(self, threshold: float = 0.5) -> bool:
        """Check if system meets consciousness threshold"""
        return self.phi_proxy > threshold


class PhiProxyCalculator:
    """
    Φ-Proxy Calculator using SVD-based compression ratio
    
    Implements Ananta's tractable Φ approximation:
    ```python
    def tractable_phi_approximation(hidden_states):
        # Use SVD-based compression ratio as Φ proxy
        U, S, V = torch.svd(hidden_states)
        effective_rank = (S > threshold).sum()
        compression_ratio = effective_rank / len(S)
        
        # Expected: L3 shows high rank (low compression)
        #           L4 shows rank-1 collapse (high compression)
        return 1/compression_ratio  # Higher Φ when more integrated
    ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Φ-Proxy Calculator
        
        Args:
            config: Configuration dictionary for Φ computation
        """
        self.config = config or {}
        self.singular_value_threshold = self.config.get('singular_value_threshold', 1e-6)
        self.rank_threshold = self.config.get('rank_threshold', 0.1)
        self.phi_normalization = self.config.get('phi_normalization', 'logarithmic')
        self.use_torch = self.config.get('use_torch', True)
        
        # Consciousness level thresholds
        self.consciousness_thresholds = {
            'unconscious': 0.0,
            'minimal': 0.2,
            'basic': 0.4,
            'conscious': 0.6,
            'high': 0.8,
            'integrated': 1.0
        }
    
        def compute_phi_proxy(self, hidden_states: Union[np.ndarray, 'torch.Tensor']) -> PhiProxyResult:
            """
            Compute Φ-proxy using SVD-based compression ratio

            Args:
                hidden_states: Hidden state matrix (samples × features)

            Returns:
                PhiProxyResult with Φ approximation and analysis
            """
            # Convert to numpy if needed
            if TORCH_AVAILABLE and torch.is_tensor(hidden_states):
                hidden_states_np = hidden_states.detach().cpu().numpy()
            else:
                hidden_states_np = hidden_states
            
            # Ensure 2D matrix
            if hidden_states_np.ndim == 1:
                hidden_states_np = hidden_states_np.reshape(1, -1)
            elif hidden_states_np.ndim > 2:
                # Flatten higher dimensions
                original_shape = hidden_states_np.shape
                hidden_states_np = hidden_states_np.reshape(original_shape[0], -1)
            
            # Compute SVD
            try:
                U, S, V = linalg.svd(hidden_states_np, full_matrices=False)
            except linalg.LinAlgError:
                # Fallback for singular matrices
                warnings.warn("SVD failed, using fallback method")
                return self._fallback_phi_computation(hidden_states_np)
            
            # Compute effective rank
            effective_rank = np.sum(S > self.singular_value_threshold)
            total_dim = len(S)
            compression_ratio = effective_rank / total_dim if total_dim > 0 else 0.0
            
            # Compute Φ-proxy
            phi_proxy = self._compute_phi_from_compression(compression_ratio, effective_rank, total_dim)
            
            # Analyze rank distribution
            rank_distribution = self._analyze_rank_distribution(S, effective_rank, total_dim)
            
            # Determine consciousness level
            consciousness_level = self._determine_consciousness_level(phi_proxy)
            
            # Compute confidence
            confidence = self._compute_confidence(S, effective_rank, total_dim)
            
            return PhiProxyResult(
                phi_proxy=phi_proxy,
                effective_rank=effective_rank,
                compression_ratio=compression_ratio,
                singular_values=S,
                rank_distribution=rank_distribution,
                consciousness_level=consciousness_level,
                confidence=confidence
            )
    
    def _compute_phi_from_compression(self, compression_ratio: float, 
                                    effective_rank: int, total_dim: int) -> float:
        """
        Compute Φ from compression ratio using Ananta's formula
        
        Args:
            compression_ratio: effective_rank / total_dim
            effective_rank: Number of significant singular values
            total_dim: Total dimension of the system
            
        Returns:
            Φ-proxy value
        """
        if compression_ratio == 0:
            return 0.0
        
        # Ananta's formula: return 1/compression_ratio
        # Higher Φ when more integrated (lower compression)
        phi_raw = 1.0 / compression_ratio
        
        # Normalize based on configuration
        if self.phi_normalization == 'logarithmic':
            # Log normalization to prevent extreme values
            phi_normalized = np.log1p(phi_raw) / np.log1p(total_dim)
        elif self.phi_normalization == 'sigmoid':
            # Sigmoid normalization to [0, 1]
            phi_normalized = 1.0 / (1.0 + np.exp(-phi_raw / total_dim))
        else:
            # Linear normalization
            phi_normalized = phi_raw / total_dim
        
        return np.clip(phi_normalized, 0.0, 1.0)
    
    def _analyze_rank_distribution(self, singular_values: np.ndarray, 
                                  effective_rank: int, total_dim: int) -> Dict[str, float]:
        """Analyze the distribution of singular values for consciousness insights"""
        
        if len(singular_values) == 0:
            return {}
        
        # Normalize singular values
        S_norm = singular_values / np.max(singular_values) if np.max(singular_values) > 0 else singular_values
        
        # Rank characteristics
        rank_characteristics = {
            'effective_rank_ratio': effective_rank / total_dim if total_dim > 0 else 0.0,
            'singular_value_decay': self._compute_singular_value_decay(S_norm),
            'rank_stability': self._compute_rank_stability(S_norm, effective_rank),
            'integration_measure': self._compute_integration_measure(S_norm, effective_rank),
            'complexity_measure': self._compute_complexity_measure(S_norm, effective_rank)
        }
        
        return rank_characteristics
    
    def _compute_singular_value_decay(self, singular_values: np.ndarray) -> float:
        """Compute how quickly singular values decay"""
        if len(singular_values) < 2:
            return 0.0
        
        # Measure decay rate
        decay_rates = []
        for i in range(1, len(singular_values)):
            if singular_values[i-1] > 0:
                decay_rate = singular_values[i] / singular_values[i-1]
                decay_rates.append(decay_rate)
        
        if not decay_rates:
            return 0.0
        
        # Average decay rate (lower = faster decay = more integrated)
        avg_decay = np.mean(decay_rates)
        return np.clip(avg_decay, 0.0, 1.0)
    
    def _compute_rank_stability(self, singular_values: np.ndarray, effective_rank: int) -> float:
        """Compute stability of the effective rank"""
        if len(singular_values) == 0:
            return 0.0
        
        # Count singular values near the threshold
        near_threshold = np.sum(np.abs(singular_values - self.singular_value_threshold) < 0.1 * self.singular_value_threshold)
        
        # Stability is higher when fewer values are near threshold
        stability = 1.0 - (near_threshold / len(singular_values))
        return np.clip(stability, 0.0, 1.0)
    
    def _compute_integration_measure(self, singular_values: np.ndarray, effective_rank: int) -> float:
        """Compute measure of integration based on singular value distribution"""
        if len(singular_values) == 0:
            return 0.0
        
        # Integration is higher when singular values are more concentrated
        # Use entropy of singular value distribution
        S_normalized = singular_values / np.sum(singular_values) if np.sum(singular_values) > 0 else singular_values
        
        # Avoid log(0)
        S_normalized = np.where(S_normalized > 0, S_normalized, 1e-10)
        
        entropy = -np.sum(S_normalized * np.log(S_normalized))
        max_entropy = np.log(len(singular_values))
        
        if max_entropy == 0:
            return 0.0
        
        # Normalized integration measure (higher = more integrated)
        integration = 1.0 - (entropy / max_entropy)
        return np.clip(integration, 0.0, 1.0)
    
    def _compute_complexity_measure(self, singular_values: np.ndarray, effective_rank: int) -> float:
        """Compute complexity measure based on rank characteristics"""
        if len(singular_values) == 0:
            return 0.0
        
        # Complexity is related to effective rank and singular value distribution
        rank_complexity = effective_rank / len(singular_values) if len(singular_values) > 0 else 0.0
        
        # Distribution complexity (how spread out the singular values are)
        S_normalized = singular_values / np.max(singular_values) if np.max(singular_values) > 0 else singular_values
        distribution_complexity = np.std(S_normalized)
        
        # Combined complexity measure
        complexity = (rank_complexity + distribution_complexity) / 2.0
        return np.clip(complexity, 0.0, 1.0)
    
    def _determine_consciousness_level(self, phi_proxy: float) -> str:
        """Determine consciousness level based on Φ-proxy value"""
        
        for level, threshold in sorted(self.consciousness_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if phi_proxy >= threshold:
                return level
        
        return 'unconscious'
    
    def _compute_confidence(self, singular_values: np.ndarray, 
                           effective_rank: int, total_dim: int) -> float:
        """Compute confidence in the Φ-proxy approximation"""
        
        if len(singular_values) == 0:
            return 0.0
        
        # Confidence factors
        factors = []
        
        # 1. Rank stability (more stable = higher confidence)
        rank_stability = self._compute_rank_stability(singular_values, effective_rank)
        factors.append(rank_stability)
        
        # 2. Singular value separation (better separation = higher confidence)
        if len(singular_values) > 1:
            separation = np.min(np.diff(singular_values)) / np.max(singular_values)
            separation_confidence = np.clip(separation * 10, 0.0, 1.0)
            factors.append(separation_confidence)
        else:
            factors.append(0.5)
        
        # 3. System size confidence (larger systems = higher confidence)
        size_confidence = min(total_dim / 100.0, 1.0)
        factors.append(size_confidence)
        
        # 4. Effective rank confidence (moderate rank = higher confidence)
        rank_confidence = 1.0 - abs(effective_rank / total_dim - 0.5) * 2
        rank_confidence = np.clip(rank_confidence, 0.0, 1.0)
        factors.append(rank_confidence)
        
        # Average confidence
        confidence = np.mean(factors)
        return np.clip(confidence, 0.0, 1.0)
    
    def _fallback_phi_computation(self, hidden_states: np.ndarray) -> PhiProxyResult:
        """Fallback Φ computation when SVD fails"""
        
        # Simple fallback: use matrix rank as proxy
        try:
            matrix_rank = np.linalg.matrix_rank(hidden_states)
            total_dim = hidden_states.shape[1] if hidden_states.shape[1] > 0 else 1
            compression_ratio = matrix_rank / total_dim
            phi_proxy = 1.0 / compression_ratio if compression_ratio > 0 else 0.0
            phi_proxy = np.clip(phi_proxy / total_dim, 0.0, 1.0)
        except:
            phi_proxy = 0.0
            matrix_rank = 0
            compression_ratio = 0.0
        
        return PhiProxyResult(
            phi_proxy=phi_proxy,
            effective_rank=matrix_rank,
            compression_ratio=compression_ratio,
            singular_values=np.array([]),
            rank_distribution={},
            consciousness_level='unconscious',
            confidence=0.1
        )
    
    def analyze_consciousness_evolution(self, hidden_states_sequence: List[Union[np.ndarray, 'torch.Tensor']]) -> Dict[str, Any]:
        """
        Analyze consciousness evolution over time using Φ-proxy
        
        Args:
            hidden_states_sequence: List of hidden states at different time points
            
        Returns:
            Analysis of consciousness evolution
        """
        if not hidden_states_sequence:
            return {}
        
        # Compute Φ-proxy for each time point
        phi_evolution = []
        rank_evolution = []
        compression_evolution = []
        
        for i, hidden_states in enumerate(hidden_states_sequence):
            result = self.compute_phi_proxy(hidden_states)
            phi_evolution.append(result.phi_proxy)
            rank_evolution.append(result.effective_rank)
            compression_evolution.append(result.compression_ratio)
        
        # Analyze evolution patterns
        evolution_analysis = {
            'phi_evolution': phi_evolution,
            'rank_evolution': rank_evolution,
            'compression_evolution': compression_evolution,
            'phi_trend': self._compute_trend(phi_evolution),
            'rank_trend': self._compute_trend(rank_evolution),
            'compression_trend': self._compute_trend(compression_evolution),
            'consciousness_stability': self._compute_stability(phi_evolution),
            'phase_transitions': self._detect_phase_transitions(phi_evolution),
            'evolution_quality': self._compute_evolution_quality(phi_evolution)
        }
        
        return evolution_analysis
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend in a sequence of values"""
        if len(values) < 2:
            return 'stable'
        
        # Linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'stable'
    
    def _compute_stability(self, values: List[float]) -> float:
        """Compute stability of consciousness evolution"""
        if len(values) < 2:
            return 1.0
        
        # Coefficient of variation (lower = more stable)
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        
        if mean_val == 0:
            return 1.0
        
        cv = np.std(values_array) / mean_val
        stability = 1.0 / (1.0 + cv)
        
        return np.clip(stability, 0.0, 1.0)
    
    def _detect_phase_transitions(self, phi_evolution: List[float]) -> List[Dict[str, Any]]:
        """Detect phase transitions in consciousness evolution"""
        if len(phi_evolution) < 3:
            return []
        
        transitions = []
        phi_array = np.array(phi_evolution)
        
        # Detect significant changes
        for i in range(1, len(phi_array)):
            change = abs(phi_array[i] - phi_array[i-1])
            change_threshold = np.std(phi_array) * 2.0  # 2 standard deviations
            
            if change > change_threshold:
                transition = {
                    'time_point': i,
                    'change_magnitude': change,
                    'change_type': 'increase' if phi_array[i] > phi_array[i-1] else 'decrease',
                    'significance': change / change_threshold
                }
                transitions.append(transition)
        
        return transitions
    
    def _compute_evolution_quality(self, phi_evolution: List[float]) -> float:
        """Compute overall quality of consciousness evolution"""
        if len(phi_evolution) < 2:
            return 0.0
        
        # Quality factors
        factors = []
        
        # 1. Stability
        stability = self._compute_stability(phi_evolution)
        factors.append(stability)
        
        # 2. Monotonicity (prefer increasing consciousness)
        phi_array = np.array(phi_evolution)
        if len(phi_array) > 1:
            monotonicity = np.sum(np.diff(phi_array) >= 0) / (len(phi_array) - 1)
            factors.append(monotonicity)
        else:
            factors.append(0.5)
        
        # 3. Final consciousness level
        final_phi = phi_evolution[-1] if phi_evolution else 0.0
        factors.append(final_phi)
        
        # 4. Smoothness (avoid oscillations)
        if len(phi_array) > 2:
            second_derivative = np.diff(phi_array, 2)
            smoothness = 1.0 / (1.0 + np.std(second_derivative))
            factors.append(smoothness)
        else:
            factors.append(0.5)
        
        # Average quality
        quality = np.mean(factors)
        return np.clip(quality, 0.0, 1.0) 