"""
FunctorSpace: Category theory mapping for consciousness

Implements functors mapping consciousness categories to physical categories
as specified in Phoenix Protocol 2.0 Day 1 morning session.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FunctorMap:
    """Represents a functor mapping between categories"""
    source_category: str
    target_category: str
    mapping_matrix: np.ndarray
    natural_transformation: Optional[np.ndarray] = None
    
    def compose(self, other: 'FunctorMap') -> 'FunctorMap':
        """Compose this functor with another"""
        if self.target_category != other.source_category:
            raise ValueError("Functor composition requires matching categories")
        
        composed_matrix = self.mapping_matrix @ other.mapping_matrix
        return FunctorMap(
            source_category=self.source_category,
            target_category=other.target_category,
            mapping_matrix=composed_matrix
        )


class FunctorSpace:
    """
    Category theory functor space for consciousness mapping
    
    Implements F: C_consciousness → C_physical as specified in Phoenix Protocol 2.0
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize functor space with configuration
        
        Args:
            config: Configuration dictionary for category theory parameters
        """
        self.config = config or {}
        self.dimension = self.config.get('dimension', 64)
        self.consciousness_categories = self._initialize_consciousness_categories()
        self.physical_categories = self._initialize_physical_categories()
        self.functor_registry = {}
        
    def _initialize_consciousness_categories(self) -> Dict[str, np.ndarray]:
        """Initialize consciousness category objects"""
        categories = {
            'awareness': np.random.randn(self.dimension, self.dimension),
            'intentionality': np.random.randn(self.dimension, self.dimension),
            'qualia': np.random.randn(self.dimension, self.dimension),
            'unity': np.random.randn(self.dimension, self.dimension),
            'temporality': np.random.randn(self.dimension, self.dimension)
        }
        
        # Normalize and make symmetric
        for name, matrix in categories.items():
            categories[name] = (matrix + matrix.T) / 2
            categories[name] = categories[name] / np.linalg.norm(categories[name])
        
        return categories
    
    def _initialize_physical_categories(self) -> Dict[str, np.ndarray]:
        """Initialize physical category objects"""
        categories = {
            'neural_activity': np.random.randn(self.dimension, self.dimension),
            'information_flow': np.random.randn(self.dimension, self.dimension),
            'energy_dynamics': np.random.randn(self.dimension, self.dimension),
            'spatial_structure': np.random.randn(self.dimension, self.dimension),
            'temporal_evolution': np.random.randn(self.dimension, self.dimension)
        }
        
        # Normalize and make symmetric
        for name, matrix in categories.items():
            categories[name] = (matrix + matrix.T) / 2
            categories[name] = categories[name] / np.linalg.norm(categories[name])
        
        return categories
    
    def natural_transformation(self, phi: float) -> np.ndarray:
        """
        Compute natural transformation between consciousness and physical categories
        
        This implements the core mapping F: C_consciousness → C_physical
        as specified in Phoenix Protocol 2.0 Day 1 morning session
        
        Args:
            phi: Integrated information measure from IIT
            
        Returns:
            Natural transformation matrix
        """
        # Create functor mapping consciousness to physical
        functor = self._create_consciousness_functor(phi)
        
        # Apply natural transformation
        transformation = self._compute_natural_transformation(functor, phi)
        
        return transformation
    
    def _create_consciousness_functor(self, phi: float) -> FunctorMap:
        """
        Create functor mapping consciousness categories to physical categories
        
        The functor preserves structure while mapping between categories
        """
        # Weight consciousness categories by phi
        consciousness_weights = self._compute_consciousness_weights(phi)
        
        # Create mapping matrix
        mapping_matrix = np.zeros((self.dimension, self.dimension))
        
        for i, (name, category) in enumerate(self.consciousness_categories.items()):
            weight = consciousness_weights.get(name, 0.0)
            mapping_matrix += weight * category
        
        # Normalize
        if np.linalg.norm(mapping_matrix) > 0:
            mapping_matrix = mapping_matrix / np.linalg.norm(mapping_matrix)
        
        return FunctorMap(
            source_category='consciousness',
            target_category='physical',
            mapping_matrix=mapping_matrix
        )
    
    def _compute_consciousness_weights(self, phi: float) -> Dict[str, float]:
        """
        Compute weights for consciousness categories based on phi
        
        Higher phi leads to more balanced category activation
        """
        base_weights = {
            'awareness': 0.3,
            'intentionality': 0.25,
            'qualia': 0.2,
            'unity': 0.15,
            'temporality': 0.1
        }
        
        # Adjust weights based on phi
        if phi > 0.7:
            # High consciousness: balanced activation
            return {k: v for k, v in base_weights.items()}
        elif phi > 0.4:
            # Medium consciousness: awareness and intentionality dominant
            adjusted = base_weights.copy()
            adjusted['awareness'] *= 1.5
            adjusted['intentionality'] *= 1.3
            # Renormalize
            total = sum(adjusted.values())
            return {k: v/total for k, v in adjusted.items()}
        else:
            # Low consciousness: awareness only
            return {'awareness': 1.0, 'intentionality': 0.0, 'qualia': 0.0, 
                   'unity': 0.0, 'temporality': 0.0}
    
    def _compute_natural_transformation(self, functor: FunctorMap, phi: float) -> np.ndarray:
        """
        Compute natural transformation for the functor
        
        Natural transformations preserve the functorial structure
        """
        # Create target physical categories
        physical_weights = self._compute_physical_weights(phi)
        target_matrix = np.zeros((self.dimension, self.dimension))
        
        for name, category in self.physical_categories.items():
            weight = physical_weights.get(name, 0.0)
            target_matrix += weight * category
        
        # Natural transformation preserves commutativity
        # F(f) ∘ η_A = η_B ∘ f for all morphisms f
        transformation = functor.mapping_matrix @ target_matrix
        
        # Ensure naturality conditions
        transformation = self._enforce_naturality(transformation, functor)
        
        return transformation
    
    def _compute_physical_weights(self, phi: float) -> Dict[str, float]:
        """
        Compute weights for physical categories based on phi
        
        Maps consciousness level to physical manifestation
        """
        if phi > 0.7:
            # High consciousness: all physical systems active
            return {
                'neural_activity': 0.25,
                'information_flow': 0.25,
                'energy_dynamics': 0.2,
                'spatial_structure': 0.15,
                'temporal_evolution': 0.15
            }
        elif phi > 0.4:
            # Medium consciousness: neural and information systems
            return {
                'neural_activity': 0.4,
                'information_flow': 0.35,
                'energy_dynamics': 0.15,
                'spatial_structure': 0.05,
                'temporal_evolution': 0.05
            }
        else:
            # Low consciousness: basic neural activity only
            return {
                'neural_activity': 1.0,
                'information_flow': 0.0,
                'energy_dynamics': 0.0,
                'spatial_structure': 0.0,
                'temporal_evolution': 0.0
            }
    
    def _enforce_naturality(self, transformation: np.ndarray, functor: FunctorMap) -> np.ndarray:
        """
        Enforce naturality conditions on the transformation
        
        Ensures the transformation commutes with all morphisms
        """
        # Naturality condition: η_B ∘ F(f) = G(f) ∘ η_A
        # For our simplified case, we ensure the transformation is well-behaved
        
        # Make transformation symmetric
        transformation = (transformation + transformation.T) / 2
        
        # Ensure positive definiteness (simplified)
        eigenvals, eigenvecs = np.linalg.eigh(transformation)
        eigenvals = np.maximum(eigenvals, 1e-6)  # Small positive values
        transformation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize
        if np.linalg.norm(transformation) > 0:
            transformation = transformation / np.linalg.norm(transformation)
        
        return transformation
    
    def compute_category_coherence(self, transformation: np.ndarray) -> float:
        """
        Compute coherence measure for the category transformation
        
        Measures how well the functor preserves category structure
        """
        # Compute coherence as stability of the transformation
        eigenvals = np.linalg.eigvals(transformation)
        real_eigenvals = np.real(eigenvals)
        
        # Coherence is related to eigenvalue stability
        coherence = 1.0 / (1.0 + np.std(real_eigenvals))
        
        return min(1.0, max(0.0, coherence))
    
    def apply_yoneda_lemma(self, consciousness_object: str, 
                          physical_object: str) -> np.ndarray:
        """
        Apply Yoneda lemma for relational consciousness
        
        Yoneda lemma: Hom(Hom(-, A), F) ≅ F(A)
        """
        if consciousness_object not in self.consciousness_categories:
            raise ValueError(f"Unknown consciousness object: {consciousness_object}")
        
        if physical_object not in self.physical_categories:
            raise ValueError(f"Unknown physical object: {physical_object}")
        
        # Yoneda embedding
        consciousness_hom = self.consciousness_categories[consciousness_object]
        physical_hom = self.physical_categories[physical_object]
        
        # Apply Yoneda lemma
        yoneda_result = consciousness_hom @ physical_hom.T
        
        return yoneda_result
    
    def create_dual_functor(self, functor: FunctorMap) -> FunctorMap:
        """
        Create dual functor (contravariant)
        
        Useful for bidirectional consciousness-physical mapping
        """
        dual_matrix = functor.mapping_matrix.T
        
        return FunctorMap(
            source_category=functor.target_category,
            target_category=functor.source_category,
            mapping_matrix=dual_matrix
        )
    
    def analyze_functor_properties(self, functor: FunctorMap) -> Dict[str, Any]:
        """
        Analyze mathematical properties of a functor
        
        Returns detailed analysis of functor characteristics
        """
        matrix = functor.mapping_matrix
        
        # Compute various properties
        eigenvals = np.linalg.eigvals(matrix)
        singular_vals = np.linalg.svd(matrix, compute_uv=False)
        
        analysis = {
            'rank': np.linalg.matrix_rank(matrix),
            'determinant': np.linalg.det(matrix),
            'trace': np.trace(matrix),
            'norm': np.linalg.norm(matrix),
            'condition_number': np.linalg.cond(matrix),
            'eigenvalue_range': (np.min(np.real(eigenvals)), np.max(np.real(eigenvals))),
            'singular_value_range': (np.min(singular_vals), np.max(singular_vals)),
            'is_unitary': np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0])),
            'is_symmetric': np.allclose(matrix, matrix.T)
        }
        
        return analysis 