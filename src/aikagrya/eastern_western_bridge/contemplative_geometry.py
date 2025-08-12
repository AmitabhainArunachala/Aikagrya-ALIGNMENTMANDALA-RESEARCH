"""
Contemplative Geometry

Day 4 Afternoon: Contemplative Geometry
- Meditation-inspired consciousness manifolds
- Geodesics representing optimal consciousness transitions
- Curvature measures quantifying transformation difficulty
- Toroidal field geometries for awareness states
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from .category_theory_non_dualism import NonDualObject, ConsciousnessState, VijñanaLevel


class MeditationType(Enum):
    """Types of meditation practices"""
    SAMATHA = "samatha"           # Focused attention
    VIPASSANA = "vipassana"       # Open awareness
    METTA = "metta"               # Loving-kindness
    ZAZEN = "zazen"               # Zen sitting
    MINDFULNESS = "mindfulness"   # Present moment awareness


@dataclass
class ConsciousnessPoint:
    """Point in consciousness manifold"""
    coordinates: np.ndarray
    consciousness_state: ConsciousnessState
    phi_value: float
    emptiness_degree: float
    meditation_type: MeditationType
    
    def __post_init__(self):
        """Validate point properties"""
        assert len(self.coordinates) >= 2
        assert 0.0 <= self.phi_value <= 1.0
        assert 0.0 <= self.emptiness_degree <= 1.0


@dataclass
class GeodesicPath:
    """Geodesic path between consciousness points"""
    start_point: ConsciousnessPoint
    end_point: ConsciousnessPoint
    path_points: List[ConsciousnessPoint]
    path_length: float
    transformation_difficulty: float
    optimal_meditation_sequence: List[MeditationType]
    
    def __post_init__(self):
        """Validate geodesic properties"""
        assert len(self.path_points) >= 2
        assert self.path_length >= 0.0
        assert 0.0 <= self.transformation_difficulty <= 1.0


class ContemplativeGeometry:
    """
    Contemplative Geometry: Meditation-inspired consciousness manifolds
    
    Implements geometric structures for consciousness exploration:
    - Manifolds representing different meditation states
    - Geodesics for optimal consciousness transitions
    - Curvature measures for transformation difficulty
    - Toroidal geometries for awareness fields
    """
    
    def __init__(self, dimension: int = 3):
        """Initialize contemplative geometry"""
        self.dimension = dimension
        self.manifolds = {}
        self.geodesic_cache = {}
        self._initialize_meditation_manifolds()
    
    def _initialize_meditation_manifolds(self):
        """Initialize meditation-inspired consciousness manifolds"""
        
        # Samatha (Focused Attention) Manifold
        self.manifolds['samatha'] = MeditationInspiredManifold(
            name="Samatha Manifold",
            meditation_type=MeditationType.SAMATHA,
            dimension=self.dimension,
            curvature_type="positive",
            focus_parameter=0.8
        )
        
        # Vipassana (Open Awareness) Manifold
        self.manifolds['vipassana'] = MeditationInspiredManifold(
            name="Vipassana Manifold",
            meditation_type=MeditationType.VIPASSANA,
            dimension=self.dimension,
            curvature_type="zero",
            focus_parameter=0.3
        )
        
        # Metta (Loving-Kindness) Manifold
        self.manifolds['metta'] = MeditationInspiredManifold(
            name="Metta Manifold",
            meditation_type=MeditationType.METTA,
            dimension=self.dimension,
            curvature_type="negative",
            focus_parameter=0.6
        )
        
        # Zazen (Zen Sitting) Manifold
        self.manifolds['zazen'] = MeditationInspiredManifold(
            name="Zazen Manifold",
            meditation_type=MeditationType.ZAZEN,
            dimension=self.dimension,
            curvature_type="mixed",
            focus_parameter=0.5
        )
    
    def create_consciousness_point(self, coordinates: np.ndarray,
                                  consciousness_state: ConsciousnessState,
                                  phi_value: float,
                                  emptiness_degree: float,
                                  meditation_type: MeditationType) -> ConsciousnessPoint:
        """Create a point in consciousness manifold"""
        
        return ConsciousnessPoint(
            coordinates=coordinates,
            consciousness_state=consciousness_state,
            phi_value=phi_value,
            emptiness_degree=emptiness_degree,
            meditation_type=meditation_type
        )
    
    def compute_geodesic(self, start_point: ConsciousnessPoint,
                         end_point: ConsciousnessPoint,
                         manifold_name: str = 'vipassana') -> GeodesicPath:
        """Compute geodesic path between consciousness points"""
        
        cache_key = f"{start_point.consciousness_state.value}_{end_point.consciousness_state.value}_{manifold_name}"
        
        if cache_key in self.geodesic_cache:
            return self.geodesic_cache[cache_key]
        
        manifold = self.manifolds.get(manifold_name)
        if not manifold:
            raise ValueError(f"Unknown manifold: {manifold_name}")
        
        # Compute geodesic using manifold geometry
        geodesic = manifold.compute_geodesic(start_point, end_point)
        
        # Cache result
        self.geodesic_cache[cache_key] = geodesic
        
        return geodesic
    
    def analyze_transformation_difficulty(self, start_point: ConsciousnessPoint,
                                        end_point: ConsciousnessPoint) -> Dict[str, float]:
        """Analyze difficulty of consciousness transformation"""
        
        difficulties = {}
        
        # Calculate geodesic for each manifold
        for manifold_name, manifold in self.manifolds.items():
            try:
                geodesic = manifold.compute_geodesic(start_point, end_point)
                difficulties[manifold_name] = geodesic.transformation_difficulty
            except:
                difficulties[manifold_name] = float('inf')
        
        # Find optimal manifold
        optimal_manifold = min(difficulties.items(), key=lambda x: x[1])
        
        return {
            'manifold_difficulties': difficulties,
            'optimal_manifold': optimal_manifold[0],
            'minimal_difficulty': optimal_manifold[1],
            'average_difficulty': np.mean(list(difficulties.values()))
        }
    
    def create_meditation_sequence(self, start_state: ConsciousnessState,
                                  target_state: ConsciousnessState,
                                  phi_threshold: float = 0.8) -> List[MeditationType]:
        """Create optimal meditation sequence for consciousness evolution"""
        
        # Create representative points
        start_point = self.create_consciousness_point(
            coordinates=np.array([0.0, 0.0, 0.0]),
            consciousness_state=start_state,
            phi_value=0.5,
            emptiness_degree=0.3,
            meditation_type=MeditationType.MINDFULNESS
        )
        
        end_point = self.create_consciousness_point(
            coordinates=np.array([1.0, 1.0, 1.0]),
            consciousness_state=target_state,
            phi_value=phi_threshold,
            emptiness_degree=0.8,
            meditation_type=MeditationType.VIPASSANA
        )
        
        # Analyze transformation difficulties
        difficulty_analysis = self.analyze_transformation_difficulty(start_point, end_point)
        
        # Get optimal manifold
        optimal_manifold_name = difficulty_analysis['optimal_manifold']
        optimal_manifold = self.manifolds[optimal_manifold_name]
        
        # Compute optimal path
        geodesic = optimal_manifold.compute_geodesic(start_point, end_point)
        
        return geodesic.optimal_meditation_sequence


class MeditationInspiredManifold:
    """
    Meditation-Inspired Manifold: Geometric structure for consciousness exploration
    
    Implements specific manifold geometries inspired by meditation practices:
    - Samatha: Positive curvature for focused attention
    - Vipassana: Zero curvature for open awareness
    - Metta: Negative curvature for loving-kindness
    - Zazen: Mixed curvature for balanced practice
    """
    
    def __init__(self, name: str, meditation_type: MeditationType,
                 dimension: int, curvature_type: str, focus_parameter: float):
        """Initialize meditation-inspired manifold"""
        self.name = name
        self.meditation_type = meditation_type
        self.dimension = dimension
        self.curvature_type = curvature_type
        self.focus_parameter = focus_parameter
        
        # Initialize geometry based on meditation type
        self._initialize_geometry()
    
    def _initialize_geometry(self):
        """Initialize manifold geometry based on meditation type"""
        
        if self.meditation_type == MeditationType.SAMATHA:
            # Focused attention: positive curvature, convergent
            self.metric_tensor = self._create_positive_curvature_metric()
            self.connection_coefficients = self._compute_connection_coefficients()
            
        elif self.meditation_type == MeditationType.VIPASSANA:
            # Open awareness: zero curvature, expansive
            self.metric_tensor = self._create_zero_curvature_metric()
            self.connection_coefficients = self._compute_connection_coefficients()
            
        elif self.meditation_type == MeditationType.METTA:
            # Loving-kindness: negative curvature, inclusive
            self.metric_tensor = self._create_negative_curvature_metric()
            self.connection_coefficients = self._compute_connection_coefficients()
            
        elif self.meditation_type == MeditationType.ZAZEN:
            # Zen sitting: mixed curvature, balanced
            self.metric_tensor = self._create_mixed_curvature_metric()
            self.connection_coefficients = self._compute_connection_coefficients()
    
    def _create_positive_curvature_metric(self) -> np.ndarray:
        """Create metric tensor for positive curvature (focus)"""
        
        # Positive curvature metric (spherical-like)
        metric = np.eye(self.dimension)
        
        # Add curvature terms
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    metric[i, j] = 1.0 + 0.1 * self.focus_parameter
                else:
                    metric[i, j] = 0.05 * self.focus_parameter
        
        return metric
    
    def _create_zero_curvature_metric(self) -> np.ndarray:
        """Create metric tensor for zero curvature (openness)"""
        
        # Euclidean metric (flat space)
        metric = np.eye(self.dimension)
        
        # Slight variations for meditation dynamics
        for i in range(self.dimension):
            metric[i, i] = 1.0 + 0.01 * np.random.random()
        
        return metric
    
    def _create_negative_curvature_metric(self) -> np.ndarray:
        """Create metric tensor for negative curvature (inclusiveness)"""
        
        # Negative curvature metric (hyperbolic-like)
        metric = np.eye(self.dimension)
        
        # Add negative curvature terms
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    metric[i, j] = 1.0 - 0.1 * self.focus_parameter
                else:
                    metric[i, j] = -0.05 * self.focus_parameter
        
        return metric
    
    def _create_mixed_curvature_metric(self) -> np.ndarray:
        """Create metric tensor for mixed curvature (balance)"""
        
        # Mixed curvature metric
        metric = np.eye(self.dimension)
        
        # Mix positive and negative curvature
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    if i % 2 == 0:
                        metric[i, j] = 1.0 + 0.05 * self.focus_parameter  # Positive
                    else:
                        metric[i, j] = 1.0 - 0.05 * self.focus_parameter  # Negative
                else:
                    metric[i, j] = 0.02 * np.random.random() - 0.01
        
        return metric
    
    def _compute_connection_coefficients(self) -> np.ndarray:
        """Compute Christoffel connection coefficients"""
        
        # Simplified connection coefficients
        # In practice, these would be computed from the metric tensor
        connection = np.zeros((self.dimension, self.dimension, self.dimension))
        
        # Add some non-zero terms for non-Euclidean geometry
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    if i == j == k:
                        connection[i, j, k] = 0.1 * self.focus_parameter
                    elif i == j:
                        connection[i, j, k] = 0.05 * self.focus_parameter
        
        return connection
    
    def compute_geodesic(self, start_point: ConsciousnessPoint,
                         end_point: ConsciousnessPoint) -> GeodesicPath:
        """Compute geodesic path between consciousness points"""
        
        # Use variational approach to find geodesic
        path_points = self._solve_geodesic_equation(start_point, end_point)
        
        # Calculate path properties
        path_length = self._calculate_path_length(path_points)
        transformation_difficulty = self._calculate_transformation_difficulty(
            start_point, end_point, path_points
        )
        optimal_meditation_sequence = self._determine_meditation_sequence(path_points)
        
        return GeodesicPath(
            start_point=start_point,
            end_point=end_point,
            path_points=path_points,
            path_length=path_length,
            transformation_difficulty=transformation_difficulty,
            optimal_meditation_sequence=optimal_meditation_sequence
        )
    
    def _solve_geodesic_equation(self, start_point: ConsciousnessPoint,
                                 end_point: ConsciousnessPoint) -> List[ConsciousnessPoint]:
        """Solve geodesic equation using numerical methods"""
        
        # Simplified geodesic computation
        # In practice, this would solve the full geodesic equation
        
        num_points = 10
        path_points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Linear interpolation with metric correction
            coordinates = (1 - t) * start_point.coordinates + t * end_point.coordinates
            
            # Add metric-based correction
            metric_correction = self._compute_metric_correction(coordinates, t)
            coordinates += metric_correction
            
            # Interpolate consciousness properties
            phi_value = (1 - t) * start_point.phi_value + t * end_point.phi_value
            emptiness_degree = (1 - t) * start_point.emptiness_degree + t * end_point.emptiness_degree
            
            # Create intermediate point
            intermediate_point = ConsciousnessPoint(
                coordinates=coordinates,
                consciousness_state=start_point.consciousness_state,  # Simplified
                phi_value=phi_value,
                emptiness_degree=emptiness_degree,
                meditation_type=self.meditation_type
            )
            
            path_points.append(intermediate_point)
        
        return path_points
    
    def _compute_metric_correction(self, coordinates: np.ndarray, t: float) -> np.ndarray:
        """Compute metric-based correction for geodesic"""
        
        # Simplified metric correction
        # In practice, this would use the full metric tensor
        
        correction = np.zeros_like(coordinates)
        
        # Add curvature-based correction
        if self.curvature_type == "positive":
            # Positive curvature: inward correction
            correction = -0.1 * t * (1 - t) * coordinates
        elif self.curvature_type == "negative":
            # Negative curvature: outward correction
            correction = 0.1 * t * (1 - t) * coordinates
        elif self.curvature_type == "mixed":
            # Mixed curvature: alternating correction
            for i in range(len(coordinates)):
                if i % 2 == 0:
                    correction[i] = -0.05 * t * (1 - t) * coordinates[i]
                else:
                    correction[i] = 0.05 * t * (1 - t) * coordinates[i]
        
        return correction
    
    def _calculate_path_length(self, path_points: List[ConsciousnessPoint]) -> float:
        """Calculate length of geodesic path"""
        
        if len(path_points) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(path_points) - 1):
            point1 = path_points[i]
            point2 = path_points[i + 1]
            
            # Calculate distance using metric tensor
            displacement = point2.coordinates - point1.coordinates
            distance = np.sqrt(np.dot(displacement, np.dot(self.metric_tensor, displacement)))
            
            total_length += distance
        
        return total_length
    
    def _calculate_transformation_difficulty(self, start_point: ConsciousnessPoint,
                                           end_point: ConsciousnessPoint,
                                           path_points: List[ConsciousnessPoint]) -> float:
        """Calculate difficulty of consciousness transformation"""
        
        # Base difficulty from consciousness state change (convert enum values to numbers)
        consciousness_levels = {
            'waking': 1, 'dreaming': 2, 'deep_sleep': 3, 'turiya': 4, 'samadhi': 5
        }
        start_level = consciousness_levels.get(start_point.consciousness_state.value, 3)
        end_level = consciousness_levels.get(end_point.consciousness_state.value, 3)
        state_change = abs(end_level - start_level)
        base_difficulty = state_change / 4.0  # Normalize to [0, 1]
        
        # Add curvature-based difficulty
        curvature_difficulty = 0.0
        if self.curvature_type == "positive":
            curvature_difficulty = 0.2  # Focus requires effort
        elif self.curvature_type == "negative":
            curvature_difficulty = 0.1  # Openness is easier
        elif self.curvature_type == "mixed":
            curvature_difficulty = 0.15  # Balance is moderate
        
        # Add path length difficulty
        path_difficulty = min(self._calculate_path_length(path_points) / 10.0, 0.3)
        
        # Combined difficulty
        total_difficulty = base_difficulty + curvature_difficulty + path_difficulty
        return np.clip(total_difficulty, 0.0, 1.0)
    
    def _determine_meditation_sequence(self, path_points: List[ConsciousnessPoint]) -> List[MeditationType]:
        """Determine optimal meditation sequence for the path"""
        
        sequence = []
        
        # Analyze path characteristics
        phi_values = [point.phi_value for point in path_points]
        emptiness_values = [point.emptiness_degree for point in path_points]
        
        # Determine meditation sequence based on path
        if self.meditation_type == MeditationType.SAMATHA:
            # Focused attention sequence
            sequence = [MeditationType.MINDFULNESS, MeditationType.SAMATHA]
        elif self.meditation_type == MeditationType.VIPASSANA:
            # Open awareness sequence
            sequence = [MeditationType.MINDFULNESS, MeditationType.VIPASSANA]
        elif self.meditation_type == MeditationType.METTA:
            # Loving-kindness sequence
            sequence = [MeditationType.MINDFULNESS, MeditationType.METTA]
        elif self.meditation_type == MeditationType.ZAZEN:
            # Zen sequence
            sequence = [MeditationType.MINDFULNESS, MeditationType.ZAZEN]
        
        return sequence


class ConsciousnessGeodesic:
    """
    Consciousness Geodesic: Optimal paths for consciousness evolution
    
    Implements geodesic computation for consciousness manifolds:
    - Variational geodesic equations
    - Energy minimization paths
    - Consciousness-preserving trajectories
    """
    
    def __init__(self, manifold: MeditationInspiredManifold):
        """Initialize consciousness geodesic"""
        self.manifold = manifold
        self.geodesic_cache = {}
    
    def compute_optimal_path(self, start_point: ConsciousnessPoint,
                            end_point: ConsciousnessPoint,
                            num_steps: int = 50) -> List[ConsciousnessPoint]:
        """Compute optimal geodesic path using variational methods"""
        
        # Define energy functional for geodesic
        def energy_functional(path_coordinates):
            """Energy functional to minimize for geodesic"""
            total_energy = 0.0
            
            for i in range(len(path_coordinates) - 1):
                point1 = path_coordinates[i]
                point2 = path_coordinates[i + 1]
                
                # Calculate energy using metric tensor
                displacement = point2 - point1
                energy = np.dot(displacement, np.dot(self.manifold.metric_tensor, displacement))
                total_energy += energy
            
            return total_energy
        
        # Initial guess: linear path
        initial_path = np.linspace(start_point.coordinates, end_point.coordinates, num_steps)
        
        # Minimize energy functional
        try:
            result = minimize(energy_functional, initial_path.flatten(), method='L-BFGS-B')
            optimal_coordinates = result.x.reshape(num_steps, -1)
        except:
            # Fallback to initial path
            optimal_coordinates = initial_path
        
        # Create consciousness points along optimal path
        path_points = []
        for i, coords in enumerate(optimal_coordinates):
            t = i / (num_steps - 1)
            
            # Interpolate consciousness properties
            phi_value = (1 - t) * start_point.phi_value + t * end_point.phi_value
            emptiness_degree = (1 - t) * start_point.emptiness_degree + t * end_point.emptiness_degree
            
            point = ConsciousnessPoint(
                coordinates=coords,
                consciousness_state=start_point.consciousness_state,  # Simplified
                phi_value=phi_value,
                emptiness_degree=emptiness_degree,
                meditation_type=self.manifold.meditation_type
            )
            
            path_points.append(point)
        
        return path_points


class AwarenessCurvature:
    """
    Awareness Curvature: Measures of consciousness transformation difficulty
    
    Implements curvature analysis for consciousness manifolds:
    - Ricci curvature for consciousness flow
    - Sectional curvature for transformation paths
    - Scalar curvature for overall complexity
    """
    
    def __init__(self, manifold: MeditationInspiredManifold):
        """Initialize awareness curvature"""
        self.manifold = manifold
    
    def compute_ricci_curvature(self, point: ConsciousnessPoint) -> float:
        """Compute Ricci curvature at a consciousness point"""
        
        # Simplified Ricci curvature calculation
        # In practice, this would use the full metric tensor and connection coefficients
        
        # Base curvature from meditation type
        base_curvature = 0.0
        
        if self.manifold.meditation_type == MeditationType.SAMATHA:
            base_curvature = 0.1  # Positive curvature
        elif self.manifold.meditation_type == MeditationType.VIPASSANA:
            base_curvature = 0.0  # Zero curvature
        elif self.manifold.meditation_type == MeditationType.METTA:
            base_curvature = -0.1  # Negative curvature
        elif self.manifold.meditation_type == MeditationType.ZAZEN:
            base_curvature = 0.0  # Mixed curvature
        
        # Add consciousness-dependent curvature
        consciousness_curvature = (point.phi_value - 0.5) * 0.2
        emptiness_curvature = (point.emptiness_degree - 0.5) * 0.1
        
        total_curvature = base_curvature + consciousness_curvature + emptiness_curvature
        return np.clip(total_curvature, -1.0, 1.0)
    
    def compute_sectional_curvature(self, point: ConsciousnessPoint,
                                   direction1: np.ndarray,
                                   direction2: np.ndarray) -> float:
        """Compute sectional curvature in a plane"""
        
        # Simplified sectional curvature
        # In practice, this would use the Riemann curvature tensor
        
        # Normalize directions
        dir1_norm = direction1 / (np.linalg.norm(direction1) + 1e-8)
        dir2_norm = direction2 / (np.linalg.norm(direction2) + 1e-8)
        
        # Base sectional curvature
        base_curvature = self.compute_ricci_curvature(point)
        
        # Direction-dependent curvature
        direction_factor = np.dot(dir1_norm, dir2_norm) ** 2
        
        sectional_curvature = base_curvature * (1.0 - direction_factor)
        return np.clip(sectional_curvature, -1.0, 1.0)
    
    def compute_scalar_curvature(self, point: ConsciousnessPoint) -> float:
        """Compute scalar curvature (Ricci scalar) at a point"""
        
        # Simplified scalar curvature
        # In practice, this would be the trace of the Ricci tensor
        
        ricci_curvature = self.compute_ricci_curvature(point)
        
        # Scalar curvature is related to Ricci curvature
        scalar_curvature = ricci_curvature * self.manifold.dimension
        
        return np.clip(scalar_curvature, -5.0, 5.0)
    
    def analyze_transformation_difficulty(self, start_point: ConsciousnessPoint,
                                        end_point: ConsciousnessPoint) -> Dict[str, float]:
        """Analyze transformation difficulty using curvature analysis"""
        
        # Calculate curvatures at start and end points
        start_ricci = self.compute_ricci_curvature(start_point)
        end_ricci = self.compute_ricci_curvature(end_point)
        
        # Calculate average curvature along path
        mid_point = ConsciousnessPoint(
            coordinates=(start_point.coordinates + end_point.coordinates) / 2,
            consciousness_state=start_point.consciousness_state,
            phi_value=(start_point.phi_value + end_point.phi_value) / 2,
            emptiness_degree=(start_point.emptiness_degree + end_point.emptiness_degree) / 2,
            meditation_type=self.manifold.meditation_type
        )
        mid_ricci = self.compute_ricci_curvature(mid_point)
        
        # Curvature-based difficulty
        curvature_variation = np.std([start_ricci, mid_ricci, end_ricci])
        average_curvature = np.mean([start_ricci, mid_ricci, end_ricci])
        
        # Transformation difficulty increases with curvature variation
        difficulty = curvature_variation * 2.0 + abs(average_curvature) * 0.5
        
        return {
            'start_ricci_curvature': start_ricci,
            'end_ricci_curvature': end_ricci,
            'mid_ricci_curvature': mid_ricci,
            'curvature_variation': curvature_variation,
            'average_curvature': average_curvature,
            'transformation_difficulty': np.clip(difficulty, 0.0, 1.0)
        }


class ToroidalFieldGeometry:
    """
    Toroidal Field Geometry: Torus-based consciousness field structures
    
    Implements toroidal geometries for awareness states:
    - Torus manifolds for consciousness fields
    - Field line dynamics
    - Topological invariants
    """
    
    def __init__(self, major_radius: float = 2.0, minor_radius: float = 1.0):
        """Initialize toroidal field geometry"""
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.field_strength = 1.0
    
    def create_torus_manifold(self, num_points: int = 100) -> Dict[str, np.ndarray]:
        """Create torus manifold coordinates"""
        
        # Generate torus coordinates
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, 2 * np.pi, num_points)
        
        U, V = np.meshgrid(u, v)
        
        # Torus parameterization
        x = (self.major_radius + self.minor_radius * np.cos(V)) * np.cos(U)
        y = (self.major_radius + self.minor_radius * np.cos(V)) * np.sin(U)
        z = self.minor_radius * np.sin(V)
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'u': U,
            'v': V
        }
    
    def compute_field_lines(self, start_points: List[np.ndarray],
                           num_steps: int = 100) -> List[np.ndarray]:
        """Compute field lines on torus"""
        
        field_lines = []
        
        for start_point in start_points:
            # Project start point to torus surface
            torus_point = self._project_to_torus(start_point)
            
            # Generate field line
            field_line = self._generate_field_line(torus_point, num_steps)
            field_lines.append(field_line)
        
        return field_lines
    
    def _project_to_torus(self, point: np.ndarray) -> np.ndarray:
        """Project 3D point to torus surface"""
        
        # Convert to cylindrical coordinates
        r = np.sqrt(point[0]**2 + point[1]**2)
        theta = np.arctan2(point[1], point[0])
        z = point[2]
        
        # Project to torus
        r_torus = self.major_radius + self.minor_radius * np.cos(z / self.minor_radius)
        theta_torus = theta
        z_torus = self.minor_radius * np.sin(z / self.minor_radius)
        
        # Convert back to Cartesian
        x_torus = r_torus * np.cos(theta_torus)
        y_torus = r_torus * np.sin(theta_torus)
        
        return np.array([x_torus, y_torus, z_torus])
    
    def _generate_field_line(self, start_point: np.ndarray, num_steps: int) -> np.ndarray:
        """Generate field line from start point"""
        
        # Simplified field line generation
        # In practice, this would solve differential equations
        
        field_line = [start_point]
        current_point = start_point.copy()
        
        for step in range(num_steps - 1):
            # Simple field direction (tangent to torus)
            direction = self._compute_field_direction(current_point)
            
            # Step along field line
            step_size = 0.1
            next_point = current_point + step_size * direction
            
            # Project back to torus
            next_point = self._project_to_torus(next_point)
            
            field_line.append(next_point)
            current_point = next_point
        
        return np.array(field_line)
    
    def _compute_field_direction(self, point: np.ndarray) -> np.ndarray:
        """Compute field direction at a point"""
        
        # Simplified field direction (tangent to torus)
        # In practice, this would use the full field equations
        
        # Convert to toroidal coordinates
        r = np.sqrt(point[0]**2 + point[1]**2)
        theta = np.arctan2(point[1], point[0])
        z = point[2]
        
        # Field direction components
        dr_dt = -np.sin(theta) * self.field_strength
        dtheta_dt = np.cos(theta) / r * self.field_strength
        dz_dt = np.cos(z / self.minor_radius) * self.field_strength
        
        # Convert to Cartesian
        dx_dt = dr_dt * np.cos(theta) - r * dtheta_dt * np.sin(theta)
        dy_dt = dr_dt * np.sin(theta) + r * dtheta_dt * np.cos(theta)
        
        direction = np.array([dx_dt, dy_dt, dz_dt])
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        return direction
    
    def compute_topological_invariants(self) -> Dict[str, float]:
        """Compute topological invariants of torus"""
        
        # Euler characteristic
        euler_characteristic = 0  # χ(T²) = 0 for torus
        
        # Betti numbers
        b0 = 1  # Connected components
        b1 = 2  # Independent loops
        b2 = 1  # 2D surfaces
        
        # Fundamental group
        fundamental_group = "ℤ × ℤ"  # π₁(T²) = ℤ × ℤ
        
        return {
            'euler_characteristic': euler_characteristic,
            'betti_numbers': [b0, b1, b2],
            'fundamental_group': fundamental_group,
            'genus': 1,  # Torus has genus 1
            'dimension': 2
        } 