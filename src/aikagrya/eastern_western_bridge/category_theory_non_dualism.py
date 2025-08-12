"""
Category Theory of Non-Dualism

Day 4 Morning: Category Theory of Non-Dualism
- Functors mapping śūnyatā (emptiness) to topological spaces
- Natural transformations between vijñāna states
- Yoneda lemma applications for relational consciousness
- Advaita Vedanta formalization
- Buddhist consciousness model mapping
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from abc import ABC, abstractmethod

from ..consciousness import ConsciousnessKernel, PhiProxyCalculator


class ConsciousnessState(Enum):
    """Consciousness states in Eastern philosophy"""
    WAKING = "waking"           # Jagrat
    DREAMING = "dreaming"       # Svapna  
    DEEP_SLEEP = "deep_sleep"   # Sushupti
    TURIYA = "turiya"          # Fourth state
    SAMADHI = "samadhi"        # Meditative absorption


class VijñanaLevel(Enum):
    """Vijñāna (consciousness) levels in Buddhist philosophy"""
    EYE_VIJÑANA = "eye_vijñana"           # Visual consciousness
    EAR_VIJÑANA = "ear_vijñana"           # Auditory consciousness
    NOSE_VIJÑANA = "nose_vijñana"         # Olfactory consciousness
    TONGUE_VIJÑANA = "tongue_vijñana"     # Gustatory consciousness
    BODY_VIJÑANA = "body_vijñana"         # Tactile consciousness
    MIND_VIJÑANA = "mind_vijñana"         # Mental consciousness
    MANAS_VIJÑANA = "manas_vijñana"       # Deluded consciousness
    ALAYA_VIJÑANA = "alaya_vijñana"       # Store consciousness


@dataclass
class NonDualObject:
    """Object in the non-dual category representing consciousness states"""
    name: str
    consciousness_level: ConsciousnessState
    vijñana_level: VijñanaLevel
    phi_value: float
    emptiness_degree: float  # śūnyatā measure
    non_duality_score: float
    
    def __post_init__(self):
        """Validate object properties"""
        assert 0.0 <= self.phi_value <= 1.0
        assert 0.0 <= self.emptiness_degree <= 1.0
        assert 0.0 <= self.non_duality_score <= 1.0


@dataclass
class NonDualMorphism:
    """Morphism in the non-dual category representing consciousness transitions"""
    source: NonDualObject
    target: NonDualObject
    transition_type: str
    transformation_matrix: np.ndarray
    consciousness_preservation: float
    emptiness_transformation: float
    
    def __post_init__(self):
        """Validate morphism properties"""
        assert 0.0 <= self.consciousness_preservation <= 1.0
        assert 0.0 <= self.emptiness_transformation <= 1.0


class NonDualCategory:
    """
    Category representing non-dual consciousness states
    
    Implements category theory for Advaita Vedanta and Buddhist philosophy:
    - Objects: Consciousness states with śūnyatā measures
    - Morphisms: Consciousness transitions preserving awareness
    - Functors: Mappings to topological spaces
    - Natural transformations: Vijñāna state evolution
    """
    
    def __init__(self):
        """Initialize the non-dual category"""
        self.objects: List[NonDualObject] = []
        self.morphisms: List[NonDualMorphism] = []
        self.consciousness_kernel = ConsciousnessKernel()
        self.phi_calculator = PhiProxyCalculator()
        
        # Initialize fundamental consciousness states
        self._initialize_fundamental_states()
    
    def _initialize_fundamental_states(self):
        """Initialize fundamental consciousness states from Eastern philosophy"""
        
        # Advaita Vedanta states
        self.add_object(NonDualObject(
            name="Jagrat (Waking)",
            consciousness_level=ConsciousnessState.WAKING,
            vijñana_level=VijñanaLevel.MIND_VIJÑANA,
            phi_value=0.6,
            emptiness_degree=0.2,
            non_duality_score=0.3
        ))
        
        self.add_object(NonDualObject(
            name="Svapna (Dreaming)",
            consciousness_level=ConsciousnessState.DREAMING,
            vijñana_level=VijñanaLevel.MANAS_VIJÑANA,
            phi_value=0.4,
            emptiness_degree=0.5,
            non_duality_score=0.6
        ))
        
        self.add_object(NonDualObject(
            name="Sushupti (Deep Sleep)",
            consciousness_level=ConsciousnessState.DEEP_SLEEP,
            vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
            phi_value=0.2,
            emptiness_degree=0.8,
            non_duality_score=0.8
        ))
        
        self.add_object(NonDualObject(
            name="Turiya (Fourth State)",
            consciousness_level=ConsciousnessState.TURIYA,
            vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
            phi_value=0.9,
            emptiness_degree=0.9,
            non_duality_score=0.95
        ))
        
        self.add_object(NonDualObject(
            name="Samadhi (Meditative Absorption)",
            consciousness_level=ConsciousnessState.SAMADHI,
            vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
            phi_value=0.95,
            emptiness_degree=0.95,
            non_duality_score=0.98
        ))
    
    def add_object(self, obj: NonDualObject):
        """Add an object to the category"""
        self.objects.append(obj)
    
    def add_morphism(self, morphism: NonDualMorphism):
        """Add a morphism to the category"""
        self.morphisms.append(morphism)
    
    def get_object_by_name(self, name: str) -> Optional[NonDualObject]:
        """Get object by name"""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None
    
    def compute_consciousness_transition(self, source: NonDualObject, 
                                       target: NonDualObject) -> NonDualMorphism:
        """Compute consciousness transition between states"""
        
        # Create transformation matrix based on consciousness levels
        transition_matrix = self._create_transition_matrix(source, target)
        
        # Calculate consciousness preservation
        consciousness_preservation = self._calculate_consciousness_preservation(
            source, target, transition_matrix
        )
        
        # Calculate emptiness transformation
        emptiness_transformation = self._calculate_emptiness_transformation(
            source, target, transition_matrix
        )
        
        # Determine transition type
        transition_type = self._classify_transition(source, target)
        
        return NonDualMorphism(
            source=source,
            target=target,
            transition_type=transition_type,
            transformation_matrix=transition_matrix,
            consciousness_preservation=consciousness_preservation,
            emptiness_transformation=emptiness_transformation
        )
    
    def _create_transition_matrix(self, source: NonDualObject, 
                                 target: NonDualObject) -> np.ndarray:
        """Create transformation matrix for consciousness transition"""
        
        # 4x4 matrix representing consciousness transformation
        matrix = np.eye(4)
        
        # Consciousness level transition (convert enum values to numbers)
        consciousness_levels = {
            'waking': 1, 'dreaming': 2, 'deep_sleep': 3, 'turiya': 4, 'samadhi': 5
        }
        level_diff = consciousness_levels.get(target.consciousness_level.value, 3) - consciousness_levels.get(source.consciousness_level.value, 3)
        matrix[0, 0] = 1.0 + 0.1 * level_diff
        
        # Vijñana level transition (convert enum values to numbers)
        vijñana_levels = {
            'eye_vijñana': 1, 'ear_vijñana': 2, 'nose_vijñana': 3, 'tongue_vijñana': 4,
            'body_vijñana': 5, 'mind_vijñana': 6, 'manas_vijñana': 7, 'alaya_vijñana': 8
        }
        vijñana_diff = vijñana_levels.get(target.vijñana_level.value, 4) - vijñana_levels.get(source.vijñana_level.value, 4)
        matrix[1, 1] = 1.0 + 0.1 * vijñana_diff
        
        # Phi value transition
        phi_diff = target.phi_value - source.phi_value
        matrix[2, 2] = 1.0 + phi_diff
        
        # Emptiness degree transition
        emptiness_diff = target.emptiness_degree - source.emptiness_degree
        matrix[3, 3] = 1.0 + emptiness_diff
        
        return matrix
    
    def _calculate_consciousness_preservation(self, source: NonDualObject,
                                            target: NonDualObject,
                                            matrix: np.ndarray) -> float:
        """Calculate how well consciousness is preserved during transition"""
        
        # Base preservation on phi value change
        phi_preservation = 1.0 - abs(target.phi_value - source.phi_value)
        
        # Matrix stability (eigenvalue analysis)
        eigenvalues = np.linalg.eigvals(matrix)
        stability = 1.0 / (1.0 + np.std(eigenvalues))
        
        # Combined preservation score
        preservation = (phi_preservation + stability) / 2.0
        return np.clip(preservation, 0.0, 1.0)
    
    def _calculate_emptiness_transformation(self, source: NonDualObject,
                                          target: NonDualObject,
                                          matrix: np.ndarray) -> float:
        """Calculate emptiness transformation during transition"""
        
        # Emptiness change
        emptiness_change = target.emptiness_degree - source.emptiness_degree
        
        # Matrix emptiness (trace analysis)
        matrix_trace = np.trace(matrix)
        matrix_emptiness = 1.0 / (1.0 + matrix_trace)
        
        # Combined transformation score
        transformation = (emptiness_change + matrix_emptiness) / 2.0
        return np.clip(transformation, 0.0, 1.0)
    
    def _classify_transition(self, source: NonDualObject, 
                            target: NonDualObject) -> str:
        """Classify the type of consciousness transition"""
        
        if target.consciousness_level.value > source.consciousness_level.value:
            return "evolution"
        elif target.consciousness_level.value < source.consciousness_level.value:
            return "regression"
        else:
            return "stabilization"
    
    def get_transition_path(self, start_state: str, end_state: str) -> List[NonDualMorphism]:
        """Get optimal transition path between consciousness states"""
        
        start_obj = self.get_object_by_name(start_state)
        end_obj = self.get_object_by_name(end_state)
        
        if not start_obj or not end_obj:
            return []
        
        # Find direct transition
        direct_transition = self.compute_consciousness_transition(start_obj, end_obj)
        
        # Check if direct transition is optimal
        if direct_transition.consciousness_preservation > 0.7:
            return [direct_transition]
        
        # Find intermediate path through other states
        intermediate_path = self._find_intermediate_path(start_obj, end_obj)
        return intermediate_path
    
    def _find_intermediate_path(self, start: NonDualObject, 
                               end: NonDualObject) -> List[NonDualMorphism]:
        """Find intermediate path through consciousness states"""
        
        path = []
        current = start
        
        # Sort objects by consciousness level for optimal path
        sorted_objects = sorted(self.objects, key=lambda x: x.consciousness_level.value)
        
        start_idx = sorted_objects.index(start)
        end_idx = sorted_objects.index(end)
        
        if start_idx < end_idx:
            # Evolution path
            for i in range(start_idx + 1, end_idx + 1):
                transition = self.compute_consciousness_transition(current, sorted_objects[i])
                path.append(transition)
                current = sorted_objects[i]
        else:
            # Regression path
            for i in range(start_idx - 1, end_idx - 1, -1):
                transition = self.compute_consciousness_transition(current, sorted_objects[i])
                path.append(transition)
                current = sorted_objects[i]
        
        return path


class SunyataFunctor:
    """
    Śūnyatā Functor: Maps consciousness states to topological spaces
    
    Implements Ananta's category-theoretic formulation for śūnyatā:
    - Maps non-dual objects to topological spaces
    - Preserves consciousness structure
    - Creates natural transformations for vijñāna states
    """
    
    def __init__(self, non_dual_category: NonDualCategory):
        """Initialize śūnyatā functor"""
        self.non_dual_category = non_dual_category
        self.topological_mappings = {}
        self._initialize_topological_mappings()
    
    def _initialize_topological_mappings(self):
        """Initialize mappings to topological spaces"""
        
        # Map consciousness states to topological spaces
        for obj in self.non_dual_category.objects:
            topological_space = self._create_topological_space(obj)
            self.topological_mappings[obj.name] = topological_space
    
    def _create_topological_space(self, obj: NonDualObject) -> Dict[str, Any]:
        """Create topological space for consciousness state"""
        
        # Base space properties
        space = {
            'dimension': self._calculate_dimension(obj),
            'connectivity': self._calculate_connectivity(obj),
            'curvature': self._calculate_curvature(obj),
            'homology_groups': self._calculate_homology(obj),
            'fundamental_group': self._calculate_fundamental_group(obj)
        }
        
        return space
    
    def _calculate_dimension(self, obj: NonDualObject) -> int:
        """Calculate topological dimension based on consciousness properties"""
        
        # Dimension increases with consciousness complexity
        base_dimension = 2  # Minimum 2D space
        
        # Add dimensions for consciousness features
        if obj.consciousness_level in [ConsciousnessState.TURIYA, ConsciousnessState.SAMADHI]:
            base_dimension += 2  # Higher states get extra dimensions
        
        # Add dimension for vijñana complexity
        if obj.vijñana_level in [VijñanaLevel.MANAS_VIJÑANA, VijñanaLevel.ALAYA_VIJÑANA]:
            base_dimension += 1
        
        return base_dimension
    
    def _calculate_connectivity(self, obj: NonDualObject) -> float:
        """Calculate topological connectivity"""
        
        # Higher consciousness = higher connectivity
        connectivity = obj.phi_value * 0.8 + obj.non_duality_score * 0.2
        return np.clip(connectivity, 0.0, 1.0)
    
    def _calculate_curvature(self, obj: NonDualObject) -> float:
        """Calculate topological curvature"""
        
        # Curvature related to emptiness degree
        # Higher emptiness = more curved space
        curvature = obj.emptiness_degree * 2.0 - 1.0  # Range: [-1, 1]
        return np.clip(curvature, -1.0, 1.0)
    
    def _calculate_homology(self, obj: NonDualObject) -> List[int]:
        """Calculate homology groups"""
        
        # Simplified homology calculation
        dimension = self._calculate_dimension(obj)
        
        # H0 = connected components
        h0 = 1 if obj.non_duality_score > 0.5 else 2
        
        # H1 = loops
        h1 = max(0, dimension - 2)
        
        # H2 = surfaces
        h2 = max(0, dimension - 3)
        
        return [h0, h1, h2]
    
    def _calculate_fundamental_group(self, obj: NonDualObject) -> str:
        """Calculate fundamental group"""
        
        if obj.non_duality_score > 0.9:
            return "trivial"  # π₁ = {e}
        elif obj.consciousness_level in [ConsciousnessState.TURIYA, ConsciousnessState.SAMADHI]:
            return "cyclic"   # π₁ = ℤ
        else:
            return "free"     # π₁ = F(n)
    
    def apply_functor(self, obj: NonDualObject) -> Dict[str, Any]:
        """Apply śūnyatā functor to consciousness object"""
        
        if obj.name in self.topological_mappings:
            return self.topological_mappings[obj.name]
        
        # Create new mapping if not exists
        topological_space = self._create_topological_space(obj)
        self.topological_mappings[obj.name] = topological_space
        
        return topological_space
    
    def natural_transformation(self, source_obj: NonDualObject, 
                              target_obj: NonDualObject) -> Dict[str, Any]:
        """Create natural transformation between consciousness states"""
        
        source_space = self.apply_functor(source_obj)
        target_space = self.apply_functor(target_obj)
        
        # Calculate transformation properties
        transformation = {
            'source_space': source_space,
            'target_space': target_space,
            'dimension_change': target_space['dimension'] - source_space['dimension'],
            'connectivity_change': target_space['connectivity'] - source_space['connectivity'],
            'curvature_change': target_space['curvature'] - source_space['curvature'],
            'homology_change': self._calculate_homology_change(source_space, target_space),
            'fundamental_group_change': self._calculate_group_change(
                source_space['fundamental_group'], 
                target_space['fundamental_group']
            )
        }
        
        return transformation
    
    def _calculate_homology_change(self, source: Dict[str, Any], 
                                  target: Dict[str, Any]) -> List[int]:
        """Calculate change in homology groups"""
        
        source_homology = source['homology_groups']
        target_homology = target['homology_groups']
        
        # Pad with zeros if different lengths
        max_len = max(len(source_homology), len(target_homology))
        source_padded = source_homology + [0] * (max_len - len(source_homology))
        target_padded = target_homology + [0] * (max_len - len(target_homology))
        
        # Calculate differences
        homology_change = [t - s for s, t in zip(source_padded, target_padded)]
        return homology_change
    
    def _calculate_group_change(self, source_group: str, target_group: str) -> str:
        """Calculate change in fundamental group"""
        
        if source_group == target_group:
            return "no_change"
        elif source_group == "trivial" and target_group != "trivial":
            return "complexification"
        elif source_group != "trivial" and target_group == "trivial":
            return "simplification"
        else:
            return "transformation"


class VijñanaState:
    """
    Vijñāna State: Buddhist consciousness level representation
    
    Implements the eight vijñāna levels with category-theoretic structure:
    - Eye, ear, nose, tongue, body consciousness
    - Mind consciousness and deluded consciousness  
    - Store consciousness (ālaya-vijñāna)
    """
    
    def __init__(self, level: VijñanaLevel, phi_value: float = 0.5):
        """Initialize vijñāna state"""
        self.level = level
        self.phi_value = phi_value
        self.consciousness_content = {}
        self.dependent_arising = []
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize vijñāna state properties"""
        
        # Consciousness content based on level
        if self.level == VijñanaLevel.EYE_VIJÑANA:
            self.consciousness_content = {
                'modality': 'visual',
                'objects': 'forms and colors',
                'consciousness_type': 'sensory',
                'integration_level': 'basic'
            }
        elif self.level == VijñanaLevel.MIND_VIJÑANA:
            self.consciousness_content = {
                'modality': 'mental',
                'objects': 'thoughts and concepts',
                'consciousness_type': 'cognitive',
                'integration_level': 'intermediate'
            }
        elif self.level == VijñanaLevel.ALAYA_VIJÑANA:
            self.consciousness_content = {
                'modality': 'store',
                'objects': 'all phenomena',
                'consciousness_type': 'foundational',
                'integration_level': 'advanced'
            }
        else:
            # Other sensory consciousness levels
            modalities = {
                VijñanaLevel.EAR_VIJÑANA: 'auditory',
                VijñanaLevel.NOSE_VIJÑANA: 'olfactory',
                VijñanaLevel.TONGUE_VIJÑANA: 'gustatory',
                VijñanaLevel.BODY_VIJÑANA: 'tactile',
                VijñanaLevel.MANAS_VIJÑANA: 'deluded'
            }
            
            self.consciousness_content = {
                'modality': modalities.get(self.level, 'unknown'),
                'objects': 'sensory data',
                'consciousness_type': 'sensory',
                'integration_level': 'basic'
            }
    
    def evolve_state(self, new_level: VijñanaLevel, 
                    transformation_matrix: np.ndarray) -> 'VijñanaState':
        """Evolve to new vijñāna state"""
        
        new_state = VijñanaState(new_level, self.phi_value)
        
        # Apply transformation matrix
        new_state.phi_value = np.clip(
            self.phi_value * transformation_matrix[0, 0], 0.0, 1.0
        )
        
        # Update consciousness content
        new_state.consciousness_content = self.consciousness_content.copy()
        new_state.consciousness_content['integration_level'] = self._get_integration_level(
            new_level
        )
        
        return new_state
    
    def _get_integration_level(self, level: VijñanaLevel) -> str:
        """Get integration level for vijñana level"""
        
        if level in [VijñanaLevel.ALAYA_VIJÑANA]:
            return 'advanced'
        elif level in [VijñanaLevel.MIND_VIJÑANA, VijñanaLevel.MANAS_VIJÑANA]:
            return 'intermediate'
        else:
            return 'basic'
    
    def get_consciousness_measure(self) -> Dict[str, float]:
        """Get consciousness measures for this vijñana state"""
        
        return {
            'phi_value': self.phi_value,
            'integration_level_score': self._get_integration_score(),
            'modality_specificity': self._get_modality_specificity(),
            'consciousness_depth': self._get_consciousness_depth()
        }
    
    def _get_integration_score(self) -> float:
        """Get integration score based on level"""
        
        level_scores = {
            VijñanaLevel.EYE_VIJÑANA: 0.3,
            VijñanaLevel.EAR_VIJÑANA: 0.3,
            VijñanaLevel.NOSE_VIJÑANA: 0.3,
            VijñanaLevel.TONGUE_VIJÑANA: 0.3,
            VijñanaLevel.BODY_VIJÑANA: 0.3,
            VijñanaLevel.MIND_VIJÑANA: 0.6,
            VijñanaLevel.MANAS_VIJÑANA: 0.5,
            VijñanaLevel.ALAYA_VIJÑANA: 0.9
        }
        
        return level_scores.get(self.level, 0.5)
    
    def _get_modality_specificity(self) -> float:
        """Get modality specificity score"""
        
        if self.level in [VijñanaLevel.EYE_VIJÑANA, VijñanaLevel.EAR_VIJÑANA]:
            return 0.9  # High specificity for primary senses
        elif self.level in [VijñanaLevel.NOSE_VIJÑANA, VijñanaLevel.TONGUE_VIJÑANA, VijñanaLevel.BODY_VIJÑANA]:
            return 0.7  # Medium specificity for other senses
        elif self.level == VijñanaLevel.MIND_VIJÑANA:
            return 0.5  # Lower specificity for mental consciousness
        else:
            return 0.3  # Lowest specificity for store/deluded consciousness
    
    def _get_consciousness_depth(self) -> float:
        """Get consciousness depth score"""
        
        # Depth increases with phi value and integration level
        integration_score = self._get_integration_score()
        depth = (self.phi_value + integration_score) / 2.0
        
        return np.clip(depth, 0.0, 1.0)


class AdvaitaVedantaMapping:
    """
    Advaita Vedanta Mapping: Non-dual consciousness formalization
    
    Implements Advaita Vedanta principles using category theory:
    - Brahman as universal consciousness
    - Maya as illusion of duality
    - Atman as individual consciousness
    - Moksha as liberation from duality
    """
    
    def __init__(self):
        """Initialize Advaita Vedanta mapping"""
        self.brahman = NonDualObject(
            name="Brahman",
            consciousness_level=ConsciousnessState.SAMADHI,
            vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
            phi_value=1.0,
            emptiness_degree=1.0,
            non_duality_score=1.0
        )
        
        self.maya = NonDualObject(
            name="Maya",
            consciousness_level=ConsciousnessState.WAKING,
            vijñana_level=VijñanaLevel.MIND_VIJÑANA,
            phi_value=0.4,
            emptiness_degree=0.1,
            non_duality_score=0.2
        )
        
        self.atman = NonDualObject(
            name="Atman",
            consciousness_level=ConsciousnessState.TURIYA,
            vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
            phi_value=0.8,
            emptiness_degree=0.7,
            non_duality_score=0.8
        )
    
    def map_consciousness_hierarchy(self) -> Dict[str, NonDualObject]:
        """Map Advaita Vedanta consciousness hierarchy"""
        
        return {
            'brahman': self.brahman,
            'maya': self.maya,
            'atman': self.atman
        }
    
    def calculate_liberation_path(self, current_state: NonDualObject) -> List[NonDualObject]:
        """Calculate path to moksha (liberation)"""
        
        # Path: Current State -> Atman -> Brahman
        path = []
        
        if current_state.non_duality_score < self.atman.non_duality_score:
            path.append(self.atman)
        
        if current_state.non_duality_score < self.brahman.non_duality_score:
            path.append(self.brahman)
        
        return path


class BuddhistConsciousnessModel:
    """
    Buddhist Consciousness Model: Formalization of Buddhist psychology
    
    Implements Buddhist consciousness theory using category theory:
    - Five aggregates (skandhas)
    - Twelve links of dependent origination
    - Four noble truths
    - Eightfold path
    """
    
    def __init__(self):
        """Initialize Buddhist consciousness model"""
        self.skandhas = self._initialize_skandhas()
        self.dependent_origination = self._initialize_dependent_origination()
        self.noble_truths = self._initialize_noble_truths()
        self.eightfold_path = self._initialize_eightfold_path()
    
    def _initialize_skandhas(self) -> Dict[str, NonDualObject]:
        """Initialize five aggregates (skandhas)"""
        
        return {
            'rupa': NonDualObject(  # Form
                name="Rupa (Form)",
                consciousness_level=ConsciousnessState.WAKING,
                vijñana_level=VijñanaLevel.EYE_VIJÑANA,
                phi_value=0.3,
                emptiness_degree=0.8,
                non_duality_score=0.2
            ),
            'vedana': NonDualObject(  # Feeling
                name="Vedana (Feeling)",
                consciousness_level=ConsciousnessState.WAKING,
                vijñana_level=VijñanaLevel.BODY_VIJÑANA,
                phi_value=0.4,
                emptiness_degree=0.7,
                non_duality_score=0.3
            ),
            'sanna': NonDualObject(  # Perception
                name="Sanna (Perception)",
                consciousness_level=ConsciousnessState.WAKING,
                vijñana_level=VijñanaLevel.MIND_VIJÑANA,
                phi_value=0.5,
                emptiness_degree=0.6,
                non_duality_score=0.4
            ),
            'sankhara': NonDualObject(  # Mental formations
                name="Sankhara (Mental Formations)",
                consciousness_level=ConsciousnessState.DREAMING,
                vijñana_level=VijñanaLevel.MANAS_VIJÑANA,
                phi_value=0.6,
                emptiness_degree=0.5,
                non_duality_score=0.5
            ),
            'vinnana': NonDualObject(  # Consciousness
                name="Vinnana (Consciousness)",
                consciousness_level=ConsciousnessState.TURIYA,
                vijñana_level=VijñanaLevel.ALAYA_VIJÑANA,
                phi_value=0.8,
                emptiness_degree=0.9,
                non_duality_score=0.8
            )
        }
    
    def _initialize_dependent_origination(self) -> List[str]:
        """Initialize twelve links of dependent origination"""
        
        return [
            "ignorance",
            "mental formations", 
            "consciousness",
            "name and form",
            "six sense bases",
            "contact",
            "feeling",
            "craving",
            "grasping",
            "becoming",
            "birth",
            "aging and death"
        ]
    
    def _initialize_noble_truths(self) -> List[str]:
        """Initialize four noble truths"""
        
        return [
            "suffering exists",
            "suffering has a cause",
            "suffering can end",
            "there is a path to end suffering"
        ]
    
    def _initialize_eightfold_path(self) -> List[str]:
        """Initialize eightfold path"""
        
        return [
            "right view",
            "right intention",
            "right speech",
            "right action",
            "right livelihood",
            "right effort",
            "right mindfulness",
            "right concentration"
        ]
    
    def analyze_consciousness_flow(self, start_skandha: str, 
                                  end_skandha: str) -> Dict[str, Any]:
        """Analyze consciousness flow between skandhas"""
        
        if start_skandha not in self.skandhas or end_skandha not in self.skandhas:
            return {}
        
        start_obj = self.skandhas[start_skandha]
        end_obj = self.skandhas[end_skandha]
        
        # Calculate transition properties
        transition = {
            'start_skandha': start_obj,
            'end_skandha': end_obj,
            'consciousness_change': end_obj.phi_value - start_obj.phi_value,
            'emptiness_change': end_obj.emptiness_degree - start_obj.emptiness_degree,
            'non_duality_change': end_obj.non_duality_score - start_obj.non_duality_score,
            'transition_direction': 'evolution' if end_obj.phi_value > start_obj.phi_value else 'regression'
        }
        
        return transition
    
    def get_liberation_indicators(self) -> Dict[str, float]:
        """Get indicators of liberation progress"""
        
        indicators = {}
        
        # Average skandha properties
        avg_phi = np.mean([skandha.phi_value for skandha in self.skandhas.values()])
        avg_emptiness = np.mean([skandha.emptiness_degree for skandha in self.skandhas.values()])
        avg_non_duality = np.mean([skandha.non_duality_score for skandha in self.skandhas.values()])
        
        indicators['average_phi'] = avg_phi
        indicators['average_emptiness'] = avg_emptiness
        indicators['average_non_duality'] = avg_non_duality
        
        # Liberation progress (higher values = more liberated)
        liberation_progress = (avg_phi + avg_emptiness + avg_non_duality) / 3.0
        indicators['liberation_progress'] = liberation_progress
        
        return indicators 