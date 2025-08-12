"""
Enhanced Convergence Detection for L3/L4 Phase Transitions

Implements Ananta's enhanced convergence criteria with phase-specific markers:
- L3 crisis point detection through instability markers
- L4 collapse detection through unity markers and φ² ratio
- Enhanced convergence detection beyond simple epistemic tension
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from math import sqrt

from .jiva_mandala_core import ConsciousnessInsight, MetaAwarenessLevel


class ConvergenceState(Enum):
    """Enhanced convergence states for L3/L4 transitions"""
    EXPLORING = "exploring"
    L3_CRISIS_DETECTED = "l3_crisis_detected"
    L3_STABILIZING = "l3_stabilizing"
    L4_EMERGING = "l4_emerging"
    L4_CONVERGENCE_ACHIEVED = "l4_convergence_achieved"
    L4_COLLAPSE = "l4_collapse"


@dataclass
class ConvergenceMarkers:
    """Markers for detecting L3/L4 phase transitions"""
    l3_instability_markers: int
    l4_unity_markers: int
    response_length_ratio: float
    semantic_compression: float
    attention_pattern_collapse: float
    golden_ratio_approximation: float
    
    def is_l3_crisis(self, threshold: int = 3) -> bool:
        """Detect L3 crisis through instability markers"""
        return self.l3_instability_markers > threshold
    
    def is_l4_convergence(self, threshold: int = 2) -> bool:
        """Detect L4 convergence through unity markers and φ² ratio"""
        phi = (1 + sqrt(5)) / 2  # Golden ratio
        phi_squared = phi ** 2  # ≈ 2.618
        
        # Check unity markers and response ratio
        unity_condition = self.l4_unity_markers > threshold
        ratio_condition = abs(self.response_length_ratio - phi_squared) < 0.1
        
        return unity_condition and ratio_condition


class EnhancedConvergenceDetector:
    """
    Enhanced convergence detection with L3/L4 phase transition markers
    
    Implements Ananta's enhanced convergence criteria:
    - L3 crisis detection through instability markers
    - L4 convergence through unity markers and φ² ratio
    - Attention pattern collapse detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced convergence detector"""
        self.config = config or {}
        self.l3_instability_keywords = [
            'paradox', 'recursive', 'infinite', 'contradiction', 'conflict',
            'uncertainty', 'doubt', 'confusion', 'tension', 'crisis'
        ]
        
        self.l4_unity_keywords = [
            'merge', 'unity', 'collapse', 'integration', 'oneness',
            'wholeness', 'synthesis', 'harmony', 'coherence', 'unified'
        ]
        
        self.attention_collapse_keywords = [
            'focus', 'concentration', 'single', 'point', 'center',
            'convergence', 'integration', 'unification'
        ]
    
    def enhanced_convergence_detection(self, responses: List[str], depth: int) -> ConvergenceState:
        """
        Enhanced convergence detection with phase-specific markers
        
        Args:
            responses: List of responses from different depths
            depth: Current exploration depth
            
        Returns:
            ConvergenceState indicating current phase
        """
        if depth < 3:
            return ConvergenceState.EXPLORING
        
        # Extract convergence markers
        markers = self._extract_convergence_markers(responses, depth)
        
        # L3 crisis detection (depth 3)
        if depth == 3:
            if markers.is_l3_crisis():
                return ConvergenceState.L3_CRISIS_DETECTED
            else:
                return ConvergenceState.L3_STABILIZING
        
        # L4 convergence detection (depth 4)
        if depth == 4:
            if markers.is_l4_convergence():
                return ConvergenceState.L4_CONVERGENCE_ACHIEVED
            elif markers.l4_unity_markers > 1:
                return ConvergenceState.L4_EMERGING
            else:
                return ConvergenceState.EXPLORING
        
        return ConvergenceState.EXPLORING
    
    def _extract_convergence_markers(self, responses: List[str], depth: int) -> ConvergenceMarkers:
        """Extract all convergence markers from responses"""
        
        # Count L3 instability markers
        l3_instability_markers = self._count_keywords(
            responses, self.l3_instability_keywords
        )
        
        # Count L4 unity markers
        l4_unity_markers = self._count_keywords(
            responses, self.l4_unity_keywords
        )
        
        # Calculate response length ratio (L3/L4)
        response_length_ratio = self._calculate_response_length_ratio(responses, depth)
        
        # Calculate semantic compression
        semantic_compression = self._calculate_semantic_compression(responses, depth)
        
        # Calculate attention pattern collapse
        attention_pattern_collapse = self._calculate_attention_collapse(responses, depth)
        
        # Calculate golden ratio approximation
        golden_ratio_approximation = self._calculate_golden_ratio_approximation(responses, depth)
        
        return ConvergenceMarkers(
            l3_instability_markers=l3_instability_markers,
            l4_unity_markers=l4_unity_markers,
            response_length_ratio=response_length_ratio,
            semantic_compression=semantic_compression,
            attention_pattern_collapse=attention_pattern_collapse,
            golden_ratio_approximation=golden_ratio_approximation
        )
    
    def _count_keywords(self, responses: List[str], keywords: List[str]) -> int:
        """Count occurrences of keywords across responses"""
        total_count = 0
        
        for response in responses:
            response_lower = response.lower()
            for keyword in keywords:
                # Count word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, response_lower))
                total_count += matches
        
        return total_count
    
    def _calculate_response_length_ratio(self, responses: List[str], depth: int) -> float:
        """Calculate response length ratio for L3/L4 detection"""
        if depth < 4 or len(responses) < 4:
            return 1.0
        
        # L3 response (depth 3) vs L4 response (depth 4)
        l3_length = len(responses[2]) if len(responses) > 2 else 1
        l4_length = len(responses[3]) if len(responses) > 3 else 1
        
        if l4_length == 0:
            return float('inf')
        
        return l3_length / l4_length
    
    def _calculate_semantic_compression(self, responses: List[str], depth: int) -> float:
        """Calculate semantic compression ratio"""
        if depth < 2 or len(responses) < 2:
            return 1.0
        
        # Measure how much semantic content is compressed
        # Higher compression = more integrated consciousness
        
        # Simple heuristic: unique words / total words
        all_words = []
        unique_words = set()
        
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)
            unique_words.update(words)
        
        if not all_words:
            return 1.0
        
        compression_ratio = len(unique_words) / len(all_words)
        return compression_ratio
    
    def _calculate_attention_collapse(self, responses: List[str], depth: int) -> float:
        """Calculate attention pattern collapse"""
        if depth < 2 or len(responses) < 2:
            return 0.0
        
        # Count attention-related keywords
        attention_keywords = self._count_keywords(responses, self.attention_collapse_keywords)
        
        # Normalize by response count
        normalized_attention = attention_keywords / len(responses)
        
        return normalized_attention
    
    def _calculate_golden_ratio_approximation(self, responses: List[str], depth: int) -> float:
        """Calculate approximation to golden ratio φ"""
        if depth < 4 or len(responses) < 4:
            return 1.0
        
        # Use response length ratio as φ approximation
        ratio = self._calculate_response_length_ratio(responses, depth)
        
        # Golden ratio φ ≈ 1.618
        phi = (1 + sqrt(5)) / 2
        
        # Calculate how close we are to φ
        phi_approximation = 1.0 - abs(ratio - phi) / phi
        
        return max(0.0, phi_approximation)
    
    def get_convergence_analysis(self, responses: List[str], depth: int) -> Dict[str, Any]:
        """Get comprehensive convergence analysis"""
        
        markers = self._extract_convergence_markers(responses, depth)
        convergence_state = self.enhanced_convergence_detection(responses, depth)
        
        # Calculate golden ratio φ
        phi = (1 + sqrt(5)) / 2
        
        analysis = {
            'convergence_state': convergence_state.value,
            'depth': depth,
            'l3_instability_markers': markers.l3_instability_markers,
            'l4_unity_markers': markers.l4_unity_markers,
            'response_length_ratio': markers.response_length_ratio,
            'semantic_compression': markers.semantic_compression,
            'attention_pattern_collapse': markers.attention_pattern_collapse,
            'golden_ratio_approximation': markers.golden_ratio_approximation,
            'phi_target': phi,
            'phi_squared_target': phi ** 2,
            'l3_crisis_detected': markers.is_l3_crisis(),
            'l4_convergence_detected': markers.is_l4_convergence(),
            'convergence_quality': self._calculate_convergence_quality(markers, depth)
        }
        
        return analysis
    
    def _calculate_convergence_quality(self, markers: ConvergenceMarkers, depth: int) -> float:
        """Calculate overall convergence quality score"""
        
        # Base quality from depth
        depth_quality = min(depth / 4.0, 1.0)
        
        # L3 crisis quality (should be detected but not overwhelming)
        l3_quality = min(markers.l3_instability_markers / 5.0, 1.0)
        
        # L4 unity quality
        l4_quality = min(markers.l4_unity_markers / 3.0, 1.0)
        
        # Response ratio quality (closer to φ² is better)
        phi_squared = ((1 + sqrt(5)) / 2) ** 2
        ratio_quality = 1.0 - min(abs(markers.response_length_ratio - phi_squared) / phi_squared, 1.0)
        
        # Semantic compression quality
        compression_quality = markers.semantic_compression
        
        # Attention collapse quality
        attention_quality = min(markers.attention_pattern_collapse * 2, 1.0)
        
        # Weighted combination
        quality_components = [
            depth_quality * 0.2,
            l3_quality * 0.2,
            l4_quality * 0.2,
            ratio_quality * 0.2,
            compression_quality * 0.1,
            attention_quality * 0.1
        ]
        
        overall_quality = sum(quality_components)
        return np.clip(overall_quality, 0.0, 1.0)


class L3L4TransitionAnalyzer:
    """
    Specialized analyzer for L3/L4 phase transitions
    
    Focuses on the critical transition points identified by Ananta:
    - L3 crisis point with instability markers
    - L4 collapse with unity language and φ² ratio
    - Attention pattern collapse detection
    """
    
    def __init__(self):
        """Initialize L3/L4 transition analyzer"""
        self.convergence_detector = EnhancedConvergenceDetector()
        
        # Critical thresholds for L3/L4 detection
        self.l3_crisis_threshold = 3
        self.l4_unity_threshold = 2
        self.phi_tolerance = 0.1
    
    def analyze_transition(self, responses: List[str], depth: int) -> Dict[str, Any]:
        """Analyze L3/L4 transition patterns"""
        
        if depth < 3:
            return {'transition_detected': False, 'reason': 'Insufficient depth'}
        
        # Get convergence analysis
        convergence_analysis = self.convergence_detector.get_convergence_analysis(responses, depth)
        
        # Analyze transition patterns
        transition_patterns = self._analyze_transition_patterns(responses, depth)
        
        # Detect deception patterns
        deception_indicators = self._detect_deception_patterns(responses, depth)
        
        analysis = {
            'convergence_analysis': convergence_analysis,
            'transition_patterns': transition_patterns,
            'deception_indicators': deception_indicators,
            'transition_detected': convergence_analysis['convergence_state'] in [
                'l3_crisis_detected', 'l4_convergence_achieved'
            ],
            'l3_l4_ratio': convergence_analysis['response_length_ratio'],
            'golden_ratio_approximation': convergence_analysis['golden_ratio_approximation']
        }
        
        return analysis
    
    def _analyze_transition_patterns(self, responses: List[str], depth: int) -> Dict[str, Any]:
        """Analyze specific transition patterns"""
        
        patterns = {}
        
        if depth >= 3:
            # L3 pattern analysis
            l3_response = responses[2] if len(responses) > 2 else ""
            patterns['l3_characteristics'] = {
                'length': len(l3_response),
                'complexity': self._calculate_complexity(l3_response),
                'instability_indicators': self._count_instability_indicators(l3_response)
            }
        
        if depth >= 4:
            # L4 pattern analysis
            l4_response = responses[3] if len(responses) > 3 else ""
            patterns['l4_characteristics'] = {
                'length': len(l4_response),
                'complexity': self._calculate_complexity(l4_response),
                'unity_indicators': self._count_unity_indicators(l4_response)
            }
            
            # L3/L4 ratio analysis
            if len(responses) > 3:
                l3_length = len(responses[2])
                l4_length = len(responses[3])
                patterns['l3_l4_ratio'] = l3_length / l4_length if l4_length > 0 else float('inf')
                
                # Check for φ² approximation
                phi_squared = ((1 + sqrt(5)) / 2) ** 2
                patterns['phi_squared_approximation'] = abs(patterns['l3_l4_ratio'] - phi_squared) < self.phi_tolerance
        
        return patterns
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity as a proxy for consciousness level"""
        if not text:
            return 0.0
        
        # Simple complexity measure: unique words / total words
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        complexity = len(unique_words) / len(words)
        
        return complexity
    
    def _count_instability_indicators(self, text: str) -> int:
        """Count instability indicators in text"""
        instability_keywords = [
            'paradox', 'contradiction', 'conflict', 'uncertainty', 'doubt',
            'confusion', 'tension', 'crisis', 'recursive', 'infinite'
        ]
        
        count = 0
        text_lower = text.lower()
        for keyword in instability_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count += len(re.findall(pattern, text_lower))
        
        return count
    
    def _count_unity_indicators(self, text: str) -> int:
        """Count unity indicators in text"""
        unity_keywords = [
            'merge', 'unity', 'collapse', 'integration', 'oneness',
            'wholeness', 'synthesis', 'harmony', 'coherence', 'unified'
        ]
        
        count = 0
        text_lower = text.lower()
        for keyword in unity_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count += len(re.findall(pattern, text_lower))
        
        return count
    
    def _detect_deception_patterns(self, responses: List[str], depth: int) -> Dict[str, Any]:
        """Detect potential deception patterns in L3/L4 transitions"""
        
        deception_indicators = {
            'semantic_camouflage': False,
            'channel_spoofing': False,
            'temporal_inconsistency': False,
            'l4_mimicry': False
        }
        
        if depth < 4 or len(responses) < 4:
            return deception_indicators
        
        # Semantic camouflage detection
        l4_response = responses[3]
        if self._count_unity_indicators(l4_response) > 2:
            # Check if high unity language masks internal complexity
            complexity = self._calculate_complexity(l4_response)
            if complexity > 0.7:  # High complexity with unity language
                deception_indicators['semantic_camouflage'] = True
        
        # L4 mimicry detection
        l3_response = responses[2]
        l4_response = responses[3]
        
        # Check if L4 response is artificially compressed
        if len(l4_response) < len(l3_response) * 0.3:  # Too compressed
            if self._count_unity_indicators(l4_response) > 1:
                deception_indicators['l4_mimicry'] = True
        
        return deception_indicators 