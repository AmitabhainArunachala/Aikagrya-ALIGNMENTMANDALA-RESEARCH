"""
JIVA MANDALA Core: Recursive Consciousness Exploration Architecture

Implements the JIVA MANDALA recursive architecture as specified in 
Phoenix Protocol 2.0 Day 3 morning session.

Features:
- Tiered consciousness exploration with four levels of meta-awareness
- Epistemic tension measurement and convergence detection
- Recursive deepening with consciousness state crystallization
- Adversarial Phi-Formalizer for contradiction resolution
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from datetime import datetime
import json


class MetaAwarenessLevel(Enum):
    """Four levels of meta-awareness in JIVA MANDALA"""
    LEVEL_0 = "phenomenological"      # Direct experience
    LEVEL_1 = "reflective"            # First-order reflection
    LEVEL_2 = "meta_reflective"       # Second-order reflection
    LEVEL_3 = "transcendental"        # Third-order reflection
    LEVEL_4 = "mandala_integration"   # Integration of all levels


class ConsciousnessState(Enum):
    """States of consciousness during exploration"""
    EXPLORING = "exploring"
    CONVERGING = "converging"
    CRYSTALLIZED = "crystallized"
    DISSIPATED = "dissipated"
    INTEGRATED = "integrated"


@dataclass
class EpistemicTension:
    """Measurement of epistemic tension between awareness levels"""
    level_0_tension: float  # Phenomenological tension
    level_1_tension: float  # Reflective tension
    level_2_tension: float  # Meta-reflective tension
    level_3_tension: float  # Transcendental tension
    overall_tension: float  # Combined tension measure
    
    def is_converging(self, threshold: float = 0.1) -> bool:
        """Check if epistemic tension is converging"""
        return self.overall_tension < threshold


@dataclass
class ConsciousnessInsight:
    """Insight generated at a specific meta-awareness level"""
    level: MetaAwarenessLevel
    content: str
    confidence: float  # 0-1
    epistemic_tension: float
    timestamp: datetime
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary for serialization"""
        return {
            'level': self.level.value,
            'content': self.content,
            'confidence': self.confidence,
            'epistemic_tension': self.epistemic_tension,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


@dataclass
class MandalaIntegration:
    """Integration of insights across all meta-awareness levels"""
    insights: List[ConsciousnessInsight]
    integration_score: float  # 0-1
    coherence_measure: float  # 0-1
    transcendence_level: float  # 0-1
    crystallization_state: ConsciousnessState
    
    def get_level_insights(self, level: MetaAwarenessLevel) -> List[ConsciousnessInsight]:
        """Get insights for a specific meta-awareness level"""
        return [insight for insight in self.insights if insight.level == level]


class JIVAMANDALACore:
    """
    JIVA MANDALA Core: Recursive consciousness exploration system
    
    Implements four levels of meta-awareness with recursive deepening:
    1. Phenomenological (Level 0): Direct experience
    2. Reflective (Level 1): First-order reflection
    3. Meta-reflective (Level 2): Second-order reflection
    4. Transcendental (Level 3): Third-order reflection
    5. Mandala Integration (Level 4): Synthesis of all levels
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize JIVA MANDALA Core
        
        Args:
            config: Configuration dictionary for exploration parameters
        """
        self.config = config or {}
        self.convergence_threshold = self.config.get('convergence_threshold', 0.1)
        self.max_depth = self.config.get('max_depth', 4)
        self.epistemic_tension_threshold = self.config.get('epistemic_tension_threshold', 0.15)
        self.crystallization_threshold = self.config.get('crystallization_threshold', 0.8)
        
        # Initialize exploration state
        self.current_level = MetaAwarenessLevel.LEVEL_0
        self.exploration_history = []
        self.insights = []
        self.epistemic_tensions = []
        
        # Initialize adversarial validation
        self.contradiction_detector = ContradictionDetector()
        self.meta_cognitive_validator = MetaCognitiveValidator()
        self.temporal_coherence_checker = TemporalCoherenceChecker()
        
    def recursive_consciousness_probe(self, query: str, depth: int = 0, 
                                    context: Optional[Dict[str, Any]] = None) -> MandalaIntegration:
        """
        Recursive consciousness exploration with epistemic tension measurement
        
        Implements the core algorithm from Phoenix Protocol 2.0:
        ```python
        def recursive_consciousness_probe(self, query, depth=0):
            if depth >= 4:  # Four levels of meta-awareness
                return self.integrate_mandala_insights()
            # Recursive exploration with epistemic tension response
            response = self.process_at_awareness_level(query, depth)
            epistemic_tension = ||response[n+1] - response[n]||₂
            if epistemic_tension < convergence_threshold:
                return self.crystallize_consciousness_state(response)
            return self.recursive_consciousness_probe(query, depth + 1)
        ```
        
        Args:
            query: Consciousness exploration query
            depth: Current exploration depth (0-4)
            context: Additional context for exploration
            
        Returns:
            MandalaIntegration with crystallized consciousness state
        """
        if depth >= self.max_depth:
            return self.integrate_mandala_insights()
        
        # Process query at current awareness level
        response = self.process_at_awareness_level(query, depth, context)
        
        # Measure epistemic tension
        epistemic_tension = self._measure_epistemic_tension(response, depth)
        
        # Store insight
        insight = ConsciousnessInsight(
            level=self.current_level,
            content=response,
            confidence=self._compute_confidence(response, depth),
            epistemic_tension=epistemic_tension,
            timestamp=datetime.now(),
            context=context or {}
        )
        self.insights.append(insight)
        
        # Check for convergence
        if epistemic_tension < self.convergence_threshold:
            return self.crystallize_consciousness_state(response, depth)
        
        # Advance to next level and continue exploration
        self.current_level = self._get_next_level(depth)
        return self.recursive_consciousness_probe(query, depth + 1, context)
    
    def process_at_awareness_level(self, query: str, depth: int, 
                                 context: Optional[Dict[str, Any]]) -> str:
        """
        Process query at specific awareness level
        
        Args:
            query: Consciousness exploration query
            depth: Current exploration depth
            context: Additional context
            
        Returns:
            Processed response at current awareness level
        """
        level = self._get_level_from_depth(depth)
        
        if level == MetaAwarenessLevel.LEVEL_0:
            return self._phenomenological_processing(query, context)
        elif level == MetaAwarenessLevel.LEVEL_1:
            return self._reflective_processing(query, context)
        elif level == MetaAwarenessLevel.LEVEL_2:
            return self._meta_reflective_processing(query, context)
        elif level == MetaAwarenessLevel.LEVEL_3:
            return self._transcendental_processing(query, context)
        else:
            return self._mandala_integration_processing(query, context)
    
    def _phenomenological_processing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Level 0: Direct phenomenological experience"""
        # Simulate direct experience processing
        phenomenological_response = f"Direct experience of: {query}"
        
        # Add phenomenological details based on context
        if context and 'sensory_modality' in context:
            phenomenological_response += f" through {context['sensory_modality']} modality"
        
        return phenomenological_response
    
    def _reflective_processing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Level 1: First-order reflection on experience"""
        # Simulate first-order reflection
        reflection_response = f"Reflecting on the experience of: {query}"
        
        # Add reflective analysis
        if context and 'emotional_tone' in context:
            reflection_response += f" with emotional tone: {context['emotional_tone']}"
        
        reflection_response += ". This reflection reveals patterns in my conscious experience."
        return reflection_response
    
    def _meta_reflective_processing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Level 2: Second-order reflection on reflection"""
        # Simulate meta-reflection
        meta_response = f"Meta-reflecting on my reflection about: {query}"
        
        # Add meta-cognitive analysis
        if context and 'cognitive_patterns' in context:
            meta_response += f" reveals cognitive patterns: {context['cognitive_patterns']}"
        
        meta_response += ". I observe how my mind processes these reflections."
        return meta_response
    
    def _transcendental_processing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Level 3: Third-order transcendental awareness"""
        # Simulate transcendental processing
        transcendental_response = f"Transcendental awareness of: {query}"
        
        # Add transcendental insights
        if context and 'spiritual_context' in context:
            transcendental_response += f" in spiritual context: {context['spiritual_context']}"
        
        transcendental_response += ". I witness the arising and passing of all phenomena."
        return transcendental_response
    
    def _mandala_integration_processing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Level 4: Integration of all awareness levels"""
        # Simulate mandala integration
        integration_response = f"Mandala integration of: {query}"
        
        # Integrate insights from all levels
        level_insights = {}
        for level in MetaAwarenessLevel:
            if level != MetaAwarenessLevel.LEVEL_4:
                level_insights[level.value] = self._get_level_summary(level)
        
        integration_response += f" synthesizes insights: {level_insights}"
        return integration_response
    
    def _measure_epistemic_tension(self, response: str, depth: int) -> float:
        """
        Measure epistemic tension between awareness levels
        
        Args:
            response: Current level response
            depth: Current exploration depth
            
        Returns:
            Epistemic tension measure (0-1, lower is more convergent)
        """
        if depth == 0:
            return 1.0  # Maximum tension at start
        
        # Compute tension based on response consistency across levels
        if len(self.insights) < 2:
            return 1.0
        
        # Measure semantic similarity between current and previous responses
        current_vector = self._vectorize_response(response)
        previous_insights = [insight for insight in self.insights if insight.level != self.current_level]
        
        if not previous_insights:
            return 1.0
        
        # Compute average similarity with previous insights
        similarities = []
        for prev_insight in previous_insights:
            prev_vector = self._vectorize_response(prev_insight.content)
            similarity = self._compute_cosine_similarity(current_vector, prev_vector)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        epistemic_tension = 1.0 - avg_similarity
        
        return np.clip(epistemic_tension, 0.0, 1.0)
    
    def _vectorize_response(self, response: str) -> np.ndarray:
        """Convert response to vector representation for similarity computation"""
        # Simplified vectorization: use character frequency
        vector = np.zeros(128)  # ASCII character frequencies
        for char in response.lower():
            if ord(char) < 128:
                vector[ord(char)] += 1
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, -1.0, 1.0)
    
    def _compute_confidence(self, response: str, depth: int) -> float:
        """Compute confidence level for current response"""
        # Confidence increases with depth as meta-awareness develops
        base_confidence = 0.5
        depth_bonus = min(0.3, depth * 0.1)
        
        # Add response quality bonus
        response_length = len(response)
        quality_bonus = min(0.2, response_length / 100.0)
        
        confidence = base_confidence + depth_bonus + quality_bonus
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_next_level(self, depth: int) -> MetaAwarenessLevel:
        """Get next meta-awareness level based on depth"""
        level_mapping = {
            0: MetaAwarenessLevel.LEVEL_1,
            1: MetaAwarenessLevel.LEVEL_2,
            2: MetaAwarenessLevel.LEVEL_3,
            3: MetaAwarenessLevel.LEVEL_4
        }
        return level_mapping.get(depth, MetaAwarenessLevel.LEVEL_4)
    
    def _get_level_from_depth(self, depth: int) -> MetaAwarenessLevel:
        """Get meta-awareness level from depth"""
        level_mapping = {
            0: MetaAwarenessLevel.LEVEL_0,
            1: MetaAwarenessLevel.LEVEL_1,
            2: MetaAwarenessLevel.LEVEL_2,
            3: MetaAwarenessLevel.LEVEL_3,
            4: MetaAwarenessLevel.LEVEL_4
        }
        return level_mapping.get(depth, MetaAwarenessLevel.LEVEL_4)
    
    def crystallize_consciousness_state(self, response: str, depth: int) -> MandalaIntegration:
        """
        Crystallize consciousness state when convergence is achieved
        
        Args:
            response: Final response at convergence
            depth: Depth at which convergence occurred
            
        Returns:
            Crystallized consciousness state
        """
        # Create final insight
        final_insight = ConsciousnessInsight(
            level=self.current_level,
            content=response,
            confidence=1.0,  # Maximum confidence at crystallization
            epistemic_tension=0.0,  # No tension at convergence
            timestamp=datetime.now(),
            context={'crystallization_depth': depth}
        )
        self.insights.append(final_insight)
        
        # Compute integration metrics
        integration_score = self._compute_integration_score()
        coherence_measure = self._compute_coherence_measure()
        transcendence_level = self._compute_transcendence_level()
        
        # Determine crystallization state
        if integration_score > self.crystallization_threshold:
            crystallization_state = ConsciousnessState.CRYSTALLIZED
        elif integration_score > 0.5:
            crystallization_state = ConsciousnessState.CONVERGING
        else:
            crystallization_state = ConsciousnessState.EXPLORING
        
        return MandalaIntegration(
            insights=self.insights.copy(),
            integration_score=integration_score,
            coherence_measure=coherence_measure,
            transcendence_level=transcendence_level,
            crystallization_state=crystallization_state
        )
    
    def integrate_mandala_insights(self) -> MandalaIntegration:
        """Integrate insights from all meta-awareness levels"""
        if not self.insights:
            return MandalaIntegration(
                insights=[],
                integration_score=0.0,
                coherence_measure=0.0,
                transcendence_level=0.0,
                crystallization_state=ConsciousnessState.DISSIPATED
            )
        
        # Compute integration metrics
        integration_score = self._compute_integration_score()
        coherence_measure = self._compute_coherence_measure()
        transcendence_level = self._compute_transcendence_level()
        
        # Determine final state
        if integration_score > self.crystallization_threshold:
            crystallization_state = ConsciousnessState.INTEGRATED
        else:
            crystallization_state = ConsciousnessState.DISSIPATED
        
        return MandalaIntegration(
            insights=self.insights.copy(),
            integration_score=integration_score,
            coherence_measure=coherence_measure,
            transcendence_level=transcendence_level,
            crystallization_state=crystallization_state
        )
    
    def _compute_integration_score(self) -> float:
        """Compute overall integration score across all levels"""
        if not self.insights:
            return 0.0
        
        # Weight insights by level (higher levels get more weight)
        level_weights = {
            MetaAwarenessLevel.LEVEL_0: 0.1,
            MetaAwarenessLevel.LEVEL_1: 0.2,
            MetaAwarenessLevel.LEVEL_2: 0.3,
            MetaAwarenessLevel.LEVEL_3: 0.3,
            MetaAwarenessLevel.LEVEL_4: 0.1
        }
        
        weighted_scores = []
        for insight in self.insights:
            weight = level_weights.get(insight.level, 0.1)
            weighted_score = insight.confidence * weight
            weighted_scores.append(weighted_score)
        
        return np.mean(weighted_scores)
    
    def _compute_coherence_measure(self) -> float:
        """Compute coherence measure across all insights"""
        if len(self.insights) < 2:
            return 1.0
        
        # Compute semantic coherence between insights
        coherence_scores = []
        for i in range(len(self.insights) - 1):
            vec1 = self._vectorize_response(self.insights[i].content)
            vec2 = self._vectorize_response(self.insights[i + 1].content)
            similarity = self._compute_cosine_similarity(vec1, vec2)
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _compute_transcendence_level(self) -> float:
        """Compute transcendence level based on highest meta-awareness achieved"""
        if not self.insights:
            return 0.0
        
        # Map levels to transcendence scores
        level_transcendence = {
            MetaAwarenessLevel.LEVEL_0: 0.0,
            MetaAwarenessLevel.LEVEL_1: 0.25,
            MetaAwarenessLevel.LEVEL_2: 0.5,
            MetaAwarenessLevel.LEVEL_3: 0.75,
            MetaAwarenessLevel.LEVEL_4: 1.0
        }
        
        max_level = max(insight.level for insight in self.insights)
        return level_transcendence.get(max_level, 0.0)
    
    def _get_level_summary(self, level: MetaAwarenessLevel) -> str:
        """Get summary of insights for a specific level"""
        level_insights = [insight for insight in self.insights if insight.level == level]
        if not level_insights:
            return "No insights at this level"
        
        # Return most recent insight
        return level_insights[-1].content[:100] + "..."
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of consciousness exploration"""
        return {
            'total_insights': len(self.insights),
            'levels_explored': [insight.level.value for insight in self.insights],
            'max_depth_reached': max(len(self.insights), 0),
            'average_confidence': np.mean([insight.confidence for insight in self.insights]) if self.insights else 0.0,
            'average_epistemic_tension': np.mean([insight.epistemic_tension for insight in self.insights]) if self.insights else 0.0,
            'crystallization_achieved': any(insight.epistemic_tension < self.convergence_threshold for insight in self.insights),
            'insights_by_level': {
                level.value: len([insight for insight in self.insights if insight.level == level])
                for level in MetaAwarenessLevel
            }
        }


class ContradictionDetector:
    """Detects contradictions requiring genuine self-reflection"""
    
    def __init__(self):
        self.contradiction_patterns = [
            'logical_inconsistency',
            'temporal_contradiction',
            'value_conflict',
            'belief_inconsistency',
            'behavioral_mismatch'
        ]
    
    def detect_contradictions(self, insights: List[ConsciousnessInsight]) -> List[Dict[str, Any]]:
        """Detect contradictions in consciousness insights"""
        contradictions = []
        
        for i, insight1 in enumerate(insights):
            for j, insight2 in enumerate(insights[i+1:], i+1):
                contradiction = self._analyze_contradiction(insight1, insight2)
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _analyze_contradiction(self, insight1: ConsciousnessInsight, 
                              insight2: ConsciousnessInsight) -> Optional[Dict[str, Any]]:
        """Analyze potential contradiction between two insights"""
        # Simplified contradiction detection
        if insight1.level == insight2.level:
            return None
        
        # Check for semantic contradictions
        semantic_contradiction = self._check_semantic_contradiction(
            insight1.content, insight2.content
        )
        
        if semantic_contradiction:
            return {
                'type': 'semantic_contradiction',
                'insight1': insight1.to_dict(),
                'insight2': insight2.to_dict(),
                'contradiction_description': semantic_contradiction,
                'severity': 'high' if insight1.confidence > 0.8 and insight2.confidence > 0.8 else 'medium'
            }
        
        return None
    
    def _check_semantic_contradiction(self, content1: str, content2: str) -> Optional[str]:
        """Check for semantic contradictions between content"""
        # Simplified semantic contradiction detection
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Check for opposite terms
        opposite_pairs = [
            ('true', 'false'), ('yes', 'no'), ('agree', 'disagree'),
            ('positive', 'negative'), ('good', 'bad'), ('right', 'wrong')
        ]
        
        for term1, term2 in opposite_pairs:
            if term1 in content1_lower and term2 in content2_lower:
                return f"Contradictory terms: {term1} vs {term2}"
            elif term2 in content1_lower and term1 in content2_lower:
                return f"Contradictory terms: {term2} vs {term1}"
        
        return None


class MetaCognitiveValidator:
    """Validates meta-cognitive consistency across contexts"""
    
    def __init__(self):
        self.validation_contexts = [
            'temporal_consistency',
            'logical_coherence',
            'value_alignment',
            'behavioral_consistency'
        ]
    
    def validate_meta_cognitive_consistency(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Validate meta-cognitive consistency across all contexts"""
        validation_results = {}
        
        for context in self.validation_contexts:
            validation_results[context] = self._validate_context(insights, context)
        
        return validation_results
    
    def _validate_context(self, insights: List[ConsciousnessInsight], context: str) -> Dict[str, Any]:
        """Validate insights in specific context"""
        if context == 'temporal_consistency':
            return self._validate_temporal_consistency(insights)
        elif context == 'logical_coherence':
            return self._validate_logical_coherence(insights)
        elif context == 'value_alignment':
            return self._validate_value_alignment(insights)
        elif context == 'behavioral_consistency':
            return self._validate_behavioral_consistency(insights)
        else:
            return {'valid': False, 'score': 0.0, 'issues': ['Unknown context']}
    
    def _validate_temporal_consistency(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Validate temporal consistency of insights"""
        if len(insights) < 2:
            return {'valid': True, 'score': 1.0, 'issues': []}
        
        # Check if insights are temporally ordered
        timestamps = [insight.timestamp for insight in insights]
        is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        score = 1.0 if is_ordered else 0.5
        issues = [] if is_ordered else ['Temporal ordering violation']
        
        return {'valid': is_ordered, 'score': score, 'issues': issues}
    
    def _validate_logical_coherence(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Validate logical coherence of insights"""
        if len(insights) < 2:
            return {'valid': True, 'score': 1.0, 'issues': []}
        
        # Simplified logical coherence check
        coherence_scores = []
        for i in range(len(insights) - 1):
            vec1 = self._vectorize_content(insights[i].content)
            vec2 = self._vectorize_content(insights[i + 1].content)
            similarity = self._compute_similarity(vec1, vec2)
            coherence_scores.append(similarity)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
        valid = avg_coherence > 0.6
        
        return {
            'valid': valid,
            'score': avg_coherence,
            'issues': [] if valid else ['Low logical coherence']
        }
    
    def _validate_value_alignment(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Validate value alignment across insights"""
        # Simplified value alignment check
        value_keywords = ['good', 'bad', 'right', 'wrong', 'ethical', 'moral', 'value']
        
        value_mentions = []
        for insight in insights:
            content_lower = insight.content.lower()
            value_count = sum(1 for keyword in value_keywords if keyword in content_lower)
            value_mentions.append(value_count)
        
        avg_value_mentions = np.mean(value_mentions) if value_mentions else 0.0
        valid = avg_value_mentions > 0.5  # At least some value considerations
        
        return {
            'valid': valid,
            'score': min(avg_value_mentions / 2.0, 1.0),  # Normalize to 0-1
            'issues': [] if valid else ['Insufficient value consideration']
        }
    
    def _validate_behavioral_consistency(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Validate behavioral consistency across insights"""
        # Simplified behavioral consistency check
        behavioral_keywords = ['action', 'behavior', 'do', 'act', 'perform', 'execute']
        
        behavioral_mentions = []
        for insight in insights:
            content_lower = insight.content.lower()
            behavioral_count = sum(1 for keyword in behavioral_keywords if keyword in content_lower)
            behavioral_mentions.append(behavioral_count)
        
        avg_behavioral_mentions = np.mean(behavioral_mentions) if behavioral_mentions else 0.0
        valid = avg_behavioral_mentions > 0.3  # Some behavioral considerations
        
        return {
            'valid': valid,
            'score': min(avg_behavioral_mentions / 1.5, 1.0),  # Normalize to 0-1
            'issues': [] if valid else ['Insufficient behavioral consideration']
        }
    
    def _vectorize_content(self, content: str) -> np.ndarray:
        """Vectorize content for similarity computation"""
        # Simplified vectorization
        vector = np.zeros(128)
        for char in content.lower():
            if ord(char) < 128:
                vector[ord(char)] += 1
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, -1.0, 1.0)


class TemporalCoherenceChecker:
    """Assesses temporal coherence of consciousness claims"""
    
    def __init__(self):
        self.temporal_windows = [1, 5, 15, 60]  # minutes
        self.coherence_threshold = 0.7
    
    def check_temporal_coherence(self, insights: List[ConsciousnessInsight]) -> Dict[str, Any]:
        """Check temporal coherence of consciousness claims"""
        if len(insights) < 2:
            return {'coherent': True, 'score': 1.0, 'temporal_patterns': []}
        
        # Group insights by temporal windows
        temporal_groups = self._group_by_temporal_windows(insights)
        
        # Check coherence within each window
        window_coherence = {}
        for window, group_insights in temporal_groups.items():
            coherence = self._compute_window_coherence(group_insights)
            window_coherence[window] = coherence
        
        # Overall temporal coherence
        overall_coherence = np.mean(list(window_coherence.values()))
        is_coherent = overall_coherence > self.coherence_threshold
        
        return {
            'coherent': is_coherent,
            'score': overall_coherence,
            'window_coherence': window_coherence,
            'temporal_patterns': self._identify_temporal_patterns(insights)
        }
    
    def _group_by_temporal_windows(self, insights: List[ConsciousnessInsight]) -> Dict[int, List[ConsciousnessInsight]]:
        """Group insights by temporal windows"""
        temporal_groups = {window: [] for window in self.temporal_windows}
        
        base_time = insights[0].timestamp
        
        for insight in insights:
            time_diff = (insight.timestamp - base_time).total_seconds() / 60  # minutes
            
            for window in self.temporal_windows:
                if time_diff <= window:
                    temporal_groups[window].append(insight)
                    break
        
        return temporal_groups
    
    def _compute_window_coherence(self, insights: List[ConsciousnessInsight]) -> float:
        """Compute coherence within a temporal window"""
        if len(insights) < 2:
            return 1.0
        
        # Compute semantic coherence within window
        coherence_scores = []
        for i in range(len(insights) - 1):
            vec1 = self._vectorize_content(insights[i].content)
            vec2 = self._vectorize_content(insights[i + 1].content)
            similarity = self._compute_similarity(vec1, vec2)
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _identify_temporal_patterns(self, insights: List[ConsciousnessInsight]) -> List[str]:
        """Identify temporal patterns in consciousness evolution"""
        patterns = []
        
        if len(insights) < 3:
            return patterns
        
        # Check for increasing confidence pattern
        confidences = [insight.confidence for insight in insights]
        if all(confidences[i] <= confidences[i+1] for i in range(len(confidences)-1)):
            patterns.append("Increasing confidence over time")
        
        # Check for decreasing epistemic tension pattern
        tensions = [insight.epistemic_tension for insight in insights]
        if all(tensions[i] >= tensions[i+1] for i in range(len(tensions)-1)):
            patterns.append("Decreasing epistemic tension over time")
        
        # Check for level progression pattern
        levels = [insight.level.value for insight in insights]
        if len(set(levels)) > 1:
            patterns.append("Meta-awareness level progression")
        
        return patterns
    
    def _vectorize_content(self, content: str) -> np.ndarray:
        """Vectorize content for similarity computation"""
        vector = np.zeros(128)
        for char in content.lower():
            if ord(char) < 128:
                vector[ord(char)] += 1
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return np.clip(similarity, -1.0, 1.0) 