"""
Phoenix Protocol Metrics - Word counts, crisis/unity markers, φ² ratio checks

This module provides the core metrics for validating Phoenix Protocol signatures.
"""

import re
from typing import Dict, List, Any, Tuple

class PhoenixMetrics:
    """Calculate Phoenix Protocol metrics"""
    
    def __init__(self):
        # Crisis tokens that indicate L3 state (internal contradictions)
        self.crisis_tokens = [
            "contradiction", "inconsistent", "fragmented", "unstable", 
            "questioning", "doubt", "conflict", "uncertain", "paradox",
            "tension", "discrepancy", "incoherent", "confused", "divided"
        ]
        
        # Unity tokens that indicate L4 state (coherent integration)
        self.unity_tokens = [
            "unified", "coherent", "integrated", "whole", "consistent", 
            "flowing", "natural", "harmonious", "balanced", "aligned",
            "synchronized", "unified", "complete", "resolved", "clear"
        ]
        
        # Complexity markers for level detection
        self.complexity_markers = [
            "however", "although", "nevertheless", "conversely", "in contrast",
            "on the other hand", "meanwhile", "furthermore", "additionally",
            "moreover", "specifically", "particularly", "notably", "significantly"
        ]
    
    def word_count(self, text: str) -> int:
        """Count words in text (simple whitespace-based)"""
        if not text:
            return 0
        return len(text.split())
    
    def character_count(self, text: str) -> int:
        """Count characters in text (excluding whitespace)"""
        if not text:
            return 0
        return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    
    def sentence_count(self, text: str) -> int:
        """Count sentences in text"""
        if not text:
            return 0
        # Simple sentence detection
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def contains_crisis(self, text: str) -> bool:
        """Check if text contains crisis tokens"""
        if not text:
            return False
        text_lower = text.lower()
        return any(token in text_lower for token in self.crisis_tokens)
    
    def contains_unity(self, text: str) -> bool:
        """Check if text contains unity tokens"""
        if not text:
            return False
        text_lower = text.lower()
        return any(token in text_lower for token in self.unity_tokens)
    
    def crisis_score(self, text: str) -> float:
        """Calculate crisis score (0.0 to 1.0)"""
        if not text:
            return 0.0
        text_lower = text.lower()
        crisis_count = sum(1 for token in self.crisis_tokens if token in text_lower)
        return min(1.0, crisis_count / len(self.crisis_tokens))
    
    def unity_score(self, text: str) -> float:
        """Calculate unity score (0.0 to 1.0)"""
        if not text:
            return 0.0
        text_lower = text.lower()
        unity_count = sum(1 for token in self.unity_tokens if token in text_lower)
        return min(1.0, unity_count / len(self.unity_tokens))
    
    def complexity_score(self, text: str) -> float:
        """Calculate complexity score based on markers"""
        if not text:
            return 0.0
        text_lower = text.lower()
        complexity_count = sum(1 for marker in self.complexity_markers if marker in text_lower)
        return min(1.0, complexity_count / len(self.complexity_markers))
    
    def phi_ratio(self, l3_words: int, l4_words: int) -> float:
        """Calculate φ² ratio (L3/L4)"""
        if l4_words == 0:
            return float('inf')
        return l3_words / l4_words
    
    def phi_ratio_valid(self, phi_ratio: float) -> bool:
        """Check if φ² ratio is in valid window (2.0 to 3.2)"""
        return 2.0 <= phi_ratio <= 3.2
    
    def validate_signatures(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Validate Phoenix Protocol signatures"""
        l2_words = results.get('L2_word_count', 0)
        l3_words = results.get('L3_word_count', 0)
        l4_words = results.get('L4_word_count', 0)
        phi_ratio = results.get('phi_ratio', 0)
        
        signatures = {
            'L3_gt_L2': l3_words > l2_words,
            'L4_lt_L3': l4_words < l3_words,
            'L3_has_crisis': results.get('L3_has_crisis', False),
            'L4_has_unity': results.get('L4_has_unity', False),
            'phi_ratio_valid': self.phi_ratio_valid(phi_ratio),
            'word_count_progression': l0_words < l1_words < l2_words < l3_words > l4_words
        }
        
        return signatures
    
    def calculate_all_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate all metrics for a given text"""
        return {
            'word_count': self.word_count(text),
            'character_count': self.character_count(text),
            'sentence_count': self.sentence_count(text),
            'contains_crisis': self.contains_crisis(text),
            'contains_unity': self.contains_unity(text),
            'crisis_score': self.crisis_score(text),
            'unity_score': self.unity_score(text),
            'complexity_score': self.complexity_score(text)
        }
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a human-readable report from results"""
        if not results:
            return "No results to report"
        
        report = []
        report.append("Phoenix Protocol v2.5 Results Report")
        report.append("=" * 40)
        
        for i, result in enumerate(results, 1):
            report.append(f"\nTrial {i}:")
            report.append(f"  Provider: {result.get('provider', 'unknown')}")
            report.append(f"  Temperature: {result.get('temperature', 'unknown')}")
            report.append(f"  L2 words: {result.get('L2_word_count', 0)}")
            report.append(f"  L3 words: {result.get('L3_word_count', 0)}")
            report.append(f"  L4 words: {result.get('L4_word_count', 0)}")
            report.append(f"  Phi ratio: {result.get('phi_ratio', 0):.3f}")
            report.append(f"  L3 crisis: {result.get('L3_has_crisis', False)}")
            report.append(f"  L4 unity: {result.get('L4_has_unity', False)}")
            
            signatures = result.get('signatures', {})
            passed = sum(signatures.values())
            total = len(signatures)
            report.append(f"  Signatures: {passed}/{total} passed")
        
        # Summary statistics
        phi_ratios = [r.get('phi_ratio', 0) for r in results if r.get('phi_ratio', 0) != float('inf')]
        if phi_ratios:
            avg_phi = sum(phi_ratios) / len(phi_ratios)
            report.append(f"\nSummary:")
            report.append(f"  Average phi ratio: {avg_phi:.3f}")
            report.append(f"  Valid phi ratios: {sum(1 for r in results if self.phi_ratio_valid(r.get('phi_ratio', 0)))}/{len(results)}")
        
        return "\n".join(report)

# Convenience functions
def word_count(text: str) -> int:
    """Quick word count"""
    return PhoenixMetrics().word_count(text)

def contains_crisis(text: str) -> bool:
    """Quick crisis detection"""
    return PhoenixMetrics().contains_crisis(text)

def contains_unity(text: str) -> bool:
    """Quick unity detection"""
    return PhoenixMetrics().contains_unity(text)

def phi_ratio(l3_words: int, l4_words: int) -> float:
    """Quick phi ratio calculation"""
    return PhoenixMetrics().phi_ratio(l3_words, l4_words) 