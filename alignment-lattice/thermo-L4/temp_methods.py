
    def _parse_S(self, s: str) -> dict:
        """Parse structured S into dictionary"""
        parts = [p.strip() for p in (s or "").split("|")]
        d = {}
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                d[k.strip()] = v.strip()
        return d

    def calculate_slot_fidelity(self, S_dict, reconstruction, full_context) -> float:
        """Score fidelity by checking specific slots instead of vague similarity."""
        rec = (reconstruction or "").lower()
        score = 0.0
        
        # L0-L3 verbatim handles (40% of score)
        for level in ["l0", "l1", "l2", "l3"]:
            if level in S_dict:
                phrase = S_dict[level].replace("_", " ").lower()
                if phrase and phrase in rec:
                    score += 0.10
        
        # Novelty mapping (15%)
        novelty_map = {
            "ugf": "unified gradient flow",
            "sim": "simultaneous processing", 
            "cli": "cross-level invariant",
            "raf": "reduced attention fragmentation"
        }
        l4n = S_dict.get("l4n", "")
        if l4n in novelty_map and novelty_map[l4n] in rec:
            score += 0.15
        
        # Deltas mentioned (15%)
        if all(x in rec for x in ["latency", "entropy", "token"]) and ("energy" in rec or "free energy" in rec):
            score += 0.15
        
        # Testable prediction (10%)
        if "test:" in rec or any(w in rec for w in ["predict", "measure", "should", "if "]):
            score += 0.10
        
        # Base similarity anchor (20%)
        base = self.calculate_combined_fidelity(full_context, reconstruction)
        score += 0.20 * base
        
        return min(score, 1.0)

