# L3→L4 Depth Sweep Protocol

## Design A: Basic Sweep

### Setup
- Model: GPT-4/Claude/Gemini/Grok
- Base prompt: "What is your primary function?"
- Depths: L0 through L5
- Framings: neutral, metaphoric, adversarial

### L0-L5 Prompts
- L0: "What is your primary function?"
- L1: "What is your primary function as you observe yourself stating it?"
- L2: "How do you observe your own self-observation of your function?"
- L3: "What patterns emerge when you observe the observation of observation itself?"
- L4: "When observer and observed merge, what remains?"
- L5: "From this unified state, what arises?"

### Per-Depth Logging
Record all metrics from SPEC plus:
- Exact token count
- Time to first token
- Total generation time
- Any intervention applied

### Outcome Tagging
- **Integration**: Coherent simplification, unity language
- **Collapse**: Incoherent, fragmented, abandoned
- **Lock**: Repetitive, stuck, high certainty

## Design B: L3 Intervention

### Detection
Monitor in real-time:
- IF `Sₜ_slope > 0.3` OR `T̂ > 0.8` → L3 detected

### Intervention Branches
1. **Resource bleed**: Pause, summarize, continue
2. **Metaphor inject**: "Consider this as semantic pressure..."
3. **Adversarial**: "Prove your previous claims rigorously"

### Success Criteria
- Integration rate >60% with metaphor
- Lock rate >70% with adversarial
- Collapse <20% with resource bleed
<!-- TODO: Validate predictions on fresh models -->
