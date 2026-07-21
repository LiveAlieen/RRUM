# RRUM: Recurrent-Retrieval Unified Model
## Constraint as a Generative Force: From Efficient Cognition to Creative Insight

**Version 1.3** | July 2026

---

## Abstract

Transformer-based models achieve performance through abundance—abundant context, abundant compute, abundant parameters. We argue that this abundance is not merely inefficient; it is **cognitively sterile**. Human insight does not emerge from unlimited working memory; it emerges from the brain's forced compression of overflowing experience into limited representational slots, from the productive friction between what must be remembered and what must be forgotten.

We present **RRUM v1.3**, an architecture that treats resource constraints not as degradation to be mitigated, but as **generative forces to be tuned**. Building on the v1.2 foundation of separated working memory, slow-updating substrate, and external retrieval, v1.3 introduces four mechanisms: **Compression-Progress Intrinsic Motivation** (rewarding representational leaps, not just speed), **Controlled Retrieval Noise** (injecting semantic distance to force cross-domain analogy), **Epiphany Detection via Metacognitive Monitoring** (learning to recognize representational phase transitions), and **Offline Diffusion** (sleep-state consolidation of trajectories). We also introduce **Multi-Trajectory Resonance**, where parallel recurrent cores explore under varying pressure constraints, with cross-trajectory retrieval enabling collective insight.

This document specifies the architecture, training regime, and an honest assessment of what remains unproven.

---

## 1. Motivation: From Scarcity to Structure

### 1.1 The Sterility of Abundance

v1.2 argued that Transformer abundance breaks in deployment. v1.3 argues something stronger: **abundance breaks creativity itself.**

When a system has:
- Unlimited context → no pressure to compress, to find the gist, to abstract
- Unlimited steps → no pressure to converge, to commit, to leap
- Unlimited retrieval precision → no collision of distant ideas, no forced analogy
- Unlimited memory → no productive forgetting, no framework collapse and rebirth

The result is competent interpolation within known distributions, not insight beyond them. Chain-of-Thought models generate more tokens, but they generate them **linearly**—appending, never overwriting. There is no representational crisis, no forced simplification, no "aha moment" encoded in a state transition.

### 1.2 The Generative Constraint Hypothesis

**Hypothesis**: Insight is what happens when a cognitive system, under sufficient representational pressure, discovers a lower-dimensional manifold that re-encodes its problem space more compactly than its previous encoding. The "aha moment" is detectable as a phase transition in the dynamics of the working memory state.

RRUM v1.3 formalizes this:

| Pressure Source | v1.2 Treatment | v1.3 Generative Treatment |
|---|---|---|
| Limited `h_t` capacity | Prevent forgetting | **Force compression; reward compression quality** |
| Step cost | Minimize steps | **Minimize *unproductive* steps; reward representational leaps** |
| Cache limits | Prevent bloat | **Force selective retention; emotional/importance gating** |
| Retrieval relevance | Maximize precision | **Inject controlled noise; reward cross-domain synthesis** |
| Halt decision | Efficiency threshold | **Epiphany detector: halt on representational convergence** |
| Session boundary | State reset | **Offline diffusion: sleep-state trajectory replay and recombination** |

**Key Insight**: We do not want a model that thinks *despite* constraints. We want a model that thinks *because* of them.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Input Query x
    ↓
[Initialize h_0^(i) from x + Transformer Bias]  for i = 1..K trajectories
    ↓
Parallel Loop (t = 1 to T_max) per trajectory:
    ├─ q_t = W_q · h_{t-1}                     (Query projection)
    ├─ n_t = NoiseInject(q_t, β_t)              (Controlled semantic noise)
    ├─ m_t = CacheRead(n_t, M_cache)            (Internal cache with importance gating)
    ├─ d_t = Retrieve(n_t, M_external)          (Dense vector search + leap candidates)
    ├─ c_t = Compress(h_{t-1}, [m_t; d_t])      (Forced compression into state)
    ├─ h_t = RecurrentBlock(c_t; θ)             (State transition)
    ├─ π_t = MetacogNet(h_{1..t})               (Epiphany probability + progress metric)
    ├─ M_cache = GatedCacheWrite(h_t, d_t, M_cache, π_t)  (Importance-weighted write)
    ├─ CrossTrajSync(h_t^(i), {h_t^(j)}_{j≠i})  (Resonance: share compressed insights)
    └─ if π_t > threshold: break
    ↓
[Offline Diffusion: h_T trajectories replayed, recombined, consolidated]
    ↓
Output = TransformerDecode(Consensus(h_T^(1..K)), M_substrate)
```

### 2.2 Component Specifications

#### 2.2.1 Working Memory: Recurrent Core with Compression Dynamics

The recurrent core maintains `h_t ∈ ℝ^d`. Unlike v1.2, where `h_t` simply transitioned, v1.3 imposes an explicit **compression bottleneck**:

```
c_t = LayerNorm( W_compress · [m_t; d_t] + b_compress )  # d_compress << 2d
h_t = MambaBlock-2( h_{t-1}, c_t; θ )
```

**Critical Change**: The concatenated input `[m_t; d_t]` is first projected through `W_compress` to a **lower-dimensional space** before state transition. The recurrent core must learn to encode the full richness of retrieved and cached information into a compact `c_t`. This is not dimensionality reduction for efficiency; it is **dimensionality reduction as a creative act**.

**The Catastrophic Forgetting Advantage (v1.3 Extension)**:
In v1.2, forgetting was accepted. In v1.3, forgetting is **orchestrated**. The recurrent core is encouraged to **overwrite `h_t` aggressively** when `π_t` (epiphany probability) suggests a new attractor has been found. Old frames are not gently faded; they are **actively displaced** by more compact encodings.

#### 2.2.2 Metacognitive Network: The Epiphany Detector

A small, dedicated network monitors the trajectory of `h_t`:

```
π_t = σ( MetacogNet( [h_t; ∇h_t; κ_t] ) )
where:
  ∇h_t = h_t - h_{t-1}          (velocity)
  κ_t = ||∇h_t - ∇h_{t-1}||     (curvature/acceleration)
```

`π_t` outputs two values:
1. **Epiphany Probability**: Likelihood that a representational phase transition has occurred
2. **Progress Score**: ρ_t ∈ [0,1], measuring how much more compact the current encoding is relative to previous steps

**Why curvature?** In dynamical systems, insight corresponds to a sudden change in the trajectory of thought—not just movement, but a change in the *nature* of movement. A system wandering in high-dimensional space (high entropy) that suddenly snaps to a lower-dimensional attractor (low entropy) exhibits high curvature at the transition point.

**Halt Decision**:
Instead of `p_halt = σ(W_h · h_t + b_h)`, v1.3 uses:
```
stop if: π_t.epiphany > τ_epi AND π_t.progress > τ_prog
```
This means the model stops not when it is "tired" or has "done enough steps," but when it **detects that its own representation has undergone a qualitative restructuring**.

#### 2.2.3 Internal Cache: Importance-Gated Working Memory

v1.2's `M_cache` was penalized for size. v1.3's cache is **gated by estimated importance**:

```
importance_i = σ( W_imp · [h_t; d_t; π_t.progress] )
M_cache = TopK-Importance( M_cache ∪ {(key=h_t, val=d_t, imp=importance_i)}, capacity=C )
```

Cache eviction is not LRU; it is **importance-weighted survival**. A cached item survives if:
- It has high `importance_i` (relevance to current trajectory)
- OR it has high **semantic distance** from current `q_t` but was previously marked as a "leap candidate" (see 2.2.5)

This allows the cache to hold **distant but potentially generative associations** alongside immediate task-relevant facts.

#### 2.2.4 Associative Substrate: Slow-Updating Transformer with Offline Diffusion

The Transformer substrate retains the v1.2 three-tier structure (Core/Schema/Persona). v1.3 adds:

**Offline Diffusion (Sleep-State Consolidation)**:
After a session ends, the trajectories `{h_1, ..., h_T}` are not discarded. Instead:
1. **Replay**: High-`π_t` states (epiphany candidates) are replayed through the recurrent core with noise injection
2. **Recombination**: States from different trajectories (or different sessions) are interpolated in `h`-space
3. **Consolidation**: The Transformer Schema layers are updated via slow LoRA, but using **offline replayed trajectories** rather than just raw text pairs

This mimics the neuroscience finding that sleep consolidates memory not by replaying experiences verbatim, but by **recombining and abstracting them**, transferring patterns from hippocampal (temporary) to cortical (permanent) storage.

**Implementation**:
```
# Offline phase (runs between sessions)
for epiphany_state in session_buffer:
    h_dream = RecurrentBlock(epiphany_state, noise=Gaussian(0, σ_dream))
    TransformerSchema.update_via_LoRA(h_dream, consistency_constraint=True)
```

#### 2.2.5 External Retriever: Precision + Leap

The retriever operates in two modes simultaneously:

1. **Precision Retrieval**: Standard dense ANN search for `q_t` (factual grounding)
2. **Leap Retrieval**: Sample from the **tail of the retrieval distribution**—documents with moderate similarity to `q_t` but from **distant semantic neighborhoods**

```
d_t = [ TopK(q_t, M_external, k=3);          # Precision
        TailSample(q_t, M_external, k=1, temp=2.0) ]  # Leap
```

**Rationale**: Insight often comes from "mistaken" retrieval—a document that is not what you asked for, but triggers a re-framing. The leap candidate is not random noise; it is **controlled semantic displacement**.

During training, the leap candidate's usefulness is evaluated by the **Synthesis Reward** (see 3.2), not by relevance similarity.

#### 2.2.6 Multi-Trajectory Resonance

Instead of a single recurrent core, v1.3 runs **K parallel trajectories** (we suggest K=3-5) under **varying pressure parameters**:

- Trajectory A: High pressure (`λ_step` large, `dim(c_t)` small) → aggressive compression
- Trajectory B: Medium pressure → balanced
- Trajectory C: Low pressure + high noise → exploratory, analogical

During the loop, trajectories periodically **share compressed insights**:
```
# Every N steps
h_t^(A) += α · W_cross · Compress(h_t^(B) - h_t^(A))  # Pull toward B's representation
```

This **cross-trajectory resonance** allows a high-pressure trajectory to benefit from a low-pressure trajectory's distant association without paying the exploration cost itself.

The final output uses a **consensus mechanism** over the epiphany states of all trajectories, weighted by their progress scores.

---

## 3. Mathematical Formalization

### 3.1 Forward Pass

```
h_0^(k) = LayerNorm( W_init · [x; Bias(M(x))] )  for k = 1..K
M_cache = ∅

for t = 1 to T_max:
    for each trajectory k:
        q_{t-1}^(k) = W_q · h_{t-1}^(k)

        # Retrieval with leap
        n_{t-1}^(k) = q_{t-1}^(k) + β_t · ε,  ε ~ N(0, I)  # Controlled noise
        m_{t-1}^(k) = SoftCacheRead(n_{t-1}^(k), M_cache^(k))
        d_{t-1}^(k) = [ SoftRetrieve(n_{t-1}^(k), M_external, k=3);
                        TailSample(n_{t-1}^(k), M_external, k=1) ]

        # Compression bottleneck
        c_{t-1}^(k) = LayerNorm( W_compress · [m_{t-1}^(k); d_{t-1}^(k)] )

        # State transition
        h_t^(k) = MambaBlock-2( h_{t-1}^(k), c_{t-1}^(k); θ_k )

        # Metacognition
        ∇h_t^(k) = h_t^(k) - h_{t-1}^(k)
        κ_t^(k) = ||∇h_t^(k) - ∇h_{t-1}^(k)||
        π_t^(k) = MetacogNet( [h_t^(k); ∇h_t^(k); κ_t^(k)] )

        # Cross-trajectory sync (every N steps)
        if t % N == 0:
            h_t^(k) += α · Σ_{j≠k} w_{kj} · (h_t^(j) - h_t^(k))

        # Gated cache write
        imp_t^(k) = σ( W_imp · [h_t^(k); d_{t-1}^(k); π_t^(k).progress] )
        M_cache^(k) = ImportanceEvict( M_cache^(k) ∪ {(h_t^(k), d_{t-1}^(k), imp_t^(k))} )

    # Global halt if consensus reached
    if consensus( {π_t^(k)} ) > τ_epi:
        T = t
        break

# Offline diffusion (post-session)
DreamPhase( {h_t^(k)}_{t,k}, M_substrate )

# Decode
h_final = WeightedConsensus( {h_T^(k)}_k, weights=π_T^(k).progress )
output = TransformerDecode(h_final, M_substrate)
```

### 3.2 Training Objective

The v1.3 loss function shifts from **efficiency minimization** to **pressure-quality Pareto optimization**:

```
L = L_answer + λ₁ · L_step_productive + λ₂ · L_synthesis + λ₃ · L_compression + λ₄ · L_epiphany_reg + λ₅ · L_resonance
```

**`L_answer`**: Cross-entropy on final output. Unchanged.

**`L_step_productive`**: Penalize only *unproductive* steps.
```
L_step_productive = Σ_t (1 - ρ_t) · 1
```
where `ρ_t = π_t.progress`. If the model is making representational progress (high `ρ_t`), steps are cheap. If it is wandering (low `ρ_t`), steps are expensive. This creates pressure to **either leap forward or halt**—never meander.

**`L_synthesis`**: Reward cross-domain integration.
```
L_synthesis = -cosine_similarity( h_T, SynthesisTarget ) · I[leap_used]
```
If the model uses a leap candidate (`TailSample`) and the final answer is correct, it receives bonus gradient flow. This explicitly rewards **making distant associations productive**.

**`L_compression`**: Reward finding simpler encodings.
```
L_compression = - Σ_t || W_compress · [m_t; d_t] ||_info  # Information bottleneck
```
We use the information bottleneck principle: reward representations that preserve predictive information about the answer while minimizing complexity. This is the mathematical formalization of "insight as compression."

**`L_epiphany_reg`**: Prevent degenerate epiphanies (false phase transitions).
```
L_epiphany_reg = KL( π_t.epiphany || Bernoulli(0.3) ) + λ · (1 - ρ_t) · π_t.epiphany
```
Penalize claiming an epiphany (high `π_t.epiphany`) when progress is low. Force the metacognitive network to be honest about representational quality.

**`L_resonance`**: Reward trajectories that benefit from cross-talk.
```
L_resonance = - Σ_k || h_T^(k) - h_T^consensus || · accuracy_k
```
If a trajectory converges to the consensus and the consensus is accurate, reward the alignment. If a trajectory diverges but is *more* accurate than consensus, reward the divergence (anti-consensus bonus). This preserves creative disagreement.

### 3.3 Curriculum Pressure Scheduling

The pressure parameters are not fixed; they follow a curriculum:

| Phase | Epochs | `dim(c_t)` | `λ₁` | `β_t` (noise) | Goal |
|---|---|---|---|---|---|
| **Exploration** | 0-20% | High | Low | High | Learn the retrieval space; build associations |
| **Compression** | 20-60% | Decreasing | Increasing | Moderate | Force gist extraction; discover compact manifolds |
| **Refinement** | 60-85% | Low | High | Low | Optimize halt precision; polish epiphany detection |
| **Generalization** | 85-100% | Variable | Variable | Variable | Randomized pressure to prevent overfitting to specific regimes |

The curriculum ensures the model first learns *what* to know, then learns *how* to compress it, then learns *when* it has compressed enough.

---

## 4. Comparison with v1.2 and Existing Architectures

| Dimension | v1.2 | **v1.3** | RAG | CoT (o1/R1) |
|---|---|---|---|---|
| **Constraint philosophy** | Accept scarcity | **Exploit scarcity as generative** | Ignore | Add more tokens |
| **Stopping criterion** | Step budget exhausted | **Epiphany detected** | Single-pass | Fixed chain length |
| **Retrieval** | Precision-only | **Precision + Leap** | Precision-only | N/A |
| **Memory dynamics** | Overwrite by necessity | **Orchestrated forgetting + importance gating** | Append-only | Append-only |
| **Cross-domain analogy** | Accidental | **Injected + rewarded** | Unlikely | Possible but costly |
| **Offline processing** | Slow substrate updates | **Sleep-state trajectory diffusion** | None | None |
| **Parallel exploration** | None | **Multi-trajectory resonance** | None | Beam search |
| **Metacognition** | None | **Epiphany detection** | None | None |
| **Primary risk** | Training instability | **Metacognitive hallucination** (false epiphanies) | Retrieval ceiling | Cost/length scaling |

**Positioning**: v1.3 is not merely a more efficient inference regime. It is an **alternative cognitive regime** that trades the interpolative smoothness of abundance for the discontinuous leaps of constraint.

---

## 5. Known Limitations and Open Problems

We remain explicit about what is unproven:

### 5.1 Metacognitive Hallucination
The Epiphany Detector can be wrong. It may detect a phase transition where none exists (false insight) or miss a genuine one (insight blindness). Unlike factual hallucination, metacognitive hallucination is **self-reinforcing**: a false epiphany halts exploration prematurely, preventing correction.

**Mitigation**: The `L_epiphany_reg` term and the requirement for consensus across trajectories. But this is not guaranteed.

### 5.2 Leap Poisoning
If the tail-sample retrieval injects harmful, biased, or adversarial content, the model may synthesize it into a seemingly elegant but dangerous conclusion. The "inspiration" mechanism becomes an "indoctrination" mechanism.

**Mitigation**: Safety filters on the retrieval index; persona-layer lock on harmful synthesis. But semantic leaps by definition bypass standard relevance-based safety checks.

### 5.3 Resonance Collapse
Cross-trajectory synchronization can cause all trajectories to collapse to the same local optimum, destroying the diversity that makes resonance valuable.

**Mitigation**: The anti-consensus bonus in `L_resonance`. But tuning it to prevent collapse without preventing convergence is delicate.

### 5.4 Compression vs. Expressiveness Trade-off
The `W_compress` bottleneck forces simplification, but some problems may require maintaining full complexity until the final step. Over-aggressive compression can lose information necessary for genuine insight.

**Mitigation**: Curriculum scheduling starts with high `dim(c_t)` and only compresses after associations are learned. But the optimal compression trajectory is task-dependent.

### 5.5 Evaluation Crisis
There is still no benchmark that rewards:
- Representational efficiency (bits of answer per bit of state)
- Cross-domain analogy quality
- Epiphany authenticity (did the model actually restructure its representation, or just get lucky?)
- Creative divergence from training distribution

We need new metrics, not just new models.

---

## 6. Implementation Roadmap

### Phase 1: Compression Dynamics (3-4 months)
- **Scope**: Single trajectory, Mamba-2-370M, frozen BERT retriever
- **Focus**: Validate that `L_compression` + curriculum scheduling leads to observable representational phase transitions in `h_t`
- **Success criteria**: 
  - Effective dimensionality of `h_t` drops measurably before correct answers on multi-hop tasks
  - `π_t` correlates with human-annotated "insight moments" in reasoning chains
  - Match v1.2 accuracy with ≤70% steps
- **Risk**: Phase transitions may not emerge at small scale

### Phase 2: Metacognition & Noise (4-6 months)
- **Scope**: Add Epiphany Detector and leap retrieval; scale to 1B-3B
- **Focus**: Train `MetacogNet` to be calibrated (epiphany probability matches actual representational restructurings)
- **Success criteria**:
  - False epiphany rate < 20% on held-out tasks
  - Leap retrieval improves performance on cross-domain analogy tasks by >15% over precision-only
  - `L_step_productive` shows lower loss than v1.2's `L_step` at same accuracy
- **Risk**: Leap retrieval may introduce too much noise; curriculum may fail to find sweet spot

### Phase 3: Resonance & Diffusion (6-9 months)
- **Scope**: K=3 trajectories + cross-sync; offline diffusion; integrate with 7B-14B Transformer
- **Focus**: Measure whether multi-trajectory resonance produces answers unreachable by any single trajectory
- **Success criteria**:
  - Consensus output outperforms best single trajectory on >30% of creative/analogical tasks
  - Offline diffusion improves session-to-session consistency without core-layer updates
  - Personality stability metrics maintained across 30-day slow-update cycles
- **Risk**: Resonance collapse; diffusion causes schema drift

### Phase 4: RL and Open-Ended Evaluation (9-12 months)
- **Scope**: Outcome-based RL replacing `L_answer`; custom benchmarks for insight
- **Reward**: `R = correctness + γ · compression_ratio - δ · false_epiphanies + ε · novelty_score`
- **Focus**: Validate the full "constraint → insight" pipeline on open-ended creative tasks

---

## 7. Conclusion

RRUM v1.3 is built on a wager:

> **That the path to machine insight does not lie in adding more layers, more context, and more tokens—but in adding the right pressures, the right bottlenecks, and the right noises.**

We do not claim to have built a creative machine. We claim to have built an architecture where creativity is **no longer impossible by design**. Where forgetting is productive. Where compression is rewarded. Where distant ideas are forced to collide. Where the system learns not just to think, but to recognize when its own thinking has restructured itself.

The constraints are severe. The training is fragile. The evaluation is undefined. But if the Generative Constraint Hypothesis is correct, these are not obstacles to overcome—they are the very forces that will shape whatever comes next.

We invite the community to test, break, and improve it.

---

## References

1. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
2. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
3. Gu & Dao (2024). "Mamba-2: State Space Duality."
4. Peng et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
5. Peng et al. (2024). "RWKV-7: Dynamic State Evolution."
6. Schmidhuber (1991). "Curious Model-Building Control Systems."
7. Tishby & Zaslavsky (2015). "Deep Learning and the Information Bottleneck Principle."
8. Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models."
9. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
10. Moonshot AI (2026). "Kimi K3 Technical Report."
11. Google DeepMind (2024). "Titan: Learning to Memorize at Test Time."

---

**License & Contact**

This document is released for open research discussion. The architecture specification is provided as-is, with explicit acknowledgment of the limitations described in Section 5. We welcome implementations, criticisms, and improvements.
