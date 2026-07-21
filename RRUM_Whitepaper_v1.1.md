# RRUM: Recurrent-Retrieval Unified Model
## A Resource-Constrained Architecture for Stateful Cognition

**Version 1.1** | July 2026

---

## Abstract

Transformer-based large language models achieve remarkable performance under the assumption of abundant parallel compute and static knowledge. However, real-world deployment faces hard constraints: limited inference budgets, continuously updating information, and the need for stable behavior over long-running sessions. We propose the **Recurrent-Retrieval Unified Model (RRUM)**, an architecture that explicitly accepts these constraints rather than fighting them. RRUM separates three functions into distinct components: a **recurrent working memory** (temporary, capacity-limited state), a **frozen associative substrate** (stable pattern completion), and an **external retriever** (on-demand factual lookup). By imposing a step-cost penalty during training, the recurrent core learns to explore efficiently, converge quickly, and halt appropriately. We present the architectural specification, training methodology, known limitations, and an honest assessment of what remains to be proven.

---

## 1. Motivation: The Cost of Abundance

### 1.1 Transformer Assumptions vs. Deployment Reality

The Transformer architecture was designed for batch-parallel machine translation under data-center conditions. Its core assumptions:
- **All tokens are simultaneously accessible** (full self-attention)
- **Context length is the only memory bound** (not state structure)
- **Knowledge is static** (frozen at training time)
- **Compute cost is acceptable** (O(n²) per forward pass)

In production agent systems, these assumptions break:
- **Long contexts degrade** ("lost in the middle" effects persist even at 128K)
- **Knowledge rots** (cutoff dates create perpetual staleness)
- **Inference costs scale linearly with thought length** (CoT tokens are expensive tokens)
- **Behavior drifts** (fine-tuning for new facts alters style, safety, and reasoning biases)

Chain-of-Thought and reasoning models (o1, R1) mitigate this by generating more tokens, but they remain **linear text expansions of a stateless forward pass**. There is no internal resource budget that forces trade-offs between exploration depth and answer quality.

### 1.2 The Resource-Constrained Alternative

RRUM is built on a different premise: **cognition under scarcity is not a degraded version of cognition under abundance—it is a different computational regime entirely.**

When a human solves a problem, they do not load all relevant facts into working memory simultaneously. They hold a small active set, query long-term memory for associations, look up external information when necessary, and feel time pressure to reach a conclusion. RRUM formalizes this regime:

| Component | Function | Biological Analogy | Resource Model |
|---|---|---|---|
| **Recurrent Core** | Temporary working memory; current reasoning trajectory | Prefrontal cortex workspace | **Limited capacity**, overwritten each step |
| **Transformer Substrate** | Stable associative pattern completion; style and prior biases | Hippocampal indexing / cortical schemas | **Frozen**, read-only, O(1) access per query |
| **External Retriever** | Factual lookup beyond training cutoff | External device (phone, book) | **On-demand**, network/storage cost only |

**Key Insight**: The recurrent core is not "reasoning" in some mystical sense. It is a **finite-state controller** operating under a budget, deciding what to retrieve next and when to stop retrieving.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Input Query x
    ↓
[Initialize h_0 from x + Transformer Bias]
    ↓
Loop (t = 1 to T_max):
    ├─ q_t = W_q · h_{t-1}                (Query projection)
    ├─ d_t = Retrieve(q_t, M_external)    (Dense vector search)
    ├─ h_t = RecurrentBlock(h_{t-1}, d_t; θ)  (State transition)
    ├─ p_halt = σ(W_h · h_t + b_h)        (Halt probability)
    └─ if halt sampled: break
    ↓
Output = TransformerDecode(h_T, M_frozen)
```

### 2.2 Component Specifications

#### 2.2.1 Working Memory: Recurrent Core

The recurrent core maintains a hidden state `h_t ∈ ℝ^d` with three operational roles:

1. **State Carrier**: `h_t` encodes the current reasoning trajectory—not facts, but *what is being done with facts right now*.
2. **Query Generator**: `q_t = W_q · h_t` projects into the retrieval embedding space.
3. **Halt Controller**: `p_halt = σ(W_h · h_t + b_h)` estimates whether sufficient exploration has occurred.

**Critical Property**: `h_t` is **temporary and bounded**. It is not a persistent memory store. Information from step `t-1` is carried forward only to the extent that the recurrent transition finds it relevant for step `t`. This is not "memorylessness" (the state does carry forward), but **volatility by design**—the state is expected to be overwritten when new information demands a shift in direction.

**Implementation**: We recommend **Mamba-2** or **RWKV-6** backbones due to linear-time recurrence and stable gradient flow over long trajectories. LSTM/GRU are discouraged for `d > 2048` due to gradient degradation.

#### 2.2.2 Associative Substrate: Frozen Transformer

The Transformer component is **frozen after pre-training**. It provides:
- Prior knowledge and factual biases
- Linguistic style and tone
- Default reasoning heuristics
- Safety guardrails

**Access Pattern**: The recurrent core does not attend over the Transformer directly. Instead, `h_t` is initialized with a **bias vector** derived from the Transformer’s output layer on the input query, and the final answer is decoded by feeding `h_T` into the Transformer as a conditioning vector. This ensures:
- The Transformer is never updated (no drift)
- Knowledge updates happen only via the external retriever
- Style and safety remain constant across sessions

#### 2.2.3 External Retriever

`Retrieve(q_t, M_external)` performs dense vector search over an updatable knowledge base.

- **Training**: Use differentiable soft attention over a cached knowledge index (e.g., contrastive retrieval with a frozen encoder) to allow gradient flow. Alternatively, use the REINFORCE estimator with a learned baseline if hard retrieval is required.
- **Inference**: Standard top-k approximate nearest neighbor (ANN) search (FAISS, ScaNN).
- **Update Mechanism**: The knowledge base `M_external` can be updated in real-time without touching model weights.

---

## 3. Mathematical Formalization

### 3.1 Forward Pass

Given input embedding `x` and frozen Transformer `M`:

```
h_0 = LayerNorm( W_init · [x; Bias(M(x))] )   # Initialize with input + prior bias

for t = 1 to T_max:
    q_{t-1} = W_q · h_{t-1}

    # Retrieval (differentiable during training, hard during inference)
    d_{t-1} = SoftRetrieve(q_{t-1}, M_external)   # training
    # d_{t-1} = HardRetrieve(q_{t-1}, M_external) # inference

    # Recurrent transition
    h_t = MambaBlock(h_{t-1}, d_{t-1}; θ)

    # Halt decision
    p_halt^{(t)} = σ(W_h · h_t + b_h)

    if sample(p_halt^{(t)}) == 1:
        T = t
        break

# Decode with frozen Transformer conditioned on final state
output = TransformerDecode(h_T, M)
```

### 3.2 Training Objective

End-to-end training uses a composite loss:

```
L = L_answer + λ₁ · L_step + λ₂ · L_relevance + λ₃ · L_halt_reg
```

**`L_answer`**: Cross-entropy on final output. Standard supervision.

**`L_step`**: Step-count penalty. This is the **exploration budget**:
```
L_step = T
```
*Rationale*: This creates pressure to reach correct answers in fewer steps. The recurrent core must learn to generate queries that retrieve high-value information early, rather than accumulating context passively.

*Implementation note*: `T` is discrete and non-differentiable. During training, we optimize the expected value `𝔼[T]` using:
- Straight-through estimators for the halt threshold
- REINFORCE with a learned baseline for the stopping policy
- Or Gumbel-Softmax relaxation of `p_halt`

**`L_relevance`**: Ensures retrieved documents are useful:
```
L_relevance = -cosine_similarity( Aggregate(d_1...d_T), target_context )
```
where `target_context` is derived from human-annotated reasoning chains or contrastive positive/negative passages.

**`L_halt_reg`**: Prevents degenerate strategies (always halt immediately, or never halt):
```
L_halt_reg = KL( Bernoulli(p_halt) || Bernoulli(0.5) )
```

### 3.3 Why Step Pressure Creates Efficient Exploration

We do not hand-design exploration strategies (epsilon-greedy, entropy bonuses). Instead, efficiency emerges from the cost structure:

- Poor early retrieval → more steps needed → higher `L_step` → gradient pushes for better query generation
- Local optima in retrieval space → steps accumulate → pressure to shift query semantics radically
- Correct but slow paths → still penalized → pressure to find shortcuts

This mirrors bounded rationality: agents with limited resources learn heuristics that approximate optimal search.

---

## 4. Honest Comparison with Existing Architectures

| Dimension | Pure Transformer | CoT Reasoning (o1/R1) | RAG | ReAct/Agent | **RRUM** |
|---|---|---|---|---|---|
| **State representation** | None (context window) | Text history only | None | Text history + tool logs | **Continuous hidden state** |
| **Knowledge update** | Fine-tuning | Fine-tuning | Vector DB update | Vector DB / API | **Vector DB update, model frozen** |
| **Behavior stability** | Fragile to updates | Fragile to updates | Prompt-dependent | Prompt-dependent | **Inherent (frozen substrate)** |
| **Reasoning structure** | Parallel pattern match | Linear token chain | Single-pass | Text-based loop | **Recurrent state transitions** |
| **Compute per reasoning step** | O(n²) | O(n²) | O(1) + O(n²) | O(n²) per action | **O(d) linear** |
| **Exploration control** | None | Temperature scaling | None | Hand-designed | **Learned halt + step budget** |
| **Known limitation** | Context degradation | Cost scales with length | No multi-hop reasoning | Unbounded loops | **Retrieval quality ceiling; training instability** |

**Positioning**: RRUM is not a replacement for Transformers. It is an **alternative inference regime** for tasks requiring:
- Multi-step information gathering
- Stable behavior over long sessions
- Real-time knowledge integration
- Predictable per-query costs (via halt control)

---

## 5. Known Limitations and Open Problems

We do not claim RRUM is proven. The following issues require resolution:

### 5.1 Training Instability
The halt decision and retrieval operation create **discontinuities** in the loss landscape. REINFORCE gradients have high variance. Straight-through estimators bias gradients. We do not yet have a training recipe that reliably converges beyond 1B parameter scales.

### 5.2 Retrieval Dependency
RRUM’s upper bound is the retriever’s upper bound. If the external index lacks relevant documents, the recurrent core may oscillate or halt prematurely. **There is no graceful degradation to internal knowledge**—the Transformer is frozen and not attended to during the loop.

### 5.3 State-Substrate Alignment
The recurrent core outputs `h_T`, which must be interpretable by the frozen Transformer decoder. If their representation spaces drift during training (e.g., the recurrent core learns idiosyncratic encodings), decoding fails. **Joint pre-training of the alignment layer** is likely necessary, but we have not validated this.

### 5.4 Expressiveness of Linear Recurrence
Mamba and RWKV achieve linear complexity through state compression. It is an open question whether this compressed state can support the **non-local reasoning jumps** that complex multi-hop questions require. If not, RRUM may be limited to simpler retrieval chains.

### 5.5 Evaluation Gap
There is no standard benchmark that measures:
- Reasoning efficiency (accuracy per retrieval step)
- Personality stability across knowledge updates
- Long-horizon session consistency

We need new metrics, not just new models.

---

## 6. Implementation Roadmap

### Phase 1: Feasibility Validation (3-4 months)
- **Scope**: Small-scale (Mamba-370M + frozen BERT-base retriever)
- **Tasks**: HotpotQA (multi-hop), StrategyQA (implicit reasoning), a custom "stable personality" test where the knowledge base is updated mid-session
- **Success criteria**: Match RAG accuracy with ≤50% retrieval steps; personality/style metrics stable across KB swaps
- **Risk**: May fail due to training instability (Section 5.1)

### Phase 2: Scale and Integration (6-9 months)
- **Scope**: Integrate with existing 7B-14B Transformer backbones (e.g., Qwen, Llama, or Kimi K3-base)
- **Method**: Freeze Transformer; train only recurrent core, query projection, and halt head
- **Focus**: State-substrate alignment (Section 5.3)
- **Success criteria**: Decode quality comparable to base model on held-out tasks; recurrent core converges in <8 steps for 80% of queries

### Phase 3: RL Optimization (9-15 months)
- **Scope**: Replace supervised `L_step` with outcome-based RL
- **Reward**: `R = correctness - α · steps - β · latency`
- **Exploration**: Train multiple RRUM instances with shared substrate but diverse exploration policies; allow cross-instance retrieval
- **Risk**: RL instability on non-differentiable retrieval; may require extensive hyperparameter search

---

## 7. Conclusion

RRUM proposes a simple shift in assumptions:

- **Stop treating inference as an abundant resource.** Impose budgets.
- **Stop treating knowledge and computation as the same layer.** Separate stable priors from temporary reasoning.
- **Stop pretending that more tokens equal deeper thought.** Give the model a state that can be overwritten, not just appended to.

We do not claim this architecture produces "inspiration" or "consciousness." We claim it is a **principled response to the constraints of real-world deployment**: limited budgets, updating knowledge, and the need for predictable behavior.

Whether these constraints matter enough to justify a new architecture is an empirical question. We invite the community to test it.

---

## References

1. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
2. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
3. Peng et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
4. Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models."
5. Shinn et al. (2023). "Reflexion: Self-Reflective Agents."
6. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
7. Moonshot AI (2026). "Kimi K3 Technical Report."

---

**License & Contact**

This document is released for open research discussion. The architecture specification is provided as-is, with explicit acknowledgment of the limitations described in Section 5. We welcome implementations, criticisms, and improvements.
