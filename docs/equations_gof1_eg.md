# gof1 Project Equations

*CIITR Benchmark — LLM_ONLY vs LLM_AI on GoF class diagram generation*

---

## 1. Physical Energy

$$E = t \times P$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| `t` | LLM inference duration (seconds) | measured per call |
| `P` | constant CPU power | 28 W |
| `E` | energy consumed (Joules) | variable |

> Energy is the physical constraint of the CIITR framework: all understanding has a cost.

---

## 2. Structural Integration — Φᵢ

$$\Phi_i = |\text{classes}| + 2 \times |\text{relationships}|$$

Structural richness metric of the generated diagram: each relationship counts double as it encodes a dependency between two elements.

> Φᵢ is a property of the **generated diagram**, identical for LLM_ONLY and LLM_AI.  
> It measures structural density, independently of the declared pattern.

---

## 3. Active Inference Agent Generative Model

### 3.1 Likelihood Matrix A

$$A \in \mathbb{R}^{K \times S}, \quad A[o, s] = P(o \mid s)$$

- `K` = number of observable keywords (33 seeds + dynamically learned tokens)
- `S` = 23 GoF patterns
- Columns normalised to 1: $\sum_{o} A[o, s] = 1 \; \forall s$

Normalisation from raw counts:

$$A[o, s] = \frac{A_{\text{raw}}[o, s]}{\sum_{o'} A_{\text{raw}}[o', s]}$$

### 3.2 Transition Matrix B

$$B[s', s] = \begin{cases} 0.85 & \text{if } s' = s \\ \dfrac{0.15}{S - 1} \approx 0.00682 & \text{otherwise} \end{cases}$$

Patterns are assumed stable (stay = 0.85): once detected, a pattern changes little from one observation to the next.

### 3.3 Initial Prior D

$$D[s] = \frac{1}{S} = \frac{1}{23} \quad \forall s$$

Uniform distribution: no pattern is favoured a priori before any observation.

---

## 4. Bayesian Update — Exact Inference

At each observed keyword `o`:

$$\text{(1) Likelihood:} \quad \ell_s = A[o, s] = P(o \mid s)$$

$$\text{(2) Unnormalised:} \quad \tilde{Q}(s) = \ell_s \cdot P(s)$$

$$\text{(3) Evidence (marginal):} \quad P(o) = \sum_s \tilde{Q}(s)$$

$$\text{(4) Posterior:} \quad Q(s) = \frac{\tilde{Q}(s)}{P(o)}$$

$$\text{(5) Predictive prior:} \quad P(s_{t+1}) = \sum_{s} B[s', s] \cdot Q(s_t)$$

> Under exact inference (`Q = P(s|o)`), the VFE reduces to Friston's surprisal: `VFE = −log P(o)`.

---

## 5. Variational Free Energy (VFE)

Full pymdp formulation:

$$\text{VFE} = \mathbb{KL}[Q(s) \| P(s)] - \mathbb{E}_Q[\log P(o \mid s)]$$

$$= \sum_s Q(s) \left[ \log Q(s) - \log P(s) - \log A[o, s] \right]$$

Simplified case (exact inference, Q = Bayes posterior):

$$\text{VFE} = -\log P(o) = -\log \sum_s A[o, s] \cdot P(s)$$

This is **Friston's surprisal**: the negative log-probability of the observation under the generative model.

---

## 6. Rhythmic Anchoring — Rᵍ (proxy)

$$R^g = \frac{1}{N} \sum_{t=1}^{N} \text{VFE}_t$$

`N` = total number of keywords observed over the session.

`Rᵍ` is the **mean surprisal** across all observations. It measures how surprised the agent was on average — a proxy for the amount of belief updating performed.

| Mode | Rᵍ | Interpretation |
|------|-----|---------------|
| LLM_ONLY | **0** | Frozen weights, no belief update — Cᵢ = 0 by definition |
| LLM_AI | **> 0** | Beliefs Q(s) updated at each observation |

> CIITR (Hansen, Theory Note #2–2026): a system without rhythmic anchoring (`Rᵍ = 0`) is **epistemically inert**, regardless of the structural quality of its output.

---

## 7. Comprehension — Cᵢ

$$C_i = \Phi_i \times R^g$$

The relationship is **multiplicative**: the absence of either factor reduces comprehension to zero.

| Case | Cᵢ |
|------|----|
| LLM_ONLY: Rᵍ = 0 | Cᵢ = Φᵢ × 0 = **0** |
| LLM_AI: Rᵍ > 0 | Cᵢ = Φᵢ × Rᵍ > **0** |

---

## 8. Comprehension Per Joule — CPJ

$$\text{CPJ} = \frac{C_i}{E} = \frac{\Phi_i \times R^g}{E}$$

Epistemic efficiency metric: comprehension produced per unit of energy consumed.

---

## 9. Best Diagram Selection Criterion

$$\text{score} = \frac{\Phi_i}{E}$$

This criterion is **independent of Rᵍ**: it selects the structurally most efficient diagram, applicable to both modes. CPJ remains the theoretical CIITR metric; `Φᵢ/E` is the operational selection criterion.

---

## 10. Continuous Learning — learn_keyword

When the agent discovers a new token `w` in the LLM JSON output:

$$A_{\text{new}}[w, s] \propto Q(s) \quad \text{normalised: } \sum_s A_{\text{new}}[w, s] = 1$$

The new row of `A` is initialised proportionally to the **current belief** Q(s). The agent thus integrates words from the generated diagram as new observations, in line with the CIITR principle of recursive belief updating.

---

## Equation Summary

| Equation | Expression | File |
|----------|-----------|------|
| Energy | `E = t × 28` | [benchmark_gof.py](../benchmark_gof.py) |
| Φᵢ | `\|classes\| + 2 × \|relationships\|` | [gof/metamodel.py](../gof/metamodel.py) |
| A normalisation | `A = A_raw / A_raw.sum(axis=0)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Transition B | `diag=0.85, off=0.15/22` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Q(s) update | `Q(s) = A[o,s]·P(s) / P(o)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Predictive prior | `P(s_{t+1}) = B @ Q(s_t)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| VFE (simplified) | `−log P(o)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Rᵍ | `mean(VFE_1..N)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Cᵢ | `Φᵢ × Rᵍ` | [benchmark_gof.py](../benchmark_gof.py) |
| CPJ | `Cᵢ / E` | [benchmark_gof.py](../benchmark_gof.py) |
| Selection | `Φᵢ / E` | [benchmark_gof.py](../benchmark_gof.py) |

---

## Central Benchmark Hypothesis

$$\text{CPJ}_{\text{AI}} \geq \text{CPJ}_{\text{LLM}} \quad \text{and} \quad R^g_{\text{AI}} > 0$$

Results GOF-006 (effective iterations, warmup excluded):

| Mode | Avg Φᵢ | Avg Rᵍ | Avg CPJ | Avg Energy |
|------|--------|--------|---------|------------|
| LLM_ONLY | 9.0 | 0.000 | 0.000 | 212 J |
| LLM_AI | 9.25 | 0.236 | 0.00817 | 267 J |

→ Hypothesis **confirmed**: CPJ_AI > CPJ_LLM and Rᵍ_AI > 0.
