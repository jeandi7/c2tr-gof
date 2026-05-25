# Free Energy Principle

*Source: Wikipedia — Free energy principle*

---

The **free energy principle** is a mathematical principle of information physics. Its application to fMRI brain imaging data as a theoretical framework suggests that the brain reduces surprise or uncertainty by making predictions based on internal models and uses sensory input to update its models so as to improve the accuracy of its predictions. This principle approximates an integration of Bayesian inference with active inference, where actions are guided by predictions and sensory feedback refines them. From it, wide-ranging inferences have been made about brain function, perception, and action. Its applicability to living systems has been questioned.

---

## Overview

In biophysics and cognitive science, the free energy principle is a mathematical principle describing a formal account of the representational capacities of physical systems: that is, why things that exist look as if they track properties of the systems to which they are coupled. It establishes that the dynamics of physical systems minimise a quantity known as **surprisal** (which is the negative log probability of some outcome); or equivalently, its variational upper bound, called **free energy**. The principle is used especially in Bayesian approaches to brain function, but also some approaches to artificial intelligence; it is formally related to variational Bayesian methods and was originally introduced by **Karl Friston** as an explanation for embodied perception-action loops in neuroscience.

The free energy principle models the behaviour of systems that are distinct from, but coupled to, another system (e.g., an embedding environment), where the degrees of freedom that implement the interface between the two systems is known as a **Markov blanket**. More formally, the free energy principle says that if a system has a "particular partition" (i.e., into particles, with their Markov blankets), then subsets of that system will track the statistical structure of other subsets (which are known as internal and external states or paths of a system).

The free energy principle is based on the Bayesian idea of the brain as an "inference engine." Under the free energy principle, systems pursue paths of least surprise, or equivalently, minimize the difference between predictions based on their model of the world and their sense and associated perception. This difference is quantified by variational free energy and is minimized by continuous correction of the world model of the system, or by making the world more like the predictions of the system. By actively changing the world to make it closer to the expected state, systems can also minimize the free energy of the system. Friston assumes this to be the principle of all biological reaction. He also believes his principle applies to mental disorders as well as to artificial intelligence. AI implementations based on the active inference principle have shown advantages over other methods.

The free energy principle is a mathematical principle of information physics: much like the principle of maximum entropy or the principle of least action, it is true on mathematical grounds. To attempt to falsify the free energy principle is a category mistake, akin to trying to falsify calculus by making empirical observations. In a 2018 interview, Friston explained what it entails for the free energy principle to not be subject to falsification:

> *I think it is useful to make a fundamental distinction at this point — that we can appeal to later. The distinction is between a state and process theory; i.e., the difference between a normative principle that things may or may not conform to, and a process theory or hypothesis about how that principle is realized. Under this distinction, the free energy principle stands in stark distinction to things like predictive coding and the Bayesian brain hypothesis. This is because the free energy principle is what it is — a principle. Like Hamilton's principle of stationary action, it cannot be falsified. It cannot be disproven. In fact, there's not much you can do with it, unless you ask whether measurable systems conform to the principle. On the other hand, hypotheses that the brain performs some form of Bayesian inference or predictive coding are what they are — hypotheses. These hypotheses may or may not be supported by empirical evidence.*

---

## Background

The notion that self-organising biological systems — like a cell or brain — can be understood as minimising variational free energy is based upon Helmholtz's work on unconscious inference and subsequent treatments in psychology and machine learning. Variational free energy is a function of observations and a probability density over their hidden causes. This variational density is defined in relation to a probabilistic model that generates predicted observations from hypothesized causes. In this setting, free energy provides an approximation to Bayesian model evidence. Therefore, its minimisation can be seen as a Bayesian inference process. When a system actively makes observations to minimise free energy, it implicitly performs active inference and maximises the evidence for its model of the world.

However, free energy is also an upper bound on the self-information of outcomes, where the long-term average of surprise is entropy. This means that if a system acts to minimise free energy, it will implicitly place an upper bound on the entropy of the outcomes — or sensory states — it samples.

---

## Relationship to Other Theories

Active inference is closely related to the good regulator theorem and related accounts of self-organisation, such as self-assembly, pattern formation, autopoiesis, and practopoiesis. It addresses the themes considered in cybernetics, synergetics, and embodied cognition. Because free energy can be expressed as the expected energy of observations under the variational density minus its entropy, it is also related to the maximum entropy principle. Finally, because the time average of energy is action, the principle of minimum variational free energy is a principle of least action.

Negative free energy is formally equivalent to the **evidence lower bound (ELBO)**, which is commonly used in machine learning to train generative models, such as variational autoencoders.

---

## Action and Perception

Active inference applies the techniques of approximate Bayesian inference to infer the causes of sensory data from a 'generative' model of how that data is caused and then uses these inferences to guide action. Bayes' rule characterizes the probabilistically optimal inversion of such a causal model, but applying it is typically computationally intractable, leading to the use of approximate methods. In active inference, the leading class of such approximate methods are **variational methods**.

These variational methods proceed by minimizing an upper bound on the divergence between the Bayes-optimal inference (or 'posterior') and its approximation. This upper bound is known as the free energy, and we can accordingly characterize perception as the minimization of the free energy with respect to inbound sensory information, and action as the minimization of the same free energy with respect to outbound action information.

### Generative Model

The system is modelled as inhabiting a state space `X`, factorized according to:

```
X = Ψ × S × A × R
```

where:
- **Ψ** : space of 'external' states hidden from the agent (not directly perceived or accessible)
- **S** : space of sensory states directly perceived by the agent
- **A** : space of the agent's possible actions
- **R** : space of 'internal' states private to the agent

The generative model is the specification of the following density functions:

- **Sensory model** `p_S(s | ψ, a)` : likelihood of sensory data given external states and actions
- **Environmental dynamics model** `p_Ψ(ψ̇ | ψ, a)` : how external states are expected to evolve over time
- **Action model** `p_A(a | μ, s)` : how the agent's actions depend upon its internal states and sensory data
- **Internal model** `p_R(μ | s)` : how the agent's internal states depend upon its sensory data

These density functions determine the factors of the **joint model**:

```
p(ψ̇, s, a, μ | ψ) = p_S(s|ψ,a) · p_Ψ(ψ̇|ψ,a) · p_A(a|μ,s) · p_R(μ|s)
```

### Definition of Free Energy

Since computing the exact posterior `p_Bayes` is computationally intractable, the free energy principle asserts the existence of a **variational density** `q(ψ̇ | s, a, μ, ψ)` approximating `p_Bayes`. The free energy is then defined as:

```
F(μ, a ; s) = E_q[-log p(ψ̇, s, a, μ | ψ)] − H[q(ψ̇ | s, a, μ, ψ)]
              ──────────────────────────────   ─────────────────────────
                      expected energy                   entropy

            = −log p(s)  +  KL[q(ψ̇|s,a,μ,ψ) ‖ p_Bayes(ψ̇|s,a,μ,ψ)]
              ──────────     ──────────────────────────────────────────
                surprise                    divergence

            ≥ −log p(s)
               surprise
```

Action and perception are defined as the joint optimisation problem:

```
μ* = argmin_μ { F(μ, a ; s) }
a* = argmin_a { F(μ*, a ; s) }
```

---

## Free Energy Minimisation

### Minimisation and Self-Organisation

Free energy minimisation has been proposed as a hallmark of self-organising systems when cast as random dynamical systems. This formulation rests on a Markov blanket (comprising action and sensory states) that separates internal and external states. If internal states and action minimise free energy, then they place an upper bound on the entropy of sensory states:

```
lim(T→∞) (1/T) ∫₀ᵀ F(s(t), μ(t)) dt  ≥  lim(T→∞) (1/T) ∫₀ᵀ −log p(s(t)|m) dt  =  H[p(s|m)]
         ──────────────────────────                      ──────────────────────
               free-action                                      surprise
```

This is because — under ergodic assumptions — the long-term average of surprise is entropy. This bound resists a natural tendency to disorder, of the sort associated with the second law of thermodynamics and the fluctuation theorem.

### Minimisation and Bayesian Inference

All Bayesian inference can be cast in terms of free energy minimisation. When free energy is minimised with respect to internal states, the Kullback–Leibler divergence between the variational and posterior density over hidden states is minimised. Free energy can be usefully decomposed into complexity and accuracy:

```
F(s, μ) = D_KL[q(ψ|μ) ‖ p(ψ|m)]  −  E_q[log p(s|ψ,m)]
           ──────────────────────     ────────────────────
                 complexity                  accuracy
```

Models with minimum free energy provide an accurate explanation of data, under complexity costs (cf. Occam's razor).

### Minimisation and Thermodynamics

Variational free energy is an information-theoretic functional and is distinct from thermodynamic (Helmholtz) free energy. However, the complexity term of variational free energy shares the same fixed point as Helmholtz free energy (under the assumption the system is thermodynamically closed but not isolated).

### Minimisation and Information Theory

Free energy minimisation is equivalent to maximising the mutual information between sensory states and internal states that parameterise the variational density. This relates free energy minimization to the **principle of minimum redundancy**.

---

## Free Energy Minimisation in Neuroscience

Free energy minimisation provides a useful way to formulate normative (Bayes optimal) models of neuronal inference and learning under uncertainty, subscribing to the **Bayesian brain hypothesis**. The hidden states comprise `Ψ = X × Θ × Π` (time-dependent variables, time-invariant parameters, and precision of random fluctuations). Minimising variables, parameters, and precision correspond to inference, learning, and the encoding of uncertainty, respectively.

### Perceptual Inference and Categorisation

Free energy minimisation formalises the notion of unconscious inference in perception and provides a normative (Bayesian) theory of neuronal processing. The associated process theory is based on minimising free energy through gradient descent, corresponding to generalised Bayesian filtering:

```
μ̃̇ = D μ̃ − ∂_μ F(s, μ)|_{μ=μ̃}
```

Special cases of generalised filtering include **Kalman filtering**, which is formally equivalent to **predictive coding** — a popular metaphor for message passing in the brain. Under hierarchical models, predictive coding involves the recurrent exchange of ascending (bottom-up) prediction errors and descending (top-down) predictions.

### Perceptual Learning and Memory

In predictive coding, optimising model parameters through a gradient descent on the time integral of free energy (free action) reduces to **associative or Hebbian plasticity** and is associated with synaptic plasticity in the brain.

### Perceptual Precision, Attention and Salience

Optimizing the precision parameters corresponds to optimizing the gain of prediction errors (cf. Kalman gain). In neuronally plausible implementations of predictive coding, this corresponds to optimizing the excitability of superficial pyramidal cells and has been interpreted in terms of **attentional gain**.

---

## Active Inference

When gradient descent is applied to action `ȧ = −∂_a F(s, μ̃)`, motor control can be understood in terms of classical reflex arcs that are engaged by descending (corticospinal) predictions. This provides a formalism that generalizes the equilibrium point solution — to the degrees of freedom problem — to movement trajectories.

### Active Inference and Optimal Control

Active inference is related to optimal control by replacing value or cost-to-go functions with **prior beliefs about state transitions or flow**. This exploits the close connection between Bayesian filtering and the solution to the Bellman equation.

Active inference starts with priors over flow:

```
f = Γ · ∇V + ∇×W
```

where `V(x)` and `W(x)` are scalar and vector value functions of state space (cf. Helmholtz decomposition), and cost is `c(x) = f·∇V + ∇·Γ·V`. In contrast, optimal control optimises the flow given a cost function under the assumption `W = 0`.

### Active Inference and Optimal Decision (Game) Theory

Optimal decision problems (usually formulated as **partially observable Markov decision processes — POMDPs**) are treated within active inference by absorbing utility functions into prior beliefs. States with high utility (low cost) are states an agent expects to occupy.

Neurobiologically, neuromodulators such as **dopamine** are considered to report the precision of prediction errors by modulating the gain of principal cells encoding prediction error.

### Active Inference and Cognitive Neuroscience

Active inference has been used to address a range of issues in cognitive neuroscience, brain function and neuropsychiatry, including: action observation, mirror neurons, saccades and visual search, eye movements, sleep, illusions, attention, action selection, consciousness, hysteria, and psychosis.

---

## Link with the gof1 Project (CIITR)

In this project, the `rg_proxy` implements a simplified version of Friston's surprisal:

```
surprise = −log P(o)  =  −log(Σ_s P(o|s) · P(s))
```

The average of this surprise over all observations constitutes the proxy for `Rᵍ`. A `Rᵍ = 0` (LLM alone, frozen weights) corresponds exactly to the absence of free energy minimisation described by Friston: the system "performs without comprehending." A `Rᵍ > 0` (LLM_AI agent) reflects active updating of beliefs Q(s) at each observation, in line with the active inference principle.
