# agent_ai/gof_ai_agent.py
#
# Active Inference agent for GoF design pattern detection from text keywords.
# Follows the pymdp generative model convention (Heins et al., 2022):
#   A[obs, state]           = P(o | s)    likelihood matrix, columns ∑=1
#   B[state, state, action] = P(s'| s, u) transition matrix
#   D[state]                = P(s_0)      uniform prior over initial states
#
# pymdp 0.0.7.1 requires numpy<2 and pymdp 1.0.0 requires JAX (DLL issue on
# Windows), so inference and VFE are computed in numpy using the same equations
# as pymdp internals.
#
# VFE = KL[Q(s) ∥ P(s)] − E_Q[log P(o|s)]
#      = Σ_s Q(s) · [log Q(s) − log P(s) − log A[o, s]]
#
# When Q = P(s|o) (exact Bayes, achieved here), VFE = −log P(o).
# States       : 23 GoF patterns
# Observations : 33 seed keywords + tokens learned dynamically from LLM JSON output
# rg_proxy     : mean VFE over all observed keywords
#
# CIITR principle — the agent learns continuously:
#   learn_keyword(token): discovers new token from LLM output,
#     initialises A[token, :] ∝ current posterior Q(s),
#     extends the instance-level _A matrix (never the module-level seed).

import numpy as np

PATTERN_NAMES = [
    "Singleton",              #  0
    "FactoryMethod",          #  1
    "AbstractFactory",        #  2
    "Builder",                #  3
    "Prototype",              #  4
    "Adapter",                #  5
    "Bridge",                 #  6
    "Composite",              #  7
    "Decorator",              #  8
    "Facade",                 #  9
    "Flyweight",              # 10
    "Proxy",                  # 11
    "ChainOfResponsibility",  # 12
    "Command",                # 13
    "Interpreter",            # 14
    "Iterator",               # 15
    "Mediator",               # 16
    "Memento",                # 17
    "Observer",               # 18
    "State",                  # 19
    "Strategy",               # 20
    "TemplateMethod",         # 21
    "Visitor",                # 22
]

TEXT_KEYWORDS = [
    "unique",        #  0
    "single",        #  1
    "global",        #  2
    "instance",      #  3
    "factory",       #  4
    "build",         #  5
    "step",          #  6
    "clone",         #  7
    "copy",          #  8
    "adapt",         #  9
    "convert",       # 10
    "incompatible",  # 11
    "decorate",      # 12
    "wrap",          # 13
    "proxy",         # 14
    "facade",        # 15
    "simplify",      # 16
    "tree",          # 17
    "leaf",          # 18
    "share",         # 19
    "notify",        # 20
    "subscribe",     # 21
    "execute",       # 22
    "undo",          # 23
    "visit",         # 24
    "iterate",       # 25
    "chain",         # 26
    "handler",       # 27
    "template",      # 28
    "skeleton",      # 29
    "snapshot",      # 30
    "grammar",       # 31
    "compatible",    # 32
    "trees",         # 33
    "format",        # 34
    "algorithm",     # 35
    "implement",     # 36
    "formats",       # 37
    "support",       # 38
]

# A[obs, state] = P(keyword | pattern) — raw counts, normalized below
#          Singleton  FactoryMethod  AbstractFactory  Builder  Prototype  Adapter  Bridge  Composite  Decorator  Facade  Flyweight  Proxy  ChainOfResp  Command  Interpreter  Iterator  Mediator  Memento  Observer  State  Strategy  TemplateMethod  Visitor
_A_RAW = np.array([
    [0.85,  0.05,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # unique
    [0.80,  0.05,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # single
    [0.70,  0.05,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # global
    [0.60,  0.10,  0.10,  0.05,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # instance
    [0.01,  0.75,  0.60,  0.10,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # factory
    [0.01,  0.10,  0.05,  0.80,  0.02,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # build
    [0.01,  0.05,  0.02,  0.75,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # step
    [0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01],  # clone
    [0.01,  0.01,  0.01,  0.01,  0.80,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.15,  0.01,  0.01,  0.01,  0.01,  0.01],  # copy
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # adapt
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.75,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.20,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # convert
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # incompatible
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # decorate
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.70,  0.01,  0.01,  0.30,  0.15,  0.01,  0.30,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # wrap
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # proxy
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # facade
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.80,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # simplify
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # tree
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # leaf
    [0.20,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.45,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # share → Flyweight(45) + Singleton(20)
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.15,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01],  # notify
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01],  # subscribe
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.75,  0.01,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # execute
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.40,  0.01,  0.01,  0.01,  0.01,  0.01],  # undo
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90],  # visit
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # iterate
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # chain
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.80,  0.20,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # handler
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01],  # template
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01],  # skeleton
    [0.01,  0.01,  0.01,  0.01,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01],  # snapshot
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.90,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # grammar
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # compatible
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],  # trees   → Composite
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.70,  0.20,  0.01],  # format  → Strategy, TemplateMethod
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.05,  0.01],  # algorithm → Strategy
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.30,  0.50,  0.01],  # implement  → TemplateMethod, Strategy
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.85,  0.05,  0.01],  # formats    → Strategy (NTFS/FAT32)
    [0.01,  0.01,  0.01,  0.01,  0.01,  0.10,  0.20,  0.01,  0.01,  0.10,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.45,  0.05,  0.01],  # support    → Strategy, Bridge, Adapter
], dtype=float)

_N_STATES = len(PATTERN_NAMES)

# Likelihood matrix A — columns normalized to 1 (pymdp convention)
_A = _A_RAW / _A_RAW.sum(axis=0, keepdims=True)

# Transition matrix B — structured co-occurrence affinities (pymdp shape: [s', s, u])
#
# B[s', s] = P(next_pattern=s' | current_pattern=s)
# Each column is stochastic (sums to 1).
# stay=0.70 on diagonal; known GoF companions get a bonus of 0.05;
# remaining probability is spread uniformly over non-companions.
#
# Companion lists encode structural co-occurrence knowledge:
#   "given belief in pattern s, which patterns are most likely to co-exist?"
_B_STAY = 0.70
_B_COMPANION_BONUS = 0.05

_COMPANIONS: dict[int, list[int]] = {
    0:  [1, 2, 7, 9, 20],     # Singleton       → FactoryMethod, AbstractFactory, Composite, Facade, Strategy
    1:  [0, 2, 4, 21],        # FactoryMethod   → Singleton, AbstractFactory, Prototype, TemplateMethod
    2:  [0, 1, 4, 6],         # AbstractFactory → Singleton, FactoryMethod, Prototype, Bridge
    3:  [7, 20, 21],          # Builder         → Composite, Strategy, TemplateMethod
    4:  [1, 2, 17],           # Prototype       → FactoryMethod, AbstractFactory, Memento
    5:  [6, 8, 9, 11],        # Adapter         → Bridge, Decorator, Facade, Proxy
    6:  [5, 2, 20, 21],       # Bridge          → Adapter, AbstractFactory, Strategy, TemplateMethod
    7:  [15, 22, 10, 20, 0],  # Composite       → Iterator, Visitor, Flyweight, Strategy, Singleton
    8:  [7, 11, 20, 5],       # Decorator       → Composite, Proxy, Strategy, Adapter
    9:  [0, 16, 5],           # Facade          → Singleton, Mediator, Adapter
    10: [7, 0, 1],            # Flyweight       → Composite, Singleton, FactoryMethod
    11: [8, 5, 19],           # Proxy           → Decorator, Adapter, State
    12: [13, 7, 15],          # ChainOfResp     → Command, Composite, Iterator
    13: [17, 7, 18, 12],      # Command         → Memento, Composite, Observer, ChainOfResp
    14: [7, 15, 22],          # Interpreter     → Composite, Iterator, Visitor
    15: [7, 1, 22],           # Iterator        → Composite, FactoryMethod, Visitor
    16: [18, 0, 9],           # Mediator        → Observer, Singleton, Facade
    17: [13, 18, 19],         # Memento         → Command, Observer, State
    18: [16, 0, 13, 19],      # Observer        → Mediator, Singleton, Command, State
    19: [20, 13, 11, 18],     # State           → Strategy, Command, Proxy, Observer
    20: [21, 19, 6, 7, 0],    # Strategy        → TemplateMethod, State, Bridge, Composite, Singleton
    21: [20, 1, 3, 6],        # TemplateMethod  → Strategy, FactoryMethod, Builder, Bridge
    22: [7, 15, 13],          # Visitor         → Composite, Iterator, Command
}


def _build_B() -> np.ndarray:
    n = _N_STATES
    B = np.zeros((n, n))
    for s in range(n):
        companions = set(_COMPANIONS.get(s, []))
        n_comp = len(companions)
        off_total = 1.0 - _B_STAY
        comp_total = n_comp * _B_COMPANION_BONUS
        base_off = (off_total - comp_total) / (n - 1 - n_comp) if (n - 1 - n_comp) > 0 else 0.0
        for sp in range(n):
            if sp == s:
                B[sp, s] = _B_STAY
            elif sp in companions:
                B[sp, s] = _B_COMPANION_BONUS
            else:
                B[sp, s] = base_off
    return B[:, :, np.newaxis]   # shape (23, 23, 1) — single null action


_B = _build_B()

# Uniform prior D over initial states (pymdp convention)
_D = np.ones(_N_STATES) / _N_STATES

_EPS = 1e-16

_STOP_TOKENS = {
    "type", "class", "name", "methods", "attributes", "relationships",
    "label", "sourceid", "targetid", "pattern", "void", "string",
    "list", "true", "false", "null", "none", "self", "return",
    "with", "from", "import", "the", "and", "for", "not", "are",
}


class GofAIAgent:
    """
    Perception-only Active Inference agent following the pymdp generative model.

    At each call to observe_keyword():
      1. Bayesian update: Q(s) ∝ P(o|s) · P(s)       — exact inference
      2. VFE: F = Σ_s Q(s)[log Q(s) − log P(s) − log A[o,s]]
      3. Predictive prior: P(s_{t+1}) = B[:,:,0] @ Q(s_t)
    """

    def __init__(self):
        self._keyword_index = {k: i for i, k in enumerate(TEXT_KEYWORDS)}
        self.reset()

    def reset(self):
        self._keyword_index = {k: i for i, k in enumerate(TEXT_KEYWORDS)}  # purge learned tokens
        self._A             = _A.copy()        # instance-level likelihood (grows via learn_keyword)
        self._posterior     = _D.copy()        # Q(s)  — current belief
        self._prior         = _D.copy()        # P(s)  — predictive prior
        self._peak_post     = _D.copy()        # max Q(s) seen at any step — for multi-pattern detection
        self._vfe_values: list[float] = []

    def observe_keyword(self, keyword: str) -> float:
        """Update beliefs from one keyword observation. Returns VFE."""
        idx = self._keyword_index.get(keyword.lower())
        if idx is None:
            return 0.0

        likelihood = self._A[idx, :]                      # P(o|s), shape (23,)
        unnorm     = likelihood * self._prior              # P(o|s) · P(s)
        evidence   = float(unnorm.sum())
        self._posterior = unnorm / (evidence + _EPS)      # Q(s) = Bayes posterior
        self._peak_post = np.maximum(self._peak_post, self._posterior)

        # VFE = KL[Q(s) ∥ P(s)] − E_Q[log P(o|s)]   (pymdp formula)
        vfe = float(np.sum(
            self._posterior * (
                np.log(self._posterior + _EPS)
                - np.log(self._prior    + _EPS)
                - np.log(likelihood     + _EPS)
            )
        ))
        self._vfe_values.append(vfe)

        # Predictive prior for next step: P(s_{t+1}) = B[:,:,0] @ Q(s_t)
        self._prior = _B[:, :, 0] @ self._posterior

        return vfe

    def learn_keyword(self, token: str) -> bool:
        """
        Discover a new token from LLM output and extend the instance-level A matrix.
        The new row is initialised ∝ current posterior Q(s), so the agent's current
        belief shapes future observations from this token.  Returns True if learned.
        """
        token = token.lower()
        if token in self._keyword_index or len(token) < 4 or token in _STOP_TOKENS:
            return False
        new_row = np.maximum(self._posterior.copy(), 1e-4)
        new_row /= new_row.sum()
        self._keyword_index[token] = len(self._keyword_index)
        self._A = np.vstack([self._A, new_row])
        return True

    @property
    def rg_proxy(self) -> float:
        return float(np.mean(self._vfe_values)) if self._vfe_values else 0.0

    def get_pattern_context(self) -> dict:
        # Use peak posterior for detection — captures patterns that dominated at any step
        dominant_idx = int(np.argmax(self._peak_post))
        top3_idx     = np.argsort(self._peak_post)[-3:][::-1]
        return {
            "pattern_name": PATTERN_NAMES[dominant_idx],
            "confidence":   float(self._peak_post[dominant_idx]),
            "top3":         [PATTERN_NAMES[i] for i in top3_idx],
            "top3_peaks":   [round(float(self._peak_post[i]), 3) for i in top3_idx],
            "rg":           self.rg_proxy,
            "posterior":    {PATTERN_NAMES[i]: round(float(self._posterior[i]), 3)
                             for i in range(_N_STATES)},
        }
