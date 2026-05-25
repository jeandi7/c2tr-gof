# Équations du projet gof1

*Benchmark CIITR — LLM_ONLY vs LLM_AI sur la génération de diagrammes GoF*

---

## 1. Énergie physique

$$E = t \times P$$

| Symbole | Signification | Valeur |
|---------|--------------|--------|
| `t` | durée d'inférence LLM (secondes) | mesurée à chaque appel |
| `P` | puissance CPU constante | 28 W |
| `E` | énergie consommée (Joules) | variable |

> L'énergie est la contrainte physique du cadre CIITR : toute compréhension a un coût.

---

## 2. Intégration structurelle — Φᵢ

$$\Phi_i = |\text{classes}| + 2 \times |\text{relationships}|$$

Métrique de richesse structurelle du diagramme généré : chaque relation vaut double car elle encode une dépendance entre deux éléments.

> Φᵢ est une propriété du **diagramme généré**, identique pour LLM_ONLY et LLM_AI.  
> Elle mesure la densité structurelle, indépendamment du pattern déclaré.

---

## 3. Modèle génératif de l'agent Active Inference

### 3.1 Matrice de vraisemblance A

$$A \in \mathbb{R}^{K \times S}, \quad A[o, s] = P(o \mid s)$$

- `K` = nombre de mots-clés observables (33 seeds + tokens appris dynamiquement)
- `S` = 23 patterns GoF
- Colonnes normalisées à 1 : $\sum_{o} A[o, s] = 1 \; \forall s$

Normalisation depuis les comptes bruts :

$$A[o, s] = \frac{A_{\text{raw}}[o, s]}{\sum_{o'} A_{\text{raw}}[o', s]}$$

### 3.2 Matrice de transition B

$$B[s', s] = \begin{cases} 0.85 & \text{si } s' = s \\ \dfrac{0.15}{S - 1} \approx 0.00682 & \text{sinon} \end{cases}$$

Les patterns sont supposés stables (stay = 0.85) : une fois détecté, un pattern change peu d'une observation à l'autre.

### 3.3 Prior initial D

$$D[s] = \frac{1}{S} = \frac{1}{23} \quad \forall s$$

Distribution uniforme : aucun pattern n'est favorisé a priori avant toute observation.

---

## 4. Mise à jour bayésienne — inférence exacte

À chaque mot-clé observé `o` :

$$\text{(1) Vraisemblance :} \quad \ell_s = A[o, s] = P(o \mid s)$$

$$\text{(2) Non-normalisé :} \quad \tilde{Q}(s) = \ell_s \cdot P(s)$$

$$\text{(3) Evidence (marginale) :} \quad P(o) = \sum_s \tilde{Q}(s)$$

$$\text{(4) Posterior :} \quad Q(s) = \frac{\tilde{Q}(s)}{P(o)}$$

$$\text{(5) Prior prédictif :} \quad P(s_{t+1}) = \sum_s B[s', s] \cdot Q(s_t)$$

> Sous inférence exacte (`Q = P(s|o)`), la VFE se réduit au surprisal de Friston : `VFE = −log P(o)`.

---

## 5. Énergie libre variationnelle (VFE)

Formulation complète pymdp :

$$\text{VFE} = \mathbb{KL}[Q(s) \| P(s)] - \mathbb{E}_Q[\log P(o \mid s)]$$

$$= \sum_s Q(s) \left[ \log Q(s) - \log P(s) - \log A[o, s] \right]$$

Cas simplifié (inférence exacte, Q = posterior de Bayes) :

$$\text{VFE} = -\log P(o) = -\log \sum_s A[o, s] \cdot P(s)$$

C'est le **surprisal de Friston** : l'information négative de l'observation sous le modèle génératif.

---

## 6. Ancrage rythmique — Rᵍ (proxy)

$$R^g = \frac{1}{N} \sum_{t=1}^{N} \text{VFE}_t$$

`N` = nombre total de mots-clés observés sur la session.

`Rᵍ` est la **moyenne des surprisals** sur toutes les observations. Il mesure à quel point l'agent a été surpris en moyenne — proxy de la quantité de mise à jour des croyances effectuée.

| Mode | Rᵍ | Interprétation |
|------|-----|---------------|
| LLM_ONLY | **0** | Poids figés, aucune mise à jour — Cᵢ = 0 par définition |
| LLM_AI | **> 0** | Croyances Q(s) mises à jour à chaque observation |

> CIITR (Hansen, Note n°2–2026) : un système sans ancrage rythmique (`Rᵍ = 0`) est **épistémiquement inerte**, quelle que soit la qualité structurelle de sa sortie.

---

## 7. Compréhension — Cᵢ

$$C_i = \Phi_i \times R^g$$

La relation est **multiplicative** : l'absence de l'un des deux facteurs annule la compréhension.

| Cas | Cᵢ |
|-----|----|
| LLM_ONLY : Rᵍ = 0 | Cᵢ = Φᵢ × 0 = **0** |
| LLM_AI : Rᵍ > 0 | Cᵢ = Φᵢ × Rᵍ > **0** |

---

## 8. Compréhension par Joule — CPJ

$$\text{CPJ} = \frac{C_i}{E} = \frac{\Phi_i \times R^g}{E}$$

Métrique d'efficacité épistémique : compréhension produite par unité d'énergie consommée.

---

## 9. Critère de sélection du meilleur diagramme

$$\text{score} = \frac{\Phi_i}{E}$$

Ce critère est **indépendant de Rᵍ** : il sélectionne le diagramme structurellement le plus efficace, applicable aux deux modes. CPJ reste la métrique théorique CIITR ; `Φᵢ/E` est le critère opérationnel de sélection.

---

## 10. Apprentissage continu — learn_keyword

Quand l'agent découvre un nouveau token `w` dans la sortie JSON du LLM :

$$A_{\text{new}}[w, s] \propto Q(s) \quad \text{normalisé : } \sum_s A_{\text{new}}[w, s] = 1$$

La nouvelle ligne de `A` est initialisée proportionnellement à la **croyance courante** Q(s). L'agent intègre ainsi les mots du diagramme généré comme nouvelles observations, conformément au principe CIITR de mise à jour récursive.

---

## Récapitulatif des équations

| Équation | Expression | Fichier |
|----------|-----------|---------|
| Énergie | `E = t × 28` | [benchmark_gof.py](../benchmark_gof.py) |
| Φᵢ | `\|classes\| + 2 × \|relationships\|` | [gof/metamodel.py](../gof/metamodel.py) |
| Normalisation A | `A = A_raw / A_raw.sum(axis=0)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Transition B | `diag=0.85, off=0.15/22` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Mise à jour Q(s) | `Q(s) = A[o,s]·P(s) / P(o)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Prior prédictif | `P(s_{t+1}) = B @ Q(s_t)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| VFE (simplifié) | `−log P(o)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Rᵍ | `mean(VFE_1..N)` | [agent_ai/gof_ai_agent.py](../agent_ai/gof_ai_agent.py) |
| Cᵢ | `Φᵢ × Rᵍ` | [benchmark_gof.py](../benchmark_gof.py) |
| CPJ | `Cᵢ / E` | [benchmark_gof.py](../benchmark_gof.py) |
| Sélection | `Φᵢ / E` | [benchmark_gof.py](../benchmark_gof.py) |

---

## Hypothèse centrale du benchmark

$$\text{CPJ}_{\text{AI}} \geq \text{CPJ}_{\text{LLM}} \quad \text{et} \quad R^g_{\text{AI}} > 0$$

Résultats GOF-006 (iters effectives, warmup exclu) :

| Mode | Φᵢ moy. | Rᵍ moy. | CPJ moy. | Énergie moy. |
|------|---------|---------|---------|-------------|
| LLM_ONLY | 9.0 | 0.000 | 0.000 | 212 J |
| LLM_AI | 9.25 | 0.236 | 0.00817 | 267 J |

→ Hypothèse **confirmée** : CPJ_AI > CPJ_LLM et Rᵍ_AI > 0.
