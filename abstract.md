# gof1 — GoF Pattern Detector with Active Inference

## Contexte théorique

Ce projet s'inscrit dans le cadre **CIITR (Tor-Ståle Hansen, Note théorique n°2–2026)**.

La compréhension est définie comme :

**Cᵢ = Φᵢ × Rᵍ**

- **Φᵢ** : intégration structurelle (cohérence du diagramme généré)
- **Rᵍ** : ancrage rythmique — mise à jour récursive des croyances
- **CPJ** : Compréhension par Joule = Cᵢ / énergie — métrique d'efficacité épistémique

Lien avec Friston :
- **LLM seul → Rᵍ = 0** : poids figés, aucune mise à jour pendant l'inférence
- **Agent Active Inference → Rᵍ > 0** : minimisation de la free energy (VFE), croyances mises à jour à chaque observation

La **surprise** au sens de Friston : `−log P(o)` = log-probabilité négative de l'observation sous le modèle génératif. Le `rg_proxy` du projet est la moyenne de cette surprise sur toutes les observations.

## Objectif du projet

Benchmark **LLM_ONLY vs LLM_AI** sur la génération de diagrammes de classes GoF :

```
LLM_ONLY : objective → prompt → LLM → JSON → validate → PlantUML → CPJ
LLM_AI   : objective → AI détecte pattern → prompt enrichi → LLM → JSON → validate → PlantUML → CPJ
```

Hypothèse à démontrer : **CPJ_AI ≥ CPJ_LLM** et **Rᵍ > 0**.

## Modèle génératif Active Inference

Implémenté dans `agent_ai/gof_ai_agent.py` :

| Matrice | Signification |
|---------|--------------|
| `_A` [32×23] | P(keyword \| pattern) — vraisemblance, **colonnes normalisées à 1** |
| `_B` [23×23] | P(pattern suivant \| pattern courant) — transitions lentes (stay=0.85) |
| `_posterior` | Q(s) — croyance courante sur le pattern actif |
| `_prior` | P(s) — prior prédictif pour la prochaine observation |

Mise à jour bayésienne à chaque `observe_keyword(kw)` :
```python
likelihood = _A[kw_idx, :]
unnorm     = likelihood * prior        # P(o|s) · P(s)
evidence   = unnorm.sum()              # P(o) — marginale
surprise   = -log(evidence)           # VFE simplifié
posterior  = unnorm / evidence         # Bayes rule
prior      = _B @ posterior            # prior prédictif t+1
```

## Structure du projet

```
gof1/
  agent_ai/gof_ai_agent.py    — agent Active Inference (matrice A 32×23)
  agent_llm/llm.py            — appels LLM local (llama3.1:8b) / remote (devstral-small-2:24b)
  gof/generator.py            — construction du prompt LLM (avec/sans AI context)
  gof/metamodel.py            — validation JSON + traduction en PlantUML
  metrics/export.py           — export CSV des résultats
  benchmark_gof.py            — pipeline principal du benchmark
  sessions_gof.yaml           — sessions de test (GOF-001 à GOF-005)
```

## JSON produit par le LLM

```json
{
  "pattern": "Singleton",
  "classes": [
    {
      "id": "uniqueId",
      "name": "ClassName",
      "type": "class | abstract | interface",
      "attributes": ["- name: Type"],
      "methods": ["+ method(): ReturnType"]
    }
  ],
  "relationships": [
    {
      "type": "inheritance | implementation | association | dependency | composition | aggregation",
      "sourceId": "...",
      "targetId": "...",
      "label": "optional"
    }
  ]
}
```

## PlantUML généré

Seuls trois éléments sont utilisés : classes (avec attributs et méthodes), associations, héritage/implémentation.

Flèches PlantUML : `--|>` héritage, `..|>` implémentation, `-->` association, `..>` dépendance, `*-->` composition, `o-->` agrégation.

## LLM

- Local : `llama3.1:8b` via `ollama` CLI
- Remote : `devstral-small-2:24b` via `https://ollama.iboo.ovh`

## Lancer le benchmark

```bash
python benchmark_gof.py --session GOF-001 --remote
python benchmark_gof.py --no-ai
python benchmark_gof.py
```
