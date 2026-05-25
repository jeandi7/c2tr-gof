# Principe de l'énergie libre

*Source : Wikipedia — Free energy principle (traduction française)*

---

Le **principe de l'énergie libre** est un principe mathématique de la physique de l'information. Son application aux données d'imagerie cérébrale par IRMf, en tant que cadre théorique, suggère que le cerveau réduit la surprise ou l'incertitude en formulant des prédictions à partir de modèles internes, et utilise les entrées sensorielles pour mettre à jour ces modèles afin d'améliorer la précision de ses prédictions. Ce principe constitue une intégration approximative de l'inférence bayésienne et de l'inférence active, où les actions sont guidées par les prédictions et le retour sensoriel les affine. Il a permis de formuler des inférences étendues sur le fonctionnement du cerveau, la perception et l'action. Son applicabilité aux systèmes vivants a été remise en question.

---

## Vue d'ensemble

En biophysique et en sciences cognitives, le principe de l'énergie libre est un principe mathématique décrivant les capacités représentationnelles des systèmes physiques : c'est-à-dire pourquoi les choses qui existent semblent suivre les propriétés des systèmes auxquels elles sont couplées. Il établit que la dynamique des systèmes physiques minimise une quantité appelée **surprisal** (qui est le logarithme négatif de la probabilité d'un résultat), ou de manière équivalente sa borne supérieure variationnelle, appelée **énergie libre**. Ce principe est utilisé notamment dans les approches bayésiennes du fonctionnement cérébral, mais aussi dans certaines approches de l'intelligence artificielle ; il est formellement lié aux méthodes bayésiennes variationnelles et a été initialement introduit par **Karl Friston** comme explication des boucles perception-action incarnées en neurosciences.

Le principe de l'énergie libre modélise le comportement de systèmes distincts mais couplés à un autre système (par exemple, un environnement englobant), où les degrés de liberté qui constituent l'interface entre les deux systèmes sont connus sous le nom de **couverture de Markov** (*Markov blanket*). Plus formellement, le principe de l'énergie libre stipule que si un système possède une « partition particulière » (c'est-à-dire en particules avec leurs couvertures de Markov), alors des sous-ensembles de ce système vont suivre la structure statistique d'autres sous-ensembles (connus comme états internes et externes du système).

Le principe de l'énergie libre repose sur l'idée bayésienne du cerveau comme « moteur d'inférence ». Sous ce principe, les systèmes empruntent des chemins de moindre surprise, ou de manière équivalente, minimisent la différence entre les prédictions fondées sur leur modèle du monde et leurs perceptions sensorielles. Cette différence est quantifiée par l'énergie libre variationnelle et est minimisée par une correction continue du modèle du monde du système, ou en rendant le monde plus conforme aux prédictions du système. En modifiant activement le monde pour le rapprocher de l'état attendu, les systèmes peuvent également minimiser leur énergie libre. Friston postule que c'est le principe de toute réaction biologique. Il estime également que son principe s'applique aux troubles mentaux ainsi qu'à l'intelligence artificielle. Les implémentations d'IA fondées sur le principe d'inférence active ont montré des avantages par rapport à d'autres méthodes.

Le principe de l'énergie libre est un principe mathématique de la physique de l'information : tout comme le principe d'entropie maximale ou le principe de moindre action, il est vrai sur des bases mathématiques. Tenter de falsifier le principe de l'énergie libre est une erreur de catégorie, analogue à essayer de falsifier le calcul infinitésimal par des observations empiriques. Dans une interview de 2018, Friston a précisé ce que cela implique que le principe de l'énergie libre ne soit pas soumis à la falsification :

> *Je pense qu'il est utile d'établir ici une distinction fondamentale. La distinction est entre une théorie d'état et une théorie de processus ; c'est-à-dire la différence entre un principe normatif auquel les choses peuvent ou non se conformer, et une théorie de processus ou hypothèse sur la façon dont ce principe est réalisé. Sous cette distinction, le principe de l'énergie libre se distingue nettement de choses comme le codage prédictif et l'hypothèse du cerveau bayésien. En fait, il n'y a pas grand-chose qu'on puisse en faire, sauf se demander si les systèmes mesurables se conforment au principe. En revanche, les hypothèses selon lesquelles le cerveau effectue une forme d'inférence bayésienne ou de codage prédictif sont ce qu'elles sont — des hypothèses. Ces hypothèses peuvent ou non être soutenues par des preuves empiriques.*

---

## Contexte

L'idée que les systèmes biologiques auto-organisés — comme une cellule ou un cerveau — peuvent être compris comme minimisant l'énergie libre variationnelle est fondée sur les travaux de Helmholtz sur l'inférence inconsciente et les traitements ultérieurs en psychologie et en apprentissage automatique. L'énergie libre variationnelle est une fonction des observations et d'une densité de probabilité sur leurs causes cachées. Cette densité variationnelle est définie en relation avec un modèle probabiliste qui génère des observations prédites à partir de causes hypothétiques. Dans ce cadre, l'énergie libre fournit une approximation de l'évidence du modèle bayésien. Sa minimisation peut donc être vue comme un processus d'inférence bayésienne. Lorsqu'un système fait activement des observations pour minimiser l'énergie libre, il effectue implicitement une inférence active et maximise l'évidence de son modèle du monde.

Cependant, l'énergie libre est aussi une borne supérieure sur l'auto-information des résultats, où la moyenne à long terme de la surprise est l'entropie. Cela signifie que si un système agit pour minimiser l'énergie libre, il placera implicitement une borne supérieure sur l'entropie des résultats — ou états sensoriels — qu'il échantillonne.

---

## Relation avec d'autres théories

L'inférence active est étroitement liée au théorème du bon régulateur et aux approches connexes de l'auto-organisation, telles que l'auto-assemblage, la formation de motifs, l'autopoïèse et la practopoïèse. Elle aborde les thèmes considérés en cybernétique, en synergétique et en cognition incarnée. Parce que l'énergie libre peut être exprimée comme l'énergie attendue des observations sous la densité variationnelle moins son entropie, elle est également liée au principe d'entropie maximale. Enfin, comme la moyenne temporelle de l'énergie est l'action, le principe de minimisation de l'énergie libre variationnelle est un principe de moindre action.

L'énergie libre négative est formellement équivalente à la **borne inférieure de l'évidence** (*ELBO — Evidence Lower BOund*), couramment utilisée en apprentissage automatique pour entraîner des modèles génératifs tels que les autoencodeurs variationnels.

---

## Action et perception

L'inférence active applique les techniques de l'inférence bayésienne approximative pour inférer les causes des données sensorielles à partir d'un modèle « génératif » de la façon dont ces données sont causées, puis utilise ces inférences pour guider l'action. La règle de Bayes caractérise l'inversion probabilistiquement optimale d'un tel modèle causal, mais son application est généralement intraitable computationnellement, ce qui conduit à l'utilisation de méthodes approximatives. Dans l'inférence active, la classe principale de telles méthodes approximatives sont les **méthodes variationnelles**.

Ces méthodes variationnelles procèdent en minimisant une borne supérieure sur la divergence entre l'inférence optimale au sens de Bayes (ou « posterior ») et son approximation. Cette borne supérieure est connue sous le nom d'énergie libre, et on peut caractériser la perception comme la minimisation de l'énergie libre par rapport aux informations sensorielles entrantes, et l'action comme la minimisation de cette même énergie libre par rapport aux informations d'action sortantes.

### Modèle génératif

Le système est modélisé comme habitant un espace d'états `X`, factorisé selon :

```
X = Ψ × S × A × R
```

où :
- **Ψ** : espace des états « externes » cachés à l'agent (non directement perçus)
- **S** : espace des états sensoriels directement perçus par l'agent
- **A** : espace des actions possibles de l'agent
- **R** : espace des états internes privés de l'agent

Le modèle génératif est la spécification des densités de probabilité suivantes :

- **Modèle sensoriel** `p_S(s | ψ, a)` : vraisemblance des données sensorielles étant donné les états externes et les actions
- **Modèle de dynamique environnementale** `p_Ψ(ψ̇ | ψ, a)` : évolution attendue des états externes dans le temps
- **Modèle d'action** `p_A(a | μ, s)` : dépendance des actions sur les états internes et les données sensorielles
- **Modèle interne** `p_R(μ | s)` : dépendance des états internes sur les données sensorielles

Ces densités déterminent le **modèle joint** :

```
p(ψ̇, s, a, μ | ψ) = p_S(s|ψ,a) · p_Ψ(ψ̇|ψ,a) · p_A(a|μ,s) · p_R(μ|s)
```

### Définition de l'énergie libre

Puisque le calcul du posterior exact `p_Bayes` est intraitable, le principe de l'énergie libre postule l'existence d'une **densité variationnelle** `q(ψ̇ | s, a, μ, ψ)` approximant `p_Bayes`. L'énergie libre est alors définie comme :

```
F(μ, a ; s) = E_q[-log p(ψ̇, s, a, μ | ψ)] − H[q(ψ̇ | s, a, μ, ψ)]
             ─────────────────────────────    ──────────────────────────
                    énergie attendue                   entropie

           = −log p(s)  +  KL[q(ψ̇|s,a,μ,ψ) ‖ p_Bayes(ψ̇|s,a,μ,ψ)]
             ──────────     ──────────────────────────────────────────
               surprise                  divergence

           ≥ −log p(s)
              surprise
```

L'action et la perception sont définies comme le problème d'optimisation conjointe :

```
μ* = argmin_μ { F(μ, a ; s) }
a* = argmin_a { F(μ*, a ; s) }
```

---

## Minimisation de l'énergie libre

### Minimisation et auto-organisation

La minimisation de l'énergie libre a été proposée comme caractéristique des systèmes auto-organisés modélisés comme des systèmes dynamiques aléatoires. Cette formulation repose sur une couverture de Markov (comprenant les états d'action et sensoriels) qui sépare les états internes et externes. Si les états internes et l'action minimisent l'énergie libre, ils placent une borne supérieure sur l'entropie des états sensoriels :

```
lim(T→∞) (1/T) ∫₀ᵀ F(s(t), μ(t)) dt  ≥  lim(T→∞) (1/T) ∫₀ᵀ −log p(s(t)|m) dt  =  H[p(s|m)]
         ──────────────────────────                      ──────────────────────
               free-action                                      surprise
```

Ceci est dû au fait que — sous des hypothèses ergodiques — la moyenne à long terme de la surprise est l'entropie. Cette borne résiste à la tendance naturelle au désordre, du type associé au deuxième loi de la thermodynamique.

### Minimisation et inférence bayésienne

Toute inférence bayésienne peut être formulée en termes de minimisation de l'énergie libre. Lorsque l'énergie libre est minimisée par rapport aux états internes, la divergence de Kullback-Leibler entre la densité variationnelle et la densité postérieure sur les états cachés est minimisée. L'énergie libre peut être décomposée en complexité et précision :

```
F(s, μ) = D_KL[q(ψ|μ) ‖ p(ψ|m)]  −  E_q[log p(s|ψ,m)]
           ──────────────────────     ────────────────────
                 complexité                  précision
```

Les modèles avec une énergie libre minimale fournissent une explication précise des données, au prix de coûts de complexité (cf. rasoir d'Occam).

### Minimisation et thermodynamique

L'énergie libre variationnelle est un fonctionnel de la théorie de l'information et est distincte de l'énergie libre thermodynamique (Helmholtz). Cependant, le terme de complexité de l'énergie libre variationnelle partage le même point fixe que l'énergie libre de Helmholtz.

### Minimisation et théorie de l'information

La minimisation de l'énergie libre est équivalente à la maximisation de l'information mutuelle entre les états sensoriels et les états internes qui paramètrent la densité variationnelle. Cela relie la minimisation de l'énergie libre au **principe de redondance minimale**.

---

## Minimisation de l'énergie libre en neurosciences

La minimisation de l'énergie libre fournit un cadre pour formuler des modèles normatifs (Bayes-optimaux) de l'inférence et de l'apprentissage neuronaux sous incertitude, s'inscrivant ainsi dans l'**hypothèse du cerveau bayésien**. Les états cachés comprennent : `Ψ = X × Θ × Π` (variables dépendant du temps, paramètres invariants dans le temps, et précision des fluctuations aléatoires). La minimisation de ces variables correspond respectivement à l'inférence, l'apprentissage, et l'encodage de l'incertitude.

### Inférence et catégorisation perceptuelles

La minimisation de l'énergie libre formalise la notion d'inférence inconsciente en perception et fournit une théorie normative du traitement neuronal. La théorie de processus associée est basée sur la minimisation de l'énergie libre par descente de gradient, correspondant au filtrage bayésien généralisé :

```
μ̃̇ = D μ̃ − ∂_μ F(s, μ)|_{μ=μ̃}
```

Les cas particuliers du filtrage généralisé incluent le **filtrage de Kalman**, formellement équivalent au **codage prédictif** — une métaphore populaire pour le passage de messages dans le cerveau. Sous des modèles hiérarchiques, le codage prédictif implique l'échange récurrent d'erreurs de prédiction ascendantes (bottom-up) et de prédictions descendantes (top-down).

### Apprentissage perceptuel et mémoire

Dans le codage prédictif, l'optimisation des paramètres du modèle via une descente de gradient sur l'intégrale temporelle de l'énergie libre (action libre) se réduit à la **plasticité associative ou hébbienne**, associée à la plasticité synaptique dans le cerveau.

### Précision perceptuelle, attention et saillance

L'optimisation des paramètres de précision correspond à l'optimisation du gain des erreurs de prédiction (cf. gain de Kalman). Dans les implémentations neuronalement plausibles du codage prédictif, cela correspond à l'optimisation de l'excitabilité des cellules pyramidales superficielles et a été interprété en termes de **gain attentionnel**.

---

## Inférence active

Lorsque la descente de gradient est appliquée à l'action `ȧ = −∂_a F(s, μ̃)`, le contrôle moteur peut être compris en termes d'arcs réflexes classiques engagés par des prédictions descendantes (corticospinales). Cela fournit un formalisme qui généralise la solution du point d'équilibre — au problème des degrés de liberté — aux trajectoires de mouvement.

### Inférence active et contrôle optimal

L'inférence active est reliée au contrôle optimal en remplaçant les fonctions de valeur ou de coût par des **croyances a priori** sur les transitions d'état ou le flux. Cela exploite la connexion étroite entre le filtrage bayésien et la solution à l'équation de Bellman.

L'inférence active commence avec des priors sur le flux :

```
f = Γ · ∇V + ∇×W
```

où `V(x)` et `W(x)` sont des fonctions de valeur scalaire et vectorielle de l'espace d'état (décomposition de Helmholtz), et le coût est `c(x) = f·∇V + ∇·Γ·V`. En contraste, le contrôle optimal optimise le flux étant donné une fonction de coût sous l'hypothèse `W = 0`.

### Inférence active et théorie de la décision optimale

Les problèmes de décision optimale (généralement formulés comme des **processus de décision markoviens partiellement observables — POMDP**) sont traités dans l'inférence active en absorbant les fonctions d'utilité dans les croyances a priori. Les états à haute utilité (faible coût) sont des états qu'un agent s'attend à occuper.

Sur le plan neurobiologique, les neuromodulateurs comme la **dopamine** sont considérés comme rapportant la précision des erreurs de prédiction en modulant le gain des cellules principales encodant l'erreur de prédiction.

### Inférence active et neurosciences cognitives

L'inférence active a été utilisée pour aborder un large éventail de questions en neurosciences cognitives, notamment : l'observation d'actions, les neurones miroirs, les saccades et la recherche visuelle, les mouvements oculaires, le sommeil, les illusions, l'attention, la sélection d'actions, la conscience, l'hystérie et la psychose.

---

## Lien avec le projet gof1 (CIITR)

Dans ce projet, le `rg_proxy` implémente une version simplifiée du surprisal de Friston :

```
surprise = −log P(o)  =  −log(Σ_s P(o|s) · P(s))
```

La moyenne de cette surprise sur toutes les observations constitue le proxy de `Rᵍ`. Un `Rᵍ = 0` (LLM seul, poids figés) correspond exactement à l'absence de minimisation de l'énergie libre décrite par Friston : le système « performe sans comprendre ». Un `Rᵍ > 0` (agent LLM_AI) traduit une mise à jour active des croyances Q(s) à chaque observation, conforme au principe d'inférence active.
