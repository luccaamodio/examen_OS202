# Rapport – Parallélisation d’une simulation de galaxie

## 1. Introduction

L’objectif de ce travail est d’étudier la parallélisation d’un code de simulation de galaxie à $N$ corps, basé sur une grille cartésienne, en utilisant à la fois numba (parallélisme par threads) et MPI (parallélisme par processus).

Nous avons implémenté et testé trois configurations principales :

1. Version séquentielle avec numba (sans MPI), puis numba avec `parallel=True` et plusieurs nombres de threads.
2. Séparation affichage / calcul avec MPI : un processus pour l’affichage, un pour le calcul.
3. Parallélisation du calcul avec MPI + numba : plusieurs processus MPI se partagent les corps, et numba parallélise les boucles internes par threads.

Toutes les mesures détaillées ont été faites sur le cas `data/galaxy_1000`, avec un pas de temps $\Delta t = 0{,}001$ et une grille $(N_i, N_j, N_k) = (20, 20, 1)$. Pour étudier l’effet de la taille du problème, nous avons également utilisé un jeu de données plus grand `data/galaxy_10000`, généré avec le script `galaxy_generator.py` (appel de type `python galaxy_generator.py 10000 data/galaxy_10000`).

---

## 2. Rappel du schéma séquentiel

Le système contient un trou noir central et $N$ étoiles. Pour chaque étoile $i$, on suit :

- la position p_i,
- la vitesse v_i,
- la masse m_i (constante).

Le schéma en temps utilisé est un schéma de Verlet :

- On calcule d’abord l’accélération a_i^(1)(t) subie par l’étoile.
- On met à jour la position : p_i ← p_i + Δt·v_i + 0,5·Δt²·a_i^(1).
- On recalcule l’accélération a_i^(2)(t + Δt).
- On met à jour la vitesse : v_i ← v_i + 0,5·Δt·(a_i^(1) + a_i^(2)).

L’accélération gravitationnelle suit la loi de Newton :

- a_i est la somme, pour tous les j ≠ i, de G·m_j·(p_j − p_i) / ||p_j − p_i||³.

Pour éviter la complexité O(N²), le code utilise une grille spatiale et, pour chaque cellule, la masse totale et le centre de masse.

---

## 3. Question préliminaire : choix de $N_k = 1$

Les galaxies générées dans les données sont essentiellement des disques minces dans le plan $Oxy$ : la distribution des étoiles est concentrée dans le plan vertical, avec une faible épaisseur selon $Oz$.

Si on choisit $N_k > 1$ (plusieurs cellules en $z$) :

- beaucoup de cellules seraient presque vides,
- le coût de gestion de la grille augmenterait (plus de cellules, plus de boucles),
- la précision physique ne serait pas significativement améliorée, car la variation en $z$ est faible.

Il est donc plus efficace d’utiliser $N_k = 1$, ce qui correspond à un disque fin.

---

## 4. Mesure du temps initial et partie à paralléliser

En observant la sortie du programme (version avec visualisation), on distingue deux temps par frame :

- `Render time` (affichage OpenGL / SDL),
- `Update time` (mise à jour des positions, donc calcul des accélérations).

Pour $N = 1000$ étoiles, on constate que :

- `Render time` reste relativement faible et peu sensible à $N$,
- `Update time` augmente fortement avec le nombre d’étoiles, et domine le coût total.

Conclusion : la partie la plus intéressante à paralléliser est le calcul des trajectoires (calcul des accélérations + mise à jour positions / vitesses), et non l’affichage.

---

## 5. Parallélisation avec numba (threads)

### 5.1. Modifications

Les principales fonctions numba ont été modifiées en :

- ajoutant `parallel=True` dans les décorateurs `@njit`,
- remplaçant certaines boucles `range` indépendantes par `prange`, notamment sur :
  - les cellules (calcul de masse totale et centre de masse),
  - les corps (calcul d’accélération).

Nous avons ensuite utilisé la variable d’environnement `NUMBA_NUM_THREADS` pour contrôler le nombre de threads.

### 5.2. Résultats mesurés (galaxy_1000)

Pour $N = 1000$, $\Delta t = 0{,}001$, grille $(20,20,1)$, $n_{\text{steps}} = 50$ :

| Threads numba | Temps moyen par update (s) | Temps moyen (ms) | Speedup vs 1 thread |
|---------------|----------------------------|------------------|---------------------|
| 1             | 0.04536                    | 45.4             | 1.00                |
| 2             | 0.04860                    | 48.6             | 0.93                |
| 4             | 0.06729                    | 67.3             | 0.67                |
| 8             | 0.10804                    | 108.0            | 0.42                |
| 12            | 0.14949                    | 149.5            | 0.30                |

Le speedup est défini par : S(p) = T(1) / T(p).

Ici, on constate que le parallélisme numba dégrade la performance pour ce cas : le meilleur temps est obtenu avec 1 seul thread.

Raison principale : pour $N = 1000$, la taille du problème est trop petite par rapport au coût d’ordonnancement des threads et au surcoût de la version `parallel=True`. L’overhead de parallélisation l’emporte sur les gains.

### 5.3. Résultats mesurés (galaxy_5000)

Pour $N = 5000$, avec le même schéma d’intégration (50 pas de temps, grille $(20,20,1)$), on obtient :

| Threads numba | Temps moyen par update (s) | Temps moyen (ms) | Speedup vs 1 thread |
|---------------|----------------------------|------------------|---------------------|
| 1             | 0.19689                    | 196.9            | 1.00                |
| 2             | 0.19678                    | 196.8            | 1.00                |
| 4             | 0.27382                    | 273.8            | 0.72                |
| 8             | 0.48217                    | 482.2            | 0.41                |
| 12            | 0.68638                    | 686.4            | 0.29                |

On remarque que 1 et 2 threads donnent pratiquement le même temps, et que toutes les configurations avec plus de threads sont plus lentes. Autrement dit, même pour un problème cinq fois plus gros, l’overhead de parallélisation reste dominant dès que l’on dépasse 2 threads.

### 5.4. Résultats mesurés (galaxy_10000)

Pour $N = 10\,000$, toujours avec 50 pas de temps et la même grille, les mesures sont :

| Threads numba | Temps moyen par update (s) | Temps moyen (ms) | Speedup vs 1 thread |
|---------------|----------------------------|------------------|---------------------|
| 1             | 0.49451                    | 494.5            | 1.00                |
| 2             | 0.43649                    | 436.5            | 1.13                |
| 4             | 0.56077                    | 560.8            | 0.88                |
| 8             | 0.95690                    | 956.9            | 0.52                |
| 12            | 1.37346                    | 1373.5           | 0.36                |

Ici, on voit enfin apparaître un petit speedup avec 2 threads ($S(2) \approx 1{,}13$), mais au‑delà de 2 threads les performances se dégradent à nouveau. Le meilleur compromis est donc obtenu avec un nombre très limité de threads (1–2) ; ajouter davantage de threads sur ce problème et cette machine n’apporte pas de gain, et peut même aggraver les temps d’exécution.

---

## 6. Séparation affichage / calcul avec MPI

Dans la version MPI `affichage + calcul` :

- le processus 0 s’occupe uniquement de l’affichage (appel au visualiseur 3D),
- le processus 1 calcule les trajectoires (mise à jour des positions),
- les deux processus communiquent les positions des étoiles à chaque pas de temps.

Effets observés qualitativement :

- l’affichage devient plus fluide, car il ne partage plus le CPU avec le calcul intensif,
- le calcul est isolé sur un processus qui peut utiliser numba.

Cependant, pour $N = 1000$, le temps total par frame n’est pas beaucoup plus faible que dans la version monolithique, car l’affichage n’était pas le véritable goulot d’étranglement. La séparation reste néanmoins une bonne architecture pour des cas plus grands (plus d’étoiles, plus de charge graphique).

Numériquement, on ne mesure donc aucune accélération significative pour $N = 1000$ en séparant affichage et calcul ; l’intérêt principal de cette variante est architectural et se manifesterait surtout pour des scènes beaucoup plus lourdes.

---

## 7. Parallélisation du calcul avec MPI (+ numba)

### 7.1. Schéma MPI utilisé

Dans la version MPI de parallélisation du calcul :

- chaque processus MPI possède une copie complète des positions / vitesses / masses,
- mais ne calcule l’accélération que pour un sous-ensemble de corps (découpage par indices),
- une opération `Allreduce` somme les contributions partielles pour obtenir un vecteur d’accélérations global.

À l’intérieur de chaque processus, numba avec `prange` parallélise encore les boucles sur les corps locaux : c’est donc un parallélisme hybride (MPI + threads).

### 7.2. Résultats MPI + numba (4 processus)

Pour $N = 1000$, même configuration que précédemment :

| Processus MPI | Threads numba | Temps moyen par pas (s) | Temps moyen (ms) |
|-------------- |--------------|--------------------------|------------------|
| 4             | 1            | 0.05459                  | 54.6             |
| 4             | 2            | 0.05243                  | 52.4             |
| 4             | 4            | 0.05070                  | 50.7             |

Comparaison avec la version numba 1 thread (0.04536 s ≈ 45.4 ms) :

- Tous les cas MPI (4 processus) sont plus lents que numba seul à 1 thread.
- Le parallélisme MPI n’apporte donc pas de speedup pour $N = 1000$.

### 7.3. Résultat MPI avec 12 processus

Nous avons aussi testé 12 processus MPI (1 thread numba par processus), pour exploiter les 12 cœurs logiques :

| Processus MPI | Threads numba | Temps moyen par pas (s) | Temps moyen (ms) |
|-------------- |--------------|--------------------------|------------------|
| 12            | 1            | 0.08360                  | 83.6             |

C’est encore plus lent que numba 1 thread (45.4 ms), et même plus lent que MPI 4 processus.

Raisons :

- pour $N = 1000$, chaque processus traite très peu de corps,
- le coût de communication (Allreduce) et de synchronisation reste relativement modéré,
- le ratio *calcul utile / overhead* est défavorable.

En instrumentant précisément le code MPI, on obtient par exemple pour $N = 1000$ :

- 4 processus, 1 thread numba : avg_step ≈ 0,053 s, temps moyen de communication ≈ 2·10⁻⁴ s, soit seulement 0,4 % du pas de temps ;
- 12 processus, 1 thread numba : avg_step ≈ 0,082 s, temps moyen de communication ≈ 6,7·10⁻³ s, soit environ 8 % du pas.

Pour $N = 5000$, le schéma est similaire :

- 4 processus, 1 thread : communication $\approx 1{,}7\%$ du temps de pas ;
- 4 processus, 4 threads : communication $\approx 5{,}8\%$ du temps de pas ;
- 12 processus, 1 thread : communication $\approx 16{,}7\%$ du temps de pas.

Ces chiffres montrent que la communication ne suffit pas à elle seule à expliquer les performances décevantes : même si l’on négligeait entièrement le temps d’Allreduce, la version MPI + numba resterait plus lente que numba seul à 1 thread, à cause de l’overhead de parallélisation (gestion de la grille sur chaque rang, scheduling des threads, synchronisations, etc.) sur un problème encore trop petit.

---

## 8. Discussion : pourquoi la parallélisation est plus lente ici ?

Le fait que la parallélisation soit plus lente que la version à 1 thread n’est pas un “bug” du code, mais plutôt une conséquence de :

1. **Taille du problème trop petite** ($N = 1000$) par rapport au coût d’initialisation des threads et des communications MPI.
2. **Coût fixe** de la compilation et du scheduling numba en mode `parallel=True`.
3. **Overhead de communication MPI** à chaque pas de temps (Allreduce sur tous les corps).
4. **Effets de cache et de mémoire** : plusieurs threads/processus peuvent se gêner (contenion, faux-partage).

On s’attendrait à voir un speedup intéressant pour des tailles de problèmes beaucoup plus grandes (par exemple $N = 10\,000$ ou $N = 50\,000$), où :

- le temps de calcul par pas devient beaucoup plus élevé,
- la part relative de l’overhead de parallélisation devient plus faible.

Nous avons justement réalisé ces mesures supplémentaires : pour $N = 5000$, le temps moyen par pas est pratiquement identique entre 1 et 2 threads, puis se dégrade nettement au‑delà ; pour $N = 10\,000$, on observe enfin un petit gain avec 2 threads ($S(2) \approx 1{,}13$), mais les configurations à 4, 8 ou 12 threads redeviennent plus lentes. Autrement dit, même si un début de speedup apparaît quand $N$ augmente, la taille reste encore trop modeste pour que le parallélisme massif soit réellement rentable sur cette machine.

Dans le cadre de cet examen, le but principal est de :

- montrer que la parallélisation a été correctement mise en œuvre,
- analyser pourquoi, dans ce cas précis, le speedup n’est pas au rendez-vous,
- et en tirer une conclusion honnête sur les limites du parallélisme pour ce problème et cette taille de données.

---

## 9. Déséquilibre de charge et distribution “intelligente”

On observe que la densité d’étoiles diminue avec la distance au trou noir. Une répartition simple des cellules (par exemple, blocs réguliers en $x$ ou en $(x,y)$) peut provoquer :

- des processus situés au centre avec beaucoup de travail (forte densité),
- des processus dans les régions externes avec peu de travail.

C’est un déséquilibre de charge (load imbalance) qui pénalise l’accélération globale.

Une distribution “intelligente” des cellules consisterait à :

- attribuer à chaque processus un ensemble de cellules dont la masse totale (ou le nombre de corps) est à peu près la même,
- ce qui améliore le balance de charge.

Mais cette stratégie peut introduire un autre problème :

- les cellules d’un même processus peuvent être géographiquement éloignées,
- augmentant le nombre de voisins distants et le coût de communication (échanges de cellules fantômes, synchronisations plus fréquentes).

On a donc un compromis classique :

- répartition géométrique simple → peu de communications, mais charge déséquilibrée,
- répartition par masse/densité → charge mieux équilibrée, mais plus de communications.

---

## 10. Piste Barnes–Hut (pour aller plus loin)

L’algorithme de Barnes–Hut construit un *quadtree* en 2D (ou *octree* en 3D) et atteint une complexité moyenne en $O(N \log N)$. Une idée de parallélisation avec MPI :

- Distribuer les sous-arbres du quadtree entre les processus, de sorte que chaque processus gère un certain nombre de boîtes (nodes) représentant un nombre similaire d’étoiles.
- Certains niveaux hauts de l’arbre (grosses boîtes englobantes) peuvent être répliqués sur plusieurs processus (en lecture seule), afin d’éviter une communication excessive lors du calcul des forces.
- Chaque processus :
  - construit localement sa partie de l’arbre,
  - participe à un échange global (par exemple `Allgather`) pour obtenir les centres de masse des boîtes de haut niveau,
  - calcule les contributions aux forces pour ses étoiles en parcourant à la fois l’arbre local et les boîtes globales nécessaires.

Cette approche réduirait la complexité asymptotique et pourrait mieux s’adapter à un grand nombre de corps et de processus, au prix d’une structure de données plus complexe.

---

## 11. Conclusion

En résumé :

- Nous avons implémenté et testé la parallélisation avec numba, la séparation affichage / calcul avec MPI, et la parallélisation du calcul avec MPI + numba.
- Pour le cas testé ($N = 1000$ étoiles), la version séquentielle avec numba 1 thread reste la plus rapide.
- Les versions parallèles sont pénalisées par l’overhead des threads et des communications MPI, ce qui est cohérent avec la taille relativement modeste du problème.
- Néanmoins, le travail met en évidence :
  - les techniques de parallélisation (threads, MPI, séparation I/O / calcul),
  - les problèmes de déséquilibre de charge et d’overhead,
  - et ouvre la voie à des tests sur des problèmes plus grands où le parallélisme deviendra réellement bénéfique.
