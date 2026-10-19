# Architecture

Ce document explique comment le projet est construit et pourquoi. Pour lancer l'application, voir le [README](../README.md).

## Vue d'ensemble

```
PC de développement (IP résidentielle)
  nba_api ──> data/collectors ──> data/preprocessing ──> data/models
                                        │                    │
                          team_state.json, schedule.json    logreg.pkl, feature_columns.json,
                                        │                    model_metrics.json
                                        ▼                    ▼
                                 api/ (FastAPI) ── lit les fichiers, prédit à la demande
                                        │ JSON
                                        ▼
                                 front/ (Next.js) ── 3 pages
```

Il n'y a pas de base de données : des fichiers CSV et JSON font office de stockage. C'est un choix assumé à cette échelle (11 969 matchs de saison régulière depuis 2016-17, 30 équipes), qui évite une dépendance de plus. Le schéma SQL cible est décrit plus bas.

## Pipeline de données (`data/`)

| Étape | Script | Entrée → sortie |
| --- | --- | --- |
| Collecte | `collectors/fetch_games.py` | `nba_api` → `raw/games_raw_all_seasons.csv` (1 ligne par équipe) puis `raw/games_history.csv` (1 ligne par match) |
| Nettoyage | `preprocessing/clean_data.py` | → `processed/games_clean.csv` (typage strict, cible `home_win`, tri chronologique) |
| Features | `preprocessing/feature_engineering.py` | → `processed/game_features.csv` |
| Entraînement | `models/train_model.py` | → `saved_models/*.pkl`, `feature_columns.json`, `model_metrics.json` (dont le paramètre du score estimé, calculé par `models/score_model.py`) |
| Évaluation seule | `models/evaluate_model.py` | walk-forward et comparaisons de variantes, sans toucher aux modèles servis → `saved_models/eval/` |
| Réglage de l'Elo | `models/tune_elo.py` | grille de paramètres notée sur l'Elo seul → `saved_models/eval/tune_elo_grid.csv` |
| Contrôles | `preprocessing/check_features.py` | cohérence entraînement / API et absence de fuite temporelle |
| État courant | `preprocessing/team_state.py` | → `processed/team_state.json` |
| Calendrier | `collectors/fetch_schedule.py` | → `processed/schedule.json` (J-3 à J+14) |

### Les variables du modèle

Pour chaque équipe, calculées **avant** le match :

- **Elo** : note qui monte après une victoire, d'autant plus que l'adversaire était fort et l'écart large (même principe que FiveThirtyEight). L'équipe qui reçoit a un bonus de 40 points, et la note revient de 50 % vers la moyenne (1500) entre deux saisons (paramètres réglés, voir « Régler l'Elo »).
- **Bilan** : pourcentage de victoires de la saison, spécifique au contexte (domicile ou extérieur), et sur les 10 derniers matchs.
- **Points marqués et encaissés** : moyennes sur 5 et 10 matchs.
- **Net Rating estimé** : (points marqués − encaissés) rapportés aux possessions estimées à partir du boxscore.
- **Jours de repos**, plafonnés à 7.
- **Différences** domicile − extérieur de chacune, généralement plus discriminantes qu'une paire de colonnes séparées.

Le modèle n'utilise que les différences et l'Elo de chaque équipe (11 variables, voir « Choisir le modèle »). Les autres colonnes restent calculées : la fiche match les affiche.

### Éviter la fuite temporelle

Une feature qui utilise une information connue seulement après le match donne un modèle excellent en test et inutile en production. Quatre protections :

1. Chaque statistique glissante est décalée d'un match (`shift(1)`), et l'Elo enregistré est celui d'**avant** le match.
2. Chaque saison testée est prédite par un modèle entraîné uniquement sur les saisons qui la précèdent : pas de découpage aléatoire (voir « Évaluer honnêtement »).
3. `game_features.csv` contient aussi le boxscore brut du match joué (`home_fga`, `away_oreb`…), connu seulement après coup. `train_model.py` choisit ses colonnes par **liste blanche de motifs** (`FEATURE_PATTERNS`) et jamais par liste noire : une nouvelle colonne ne devient pas une feature par accident.
4. `check_features.py` le vérifie automatiquement : il inverse le résultat de quelques matchs et recalcule toutes les features. Celles du match et des matchs antérieurs ne doivent pas bouger, celles des matchs suivants si. Le même script contrôle que l'API calcule exactement les mêmes features que l'entraînement (50 matchs, `team_state.json` reconstruit avec les seuls matchs antérieurs, écart toléré 1e-9), et que l'explication renvoyée pour ces matchs redonne bien la probabilité.

### Évaluer honnêtement

Jusqu'au 2026-10-19, le modèle était testé sur une seule saison (2025-26, 1 186 matchs) : 69 % de précision, avec un intervalle de confiance d'environ ±2,7 points. Deux problèmes : impossible de dire si une modification aide, et une saison peut être plus facile qu'une autre.

`evaluate_model.py` fait désormais une validation **walk-forward** : 2021-22 est prédite par un modèle entraîné sur les saisons 2016-17 à 2020-21 (2019-20 et 2020-21 seulement avant l'extension de l'historique), 2022-23 par un modèle entraîné sur toutes les saisons précédentes, et ainsi de suite jusqu'à 2025-26 (6 140 matchs prédits). Les matchs où une variable manque (premier match de chaque équipe dans la saison) sont prédits aussi, puisque le site les affiche. XGBoost tourne avec 5 graines aléatoires et on garde la moyenne.

Deux variantes se comparent sur les **mêmes matchs** : on calcule l'écart de log loss match par match, puis un intervalle de confiance par bootstrap en tirant au hasard des journées entières (les matchs d'un même soir ne sont pas indépendants). Cette comparaison appariée est bien plus sensible que de comparer deux intervalles séparés. Règle : un changement n'est adopté que si l'intervalle de l'écart de log loss est entièrement sous zéro.

| Modèle | Précision | Log loss | Brier |
| --- | --- | --- | --- |
| Règle naïve (taux de victoire à domicile) | 55,3 % | 0,688 | 0,247 |
| **Régression logistique (servie par l'API)** | 65,7 % | 0,616 | 0,214 |
| XGBoost | 65,6 % | 0,620 | 0,215 |
| Elo seul (sans apprentissage) | 65,9 % | 0,618 | 0,215 |

(État du 2026-10-19 après le réglage de l'Elo, le choix du modèle et l'extension de l'historique. Au départ, avec 31 variables et l'Elo d'origine : XGBoost 0,635, régression logistique 0,620, Elo seul 0,632.)

Ce que ça a montré :
- La précision varie de 63 % (2022-23) à 69 % (2025-26) d'une saison à l'autre. Le 69 % mesuré sur la seule saison 2025-26 surestimait ce qu'on peut attendre en moyenne.
- Avant le choix du modèle, XGBoost, alors servi, donnait des probabilités nettement moins bonnes que la régression logistique et que l'Elo seul (écart de log loss de 0,014 à 0,015, dans le même sens sur les 5 saisons). D'où les essais de « Choisir le modèle ».
- Résultat négatif : entraîner aussi sur les matchs où une variable manque n'apporte rien de mesurable. L'entraînement se fait donc sur les matchs complets.

### Régler l'Elo

Les paramètres d'origine (K = 20, avantage du terrain de 100 points, régression de 25 %) venaient de valeurs usuelles. Avec un avantage de 100, l'Elo donnait au domicile 62 % de chances en moyenne, pour 55 % de victoires réelles.

`tune_elo.py` essaie 396 combinaisons (K de 10 à 35, avantage de 0 à 100, régression de 0 à 50 %) et rejoue l'Elo sur tout l'historique pour chacune, avec le code de `features_lib.py`. Chaque combinaison est notée sur la log loss de l'Elo seul de 2020-21 à 2023-24 (2019-20 sert de rodage : toutes les équipes partent de 1500). 2024-25 et 2025-26 ne servent qu'à confirmer.

Résultat : K = 20, avantage de 40, régression de 50 %. Sur les deux saisons de confirmation, la log loss de l'Elo seul passe de 0,616 à 0,600 (écart −0,016, intervalle −0,023 à −0,010), et la probabilité moyenne à domicile tombe sur le taux réel (55,0 % pour 54,9 %). La régression de 50 % est au bord de la grille, mais l'optimum est réel : au-delà (60 %, 75 %, 100 %), la log loss remonte.

Effet sur les modèles complets (walk-forward, mêmes matchs) : XGBoost −0,0016 (−0,0035 à +0,0001), régression logistique −0,0009 (−0,0018 à +0,0001), Brier en baisse. Gain modeste mais adopté : le critère fixé (pas de dégradation, borne haute sous +0,001) est rempli, et l'Elo affiché sur le site devient une probabilité calibrée.

Les paramètres sont écrits dans `model_metrics.json` (`elo_params`) et dans `team_state.json`. Si les deux diffèrent, l'API journalise un avertissement : c'est le cas d'un déploiement de nouveaux paramètres sans synchro derrière, où le modèle recevrait des Elo calculés sur une autre échelle que ceux de son entraînement.

Les modèles servis sont ensuite réentraînés sur **toutes** les saisons, 2025-26 comprise, avec les mêmes poids par saison qu'en évaluation : c'est la situation que simule chaque pli (prédire la saison suivante avec tout le passé connu).

### Choisir le modèle

Cinq essais, un par un, chacun adopté seulement s'il passe la règle de la comparaison appariée. Réglages sur 2021-22 à 2023-24, confirmation sur l'ensemble et sur 2024-25 + 2025-26 :

| Essai | Écart de log loss (intervalle à 95 %) | Décision |
| --- | --- | --- |
| Bilan de la saison lissé en début de saison : (V + 4) / (N + 8) au lieu d'une valeur manquante | XGBoost −0,0011 (−0,0039 à +0,0019), mais +0,0028 sur les 2 saisons de confirmation ; régression logistique −0,0002 | Écarté |
| Garder seulement les écarts domicile − extérieur et l'Elo des deux équipes (11 variables au lieu de 31) | Régression logistique −0,0016 (−0,0026 à −0,0006), même sens sur les 5 saisons ; XGBoost dans le bruit | Adopté |
| XGBoost avec arrêt précoce (profondeur 2, environ 150 arbres au lieu de 300) | −0,0113 (−0,0152 à −0,0076), 5 saisons sur 5 | Adopté pour XGBoost |
| Modèle de marge : régression Ridge sur l'écart de points, P = Φ(μ/σ) | +0,0000 (−0,0011 à +0,0011) face à la régression logistique | Écarté (égalité, plus complexe), à reprendre pour le score |
| Moyennes de plusieurs modèles, en log-odds | Toutes dans le bruit face à la régression logistique seule | Écartées |

Une recalibration (Platt) n'était prévue que si l'erreur de calibration dépassait 2 points : elle est de 1,5 point.

**Modèle servi : la régression logistique**, qui a la meilleure log loss. Face à XGBoost amélioré : −0,0040 (−0,0064 à −0,0015), 5 saisons sur 5 ; face à l'Elo seul : −0,0010 (−0,0030 à +0,0011), dans le bruit. Face à l'ancien modèle servi (XGBoost à 31 variables) : −0,0156 (−0,0202 à −0,0106) et +0,9 point de précision. Sur les 107 premiers matchs de saison régulière 2026-27, ses probabilités suivent celles de l'Elo seul (corrélation 0,98, 56 % en moyenne pour l'équipe qui reçoit) ; l'ancien XGBoost sous-estimait le domicile (53,5 %).

### Étendre l'historique et pondérer les saisons

L'historique commençait en 2019-20. Il part désormais de 2016-17 (trois saisons, 3 690 matchs de plus). Seul, l'ajout reste dans le bruit : régression logistique −0,0006 (−0,0015 à +0,0003), et l'Elo seul ne bouge pas du tout. Avec une régression inter-saison de 50 %, l'influence du point de départ de l'Elo a disparu au bout de deux saisons : un « meilleur rodage » ne change rien à partir de 2021-22.

La raison se lit dans les données : on gagnait plus souvent à domicile en 2016-19 (58-59 %) que depuis 2020 (54-55 %). Entraînée sur toutes ces saisons à égalité, la régression logistique surestime le domicile (+1,3 point de probabilité moyenne en trop). D'où un poids par saison qui diminue de moitié toutes les deux saisons (la plus récente compte 1, celle d'avant 0,71, etc.). La demi-vie a été réglée sur 2021-22 à 2023-24, où les résultats sont plats de 1 à 5 saisons.

| Variante (régression logistique, face à l'historique 2019-20 sans poids) | Écart de log loss (intervalle à 95 %) |
| --- | --- |
| Historique 2016-17, sans poids | −0,0006 (−0,0015 à +0,0003) |
| Historique 2019-20, poids (demi-vie 2) | −0,0004 (−0,0009 à +0,0000) |
| **Historique 2016-17, poids (demi-vie 2)** | **−0,0010 (−0,0017 à −0,0003)**, Brier −0,0005, 4 saisons sur 5 |

Seule la combinaison des deux passe la règle. Le gain est petit, mais le biais domicile retombe à +0,6 point et l'erreur de calibration de 0,015 à 0,014. XGBoost profite aussi des poids (−0,0019, −0,0034 à −0,0005, 5 saisons sur 5) alors que l'historique seul lui coûtait 0,5 point de précision. Sur les 107 premiers matchs de 2026-27, les probabilités servies bougent peu : 1,3 point en moyenne, 3,3 au plus.

### Expliquer une prédiction

L'explication d'une prédiction reste exacte. Pour un modèle linéaire, la contribution d'une variable est son coefficient multiplié par sa valeur standardisée (0 = valeur moyenne de l'entraînement), en log-odds ; le biais plus la somme des contributions redonne la probabilité. Le bilan des 10 derniers matchs et le Net Rating récent forment une seule famille, « Forme récente » : corrélés à 0,88, ils se partagent le même effet de façon instable (le coefficient du bilan récent passe de +0,04 à −0,13 selon les saisons d'entraînement). Séparés, ils donneraient des facteurs de signes opposés difficiles à lire.

### `features_lib.py` : une seule définition des formules

L'entraînement calcule les features sur tout l'historique (vectorisé, pandas). L'API les calcule pour un seul match futur à partir de `team_state.json`. Si les deux codes divergeaient, le modèle tournerait sans erreur mais avec de moins bonnes prédictions, sans qu'on le voie. `features_lib.py` contient donc les constantes (fenêtres, paramètres Elo `DEFAULT_ELO_PARAMS`, plafond de repos) et les fonctions pures utilisées des deux côtés. La non-régression a été vérifiée : l'ancienne et la nouvelle implémentation de l'Elo donnent un écart de 0,0 sur les 8 279 matchs de l'époque. Les formules du score estimé y sont aussi, et `check_features.py` vérifie que l'API sert exactement le score qu'elles donnent sur les features d'entraînement.

### Cas particuliers gérés à la prédiction

- **Changement de saison** : la saison d'un match est déduite de sa date (frontière en août). Si elle est postérieure à celle de `team_state.json`, on reproduit ce que fait l'entraînement : Elo régressé vers la moyenne (50 %) et bilans de la saison remis à zéro (manquants). Les moyennes glissantes ne sont pas remises à zéro, comme à l'entraînement.
- **Valeurs manquantes** (bilan de la saison avant le premier match) : remplacées par la médiane de l'entraînement, par le modèle lui-même (`SimpleImputer` en tête du pipeline), comme pendant l'évaluation. Leur contribution à l'explication est alors presque nulle.
- **Jours de repos** : plafonnés à 7 à l'entraînement et en prédiction (`REST_DAYS_CAP`). Sans plafond, les ~100 matchs d'ouverture de saison de l'historique (130 à 288 jours après le dernier match, à cause de l'intersaison et de la « bulle » de 2020) donnent des valeurs que le modèle voit trop peu. Mesuré sur 5 graines : le plafond ne change pas la qualité de façon mesurable (69,1 % sans plafond contre 68,6 % avec, pour un écart-type de 0,4 point entre graines). Une date antérieure au dernier match connu donne un repos manquant plutôt qu'un nombre négatif.

## API (`api/`)

FastAPI, un routeur par domaine. La logique est dans `services/`, les routes dans `routers/`.

- `services/predictor_service.py` : charge une seule fois le modèle, `feature_columns.json` et `team_state.json`, construit les features d'un match et prédit. Relit `team_state.json` quand sa date de modification change, sans redémarrage. Calcule aussi l'explication : les contributions de la régression logistique (`features_lib.linear_contributions`, coefficient × valeur standardisée) en log-odds, sommées par famille de statistiques (`FACTOR_FAMILIES`, aussi utilisées pour les poids de la page « Le modèle »). La sigmoïde de (biais + somme des contributions) redonne exactement la probabilité affichée. Avertit dans le journal si les paramètres Elo de `team_state.json`, de `model_metrics.json` et du code diffèrent.
- `services/schedule_service.py` : lit `schedule.json`. Un appel direct à la NBA n'a lieu que si `NBA_LIVE_SCHEDULE_FALLBACK` est actif (développement local).
- `routers/games.py` : `/games/today`, `/games/calendar`, `/games/{game_id}`. Erreurs distinguées : 400 date mal formée, 404 date non synchronisée, 502 calendrier NBA indisponible, 503 modèle non prêt. Un match avec une équipe inconnue renvoie `prediction: null` sans faire échouer le reste de la liste.
- `routers/model.py` : `/model/metrics`.

### Limite connue : pas d'historique par date

`team_state.json` ne contient que l'état **actuel** des équipes. Interroger une date passée donne donc les matchs réels de ce jour mais avec l'Elo et la forme d'aujourd'hui : utile pour tester l'interface, inutile pour un vrai backtest.

### Le score estimé

Le modèle ne classe que victoire ou défaite ; le score s'en déduit en deux temps (formules dans `features_lib.py`, évaluation dans `models/score_model.py`) :

- **Total** des deux équipes : pour chacune, la moyenne de ses points marqués et des points encaissés par l'adversaire sur les 10 derniers matchs.
- **Écart** : σ · Φ⁻¹(p), où p est la probabilité de victoire à domicile du modèle servi. Si l'écart final suit une loi normale de moyenne μ et d'écart-type σ, alors p = Φ(μ/σ), donc μ = σ · Φ⁻¹(p). σ (12,8 points) est ajusté par moindres carrés sur les matchs d'entraînement et écrit dans `model_metrics.json`.

Le score est arrondi à l'entier, sans égalité possible : si les deux arrondis tombent sur la même valeur, le favori prend un point. Le vainqueur du score est donc toujours celui du modèle. L'ancienne méthode estimait chaque équipe séparément (sa forme offensive et la défense adverse) et contredisait le favori sur 22 % des matchs.

Mesures en walk-forward (6 140 matchs, erreur absolue moyenne sur les scores affichés) :

| Méthode | Par équipe | Sur l'écart | Sur le total |
| --- | --- | --- | --- |
| Moyennes de la saison précédente (naïf) | 10,2 | 12,3 | 16,2 |
| Ancienne méthode | 9,4 | 11,4 | 14,9 |
| **Écart déduit de la probabilité** | **9,3** | **10,9** | **14,9** |

Gain par équipe −0,15 point (−0,17 à −0,11), sur l'écart −0,52 (−0,62 à −0,43), même sens sur les 5 saisons. Le total ne change pas, c'est la même formule. Une régression du total (linéaire ou Ridge, moyennes sur 5 et 10 matchs, jours de repos, poids par saison) a été essayée : aucune variante ne fait mieux (écarts de −0,004 à +0,02 point, dans le bruit), on garde la formule, plus simple. Un σ estimé autrement (écart-type résiduel d'une régression de l'écart, 13,5) donne la même erreur à 0,01 point près.

## Front (`front/`)

Next.js (App Router), JavaScript, Tailwind CSS v4 (configuration dans `globals.css`, pas de `tailwind.config.js`).

- **Pages** : `/` (matchs du jour, date dans l'URL), `/match?id=…&date=…`, `/modele`. Le front est un export statique (`output: "export"`) : aucune route dynamique côté serveur, d'où l'identifiant du match en paramètre plutôt que dans le chemin.
- **Couleurs** : chaque couleur est déclarée une fois en `light-dark(clair, sombre)`. `npm run check:contrast` vérifie toutes les paires (7:1 pour le texte, 3:1 pour les graphiques) : à relancer après toute modification de couleur. Le rouge NBA n'est jamais utilisé pour du texte (contraste insuffisant).
- **Graphiques** : SVG et HTML maison, sans bibliothèque. Chacun a un `aria-label`, un tableau de données alternatif (« Voir les données ») et une infobulle au survol.
- **Animations** : `motion`, avec respect de `prefers-reduced-motion`.
- **Logos** : les 60 fichiers SVG sont versionnés dans `public/team-logos/` et servis par le front, car `cdn.nba.com` est bloqué depuis certains réseaux.
- **Calendrier** : composant fait maison (bandeau de 14 jours + grille mensuelle, clavier complet : flèches, Début/Fin, Page précédente/suivante, Entrée, Échap).

## Pourquoi la synchronisation part du PC

`stats.nba.com` ne répond pas aux requêtes venant d'IP de datacenters (le serveur laisse la connexion ouverte sans répondre), et `cdn.nba.com` renvoie un 403 (Akamai). Testé depuis un VPS OVH : les deux sont bloqués. L'architecture de déploiement en découle :

```
PC (tâche planifiée Windows, 1 fois par jour)
  infra/sync_to_vps.ps1 : fetch_games --refresh → clean_data → team_state → fetch_schedule
        │  scp vers un fichier .tmp puis mv atomique (SSH par clé)
        ▼
Serveur : data/processed/team_state.json + schedule.json
        ▲ relus automatiquement par l'API quand leur date de modification change
Internet ──HTTPS──> Caddy ──> /     fichiers statiques du front (export Next.js)
                          └─> /api  API FastAPI, service systemd isolé (NBA_LIVE_SCHEDULE_FALLBACK=0)
```

Détails du déploiement (isolation de l'API, redéploiement, retour arrière) : [deploiement.md](deploiement.md).

L'API sur le serveur ne fait aucun appel réseau vers la NBA. Si le PC reste éteint plusieurs jours, les prédictions s'appuient sur un état de plus en plus ancien, puis le calendrier ne couvre plus la date du jour (404) : le service se dégrade, il ne tombe pas.

## Schéma de base de données (cible, non implémenté)

Si les fichiers deviennent insuffisants : PostgreSQL local (conteneur Docker). Tables prévues : `teams`, `games`, `game_features`, `predictions` (version du modèle, vainqueur prédit, probabilité), `model_metrics`. Les colonnes de `game_features` devront suivre les vraies variables du modèle (Elo, Net Rating, bilan selon le contexte), pas le schéma initial du projet.

## Décisions et pistes

| Sujet | Décision |
| --- | --- |
| Elo plutôt que bilan seul | L'Elo pondère par la force de l'adversaire : +2 points de précision, la variable la plus importante du modèle |
| Bilan selon le contexte **en plus** du bilan global | Le remplacer abîmait la précision (échantillon divisé par deux, environ 20 matchs par saison) |
| Régression logistique avec `StandardScaler` | L'Elo (1300 à 1700) et les pourcentages (0 à 1) ont des échelles très différentes |
| Matchs de présaison | Gardés, avec un badge « Présaison » : le modèle n'est entraîné que sur la saison régulière |
| Validation walk-forward plutôt qu'une saison de test | Mesure la variation d'une saison à l'autre et permet des comparaisons appariées assez sensibles pour décider |
| Entraînement sur les matchs complets | Garder les matchs incomplets n'aidait pas (écart dans le bruit) et dégradait la régression logistique sur ces matchs |
| Paramètres Elo réglés sur l'Elo seul | Objectif propre à l'Elo (pas au modèle complet), réglage sur 2020-21 à 2023-24, confirmation sur 2 saisons jamais vues : avantage du terrain 100 → 40, régression 25 % → 50 % |
| Régression logistique servie plutôt que XGBoost | Meilleure log loss en walk-forward (écart significatif, même sens sur les 5 saisons) et explication exacte de chaque prédiction |
| 11 variables seulement (écarts + Elo) | Les 20 autres colonnes n'aidaient aucun modèle et gênaient la régression logistique ; elles restent affichées sur la fiche match |
| Historique depuis 2016-17, saisons pondérées (demi-vie 2) | Plus de données, sans que l'avantage du terrain plus fort de 2016-19 ne biaise le modèle ; ni l'un ni l'autre ne suffisait seul |
| Écart du score déduit de la probabilité | Erreur plus faible que l'ancienne méthode, et le score ne contredit jamais le favori ; la régression du total n'apportait rien |
| Pas de variables de fatigue ni de déplacement | Matchs des 4 derniers jours, distance parcourue et décalage horaire essayés (2026-10-19) : aucun gain qui tienne sur les saisons non utilisées pour le réglage ; les jours de repos captent déjà les back-to-back |
| Piste restante (plan du 2026-10-19) | Joueurs absents |
