"""
features_lib.py

Logique de calcul de features PARTAGÉE entre :
- feature_engineering.py (pipeline batch, vectorisé pandas, calcule
  l'historique complet pour l'entraînement)
- predictor_service.py (API, calcule l'état COURANT d'une équipe pour
  prédire un match futur, une équipe à la fois)

Objectif de ce module : une seule source de vérité pour les constantes et
les formules (Elo, fenêtres glissantes), pour que le modèle ne soit jamais
entraîné sur une définition de feature et servi en production avec une
définition légèrement différente (dérive silencieuse, difficile à détecter
puisque le modèle continuerait à tourner sans erreur, juste avec de moins
bonnes prédictions).

Toutes les fonctions ici sont des fonctions PURES (pas de lecture de
fichier, pas d'I/O) : entrée -> sortie, testables isolément.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# --- Fenêtres de calcul (partagées batch + live) ---
ROLLING_WINDOWS = [5, 10]
FORM_WINDOW = 10  # fenêtre pour le win_pct "forme récente"

# Plafond des jours de repos. Au-delà d'une semaine, une équipe est
# pleinement reposée : 8-9 jours (pause du All-Star Game) et 130-290 jours
# (premier match d'une saison, après l'intersaison ou la bulle de 2020) ne
# diffèrent plus que par du bruit. Sans plafond, ces ~100 matchs d'ouverture
# donnent des valeurs que le modèle n'a presque jamais vues, et un match de
# reprise de saison est prédit hors distribution.
REST_DAYS_CAP = 7

# --- Configuration Elo (partagée batch + live) ---
ELO_INITIAL = 1500.0


@dataclass(frozen=True)
class EloParams:
    """Paramètres réglables de l'Elo (voir models/tune_elo.py)."""
    k: float                  # vitesse d'ajustement du rating après un match
    home_advantage: float     # bonus (en points d'Elo) accordé à l'équipe qui reçoit
    season_regression: float  # part de régression vers la moyenne entre 2 saisons

    def as_dict(self) -> dict:
        return asdict(self)


# Paramètres utilisés par l'entraînement ET l'API. Les changer impose de
# régénérer game_features.csv, de réentraîner, puis de régénérer
# team_state.json (sur le VPS : deploy_to_vps.ps1 puis sync_to_vps.ps1).
#
# Réglés le 2026-10-19 par models/tune_elo.py (log loss de l'Elo seul sur
# 2020-21 -> 2023-24, confirmé sur 2024-25 et 2025-26). Avant : K 20,
# avantage 100, régression 0,25, qui donnaient au domicile 62 % de chances
# en moyenne pour 55 % de victoires réelles. Avantage 40 : proba moyenne
# 55 % ; régression 0,5 : optimum net (plateau de 0,4 à 0,67).
DEFAULT_ELO_PARAMS = EloParams(k=20.0, home_advantage=40.0, season_regression=0.5)

# Colonnes de boxscore nécessaires pour estimer Pace / Net Rating.
BOX_SCORE_COLS = ["fga", "fta", "oreb", "tov"]


# ---------------------------------------------------------------------------
# Elo — fonctions pures, un match à la fois
# ---------------------------------------------------------------------------

def elo_expected_home(r_home: float, r_away: float, params: EloParams = DEFAULT_ELO_PARAMS) -> float:
    """
    Probabilité de victoire attendue pour l'équipe à domicile (avantage du
    terrain inclus). Accepte aussi des tableaux numpy / séries pandas.
    """
    return 1 / (1 + 10 ** (-((r_home + params.home_advantage) - r_away) / 400))


def elo_mov_multiplier(margin: float, elo_diff_for_winner: float) -> float:
    """Multiplicateur d'écart de score façon FiveThirtyEight."""
    return ((margin + 3) ** 0.8) / (7.5 + 0.006 * max(elo_diff_for_winner, 0))


def elo_update(
    r_home: float, r_away: float, home_win: float, home_score: float, away_score: float,
    params: EloParams = DEFAULT_ELO_PARAMS,
):
    """
    Met à jour le rating Elo des deux équipes après UN match.

    Fonction pure : ne connaît rien de l'historique, ne fait aucune I/O.
    Utilisée à la fois par compute_elo_timeline() (boucle sur tout
    l'historique pour l'entraînement) et potentiellement pour rejouer/
    vérifier un match précis.

    Retourne (nouveau_r_home, nouveau_r_away).
    """
    expected_home = elo_expected_home(r_home, r_away, params)
    actual_home = 1.0 if home_win == 1 else 0.0

    margin = abs(home_score - away_score)
    elo_diff_for_winner = (
        (r_home + params.home_advantage - r_away) if actual_home == 1.0
        else (r_away - (r_home + params.home_advantage))
    )
    mov_multiplier = elo_mov_multiplier(margin, elo_diff_for_winner)

    delta = params.k * mov_multiplier * (actual_home - expected_home)
    return r_home + delta, r_away - delta


def regress_rating(rating: float, params: EloParams = DEFAULT_ELO_PARAMS) -> float:
    """Régression partielle d'un rating vers la moyenne, appliquée au changement de saison."""
    return ELO_INITIAL + (1 - params.season_regression) * (rating - ELO_INITIAL)


def regress_elo_between_seasons(elo: dict, params: EloParams = DEFAULT_ELO_PARAMS) -> dict:
    """regress_rating() appliquée à toutes les équipes."""
    return {team_id: regress_rating(rating, params) for team_id, rating in elo.items()}


@dataclass
class EloTimeline:
    """Résultat du passage séquentiel sur tout l'historique."""
    per_game: pd.DataFrame        # nba_game_id, home_elo, away_elo (rating AVANT le match)
    final_state: dict             # {team_id: elo courant, après le dernier match connu}
    final_season: object          # saison du dernier match traité (pour savoir si une régression
                                   # inter-saison doit être appliquée avant de prédire un match futur)
    params: EloParams             # paramètres avec lesquels l'historique a été calculé


def compute_elo_timeline(df_games: pd.DataFrame, params: EloParams = DEFAULT_ELO_PARAMS) -> EloTimeline:
    """
    Calcule le rating Elo de chaque équipe match après match, dans l'ordre
    chronologique global (un match met à jour les deux équipes en même
    temps, donc traitement strictement séquentiel, pas de groupby par équipe).

    Retourne à la fois :
    - l'historique pré-match ligne par ligne (pour l'entraînement,
      anti-fuite : on enregistre le rating AVANT le match)
    - l'état final par équipe (pour la prédiction d'un match futur : c'est
      le rating "courant" de chaque équipe, à utiliser tel quel — voir
      current_elo_for_prediction() pour la régression inter-saison)
    """
    df = df_games.sort_values(["game_date", "nba_game_id"]).reset_index(drop=True).copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")

    n_bad = df[["home_score", "away_score", "home_win"]].isna().any(axis=1).sum()
    if n_bad:
        print(f"[WARN] {n_bad} matchs avec score/résultat non numérique -> ignorés pour l'Elo (rating inchangé)")

    seasons = df["season"].to_numpy()
    home_ids = df["home_team_id"].to_numpy()
    away_ids = df["away_team_id"].to_numpy()
    home_scores = df["home_score"].to_numpy(dtype=float)
    away_scores = df["away_score"].to_numpy(dtype=float)
    home_wins = df["home_win"].to_numpy(dtype=float)

    elo: dict = {}
    current_season = None
    home_elo_pre = []
    away_elo_pre = []

    for i in range(len(df)):
        season = seasons[i]
        home_id = home_ids[i]
        away_id = away_ids[i]

        if season != current_season:
            if current_season is not None:
                elo = regress_elo_between_seasons(elo, params)
            current_season = season

        r_home = elo.get(home_id, ELO_INITIAL)
        r_away = elo.get(away_id, ELO_INITIAL)

        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        home_score = home_scores[i]
        away_score = away_scores[i]
        home_win = home_wins[i]

        if np.isnan(home_score) or np.isnan(away_score) or np.isnan(home_win):
            continue

        elo[home_id], elo[away_id] = elo_update(r_home, r_away, home_win, home_score, away_score, params)

    per_game = pd.DataFrame({
        "nba_game_id": df["nba_game_id"],
        "home_elo": home_elo_pre,
        "away_elo": away_elo_pre,
    })

    return EloTimeline(per_game=per_game, final_state=elo, final_season=current_season, params=params)


def season_for_date(game_date) -> str:
    """
    Saison NBA (format "2025-26", comme la colonne `season` de
    games_clean.csv) à laquelle appartient une date. La saison régulière
    démarre en octobre : toute date à partir d'août est rattachée à la
    saison qui commence cette année-là (l'intersaison sert de frontière).
    """
    ts = pd.Timestamp(game_date)
    start_year = ts.year if ts.month >= 8 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def is_later_season(target_season, last_known_season) -> bool:
    """
    True si `target_season` est postérieure à `last_known_season`. Le
    format "YYYY-YY" se compare correctement en tant que chaîne.
    """
    return last_known_season is not None and str(target_season) > str(last_known_season)


def elo_for_target_season(
    rating: float, last_known_season, target_season, params: EloParams = DEFAULT_ELO_PARAMS,
) -> float:
    """
    Applique la régression inter-saison à un rating Elo courant si la
    saison cible est POSTÉRIEURE à la dernière saison connue (même règle
    que compute_elo_timeline() : régression au passage à une nouvelle
    saison, jamais en remontant le temps).

    Version "légère" de current_elo_for_prediction() : ne nécessite pas un
    EloTimeline complet (donc pas tout l'historique des matchs), juste le
    rating stocké et sa saison de référence — c'est ce que predictor_service.py
    lit depuis team_state.json (qui ne contient que l'état courant, pas
    l'historique complet).
    """
    if is_later_season(target_season, last_known_season):
        return regress_rating(rating, params)
    return rating


def current_elo_for_prediction(timeline: EloTimeline, team_id, target_season) -> float:
    """
    Rating Elo "courant" d'une équipe, à utiliser pour prédire un match à
    venir dans `target_season`. Nécessite un EloTimeline complet (calculé
    sur tout l'historique) — voir elo_for_target_season() pour la version
    légère utilisée en production à partir de team_state.json.
    """
    rating = timeline.final_state.get(team_id, ELO_INITIAL)
    return elo_for_target_season(rating, timeline.final_season, target_season, timeline.params)


# ---------------------------------------------------------------------------
# Stats glissantes / win% — fonctions pures sur une séquence de matchs passés
# ---------------------------------------------------------------------------
#
# Ces fonctions prennent en entrée l'historique d'UNE équipe, DÉJÀ trié
# chronologiquement et DÉJÀ filtré aux matchs qui précèdent le match à
# prédire (donc pas besoin de shift(1) ici : contrairement au pipeline
# batch qui calcule une colonne sur tout l'historique d'un coup, on ne
# reçoit ici que le passé pertinent).

def rolling_mean_last_n(values: list, n: int):
    """Moyenne des n dernières valeurs (ou moins si l'équipe a joué moins de n matchs). None si aucune."""
    if not values:
        return None
    return float(np.mean(values[-n:]))


def win_pct(results: list):
    """% de victoires sur la séquence de résultats (1 = victoire, 0 = défaite). None si aucun match."""
    if not results:
        return None
    return float(np.mean(results))


def rest_days(last_game_date, as_of_date) -> float | None:
    """
    Jours de repos depuis le dernier match, plafonnés à REST_DAYS_CAP (même
    plafond qu'à l'entraînement). None si aucun match précédent connu, ou si
    la date demandée précède le dernier match connu (date passée interrogée
    avec l'état actuel des équipes : valeur sans objet).
    """
    if last_game_date is None:
        return None
    days = (pd.Timestamp(as_of_date) - pd.Timestamp(last_game_date)).days
    if days < 0:
        return None
    return float(min(days, REST_DAYS_CAP))


def net_rtg_from_boxscore(pts_scored: float, pts_allowed: float, fga: float, oreb: float, tov: float, fta: float):
    """
    Net Rating estimé pour UN match, à partir du boxscore de l'équipe
    (même approximation que add_advanced_ratings() dans feature_engineering.py :
    le nombre de possessions de l'équipe sert de dénominateur pour attaque
    ET défense, faute du boxscore adverse aligné sur la même ligne).
    """
    poss = fga - oreb + tov + 0.4 * fta
    if not poss:
        return None
    off_rtg = 100 * pts_scored / poss
    def_rtg = 100 * pts_allowed / poss
    return off_rtg - def_rtg