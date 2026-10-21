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
from scipy.special import ndtri  # Φ⁻¹, inverse de la fonction de répartition de la loi normale

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


# ---------------------------------------------------------------------------
# Effectif (absences) — partagé entre feature_engineering.py (batch) et
# team_state.py (état courant, lu par l'API)
# ---------------------------------------------------------------------------
#
# Les blessures ne sont connues avec certitude qu'une demi-heure avant le
# match ; la synchro tourne à 10 h. On ne peut donc s'appuyer que sur les
# matchs déjà joués : un joueur absent du dernier match de son équipe est
# supposé encore absent.
#
# - valeur d'un joueur : Game Score moyen (Hollinger) sur ses
#   PLAYER_VALUE_GAMES dernières apparitions avant le match (toutes équipes,
#   saisons précédentes comprises) ;
# - force d'un effectif : somme des LINEUP_TOP meilleures valeurs parmi les
#   joueurs entrés en jeu (le top 8 neutralise les fins de match pliées, où
#   les remplaçants de fin de banc entrent) ;
# - lineup_delta : force de l'effectif du dernier match de l'équipe moins la
#   force moyenne de ses LINEUP_WINDOW derniers matchs de la saison, toutes
#   évaluées avec les valeurs des joueurs à la date du match. Négatif :
#   des joueurs importants manquaient au dernier match, ce que l'Elo et la
#   forme récente (calculés sur l'effectif habituel) ne voient pas encore.
#   0 au premier match de la saison (aucune information).
#
# Mesuré le 2026-10-19 en walk-forward (CLAUDE.md, "Étape 6") : régression
# logistique -0,0023 de log loss [-0,0041 ; -0,0007], tenu sur 2024-25 +
# 2025-26 ; le gain se fait sur le tiers des matchs où l'effectif a changé.

PLAYER_VALUE_GAMES = 15
LINEUP_WINDOW = 10
LINEUP_TOP = 8

# Colonnes des feuilles de match par joueur (preprocessing/player_games.py).
PLAYER_STAT_COLS = ["pts", "fgm", "fga", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]


def game_score(stats) -> pd.Series:
    """Game Score de Hollinger d'une ou plusieurs lignes joueur-match (colonnes PLAYER_STAT_COLS)."""
    s = stats
    return (
        s["pts"] + 0.4 * s["fgm"] - 0.7 * s["fga"] - 0.4 * (s["fta"] - s["ftm"]) + 0.7 * s["oreb"]
        + 0.3 * s["dreb"] + s["stl"] + 0.7 * s["ast"] + 0.7 * s["blk"] - 0.4 * s["pf"] - s["tov"]
    )


def player_value_timeline(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Valeur de chaque joueur APRÈS chacune de ses apparitions (moyenne de son
    Game Score sur ses PLAYER_VALUE_GAMES dernières apparitions, celle-ci
    comprise) : c'est sa valeur pour tout match d'une date ultérieure.
    Colonnes : player_id, game_date, value.
    """
    p = player_games.sort_values(["player_id", "game_date", "nba_game_id"]).copy()
    p["gmsc"] = game_score(p)
    p["value"] = (
        p.groupby("player_id")["gmsc"].rolling(PLAYER_VALUE_GAMES, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return p[["player_id", "game_date", "value"]]


def lineup_strength(values) -> float:
    """Force d'un effectif : somme des LINEUP_TOP meilleures valeurs."""
    return float(np.sort(np.asarray(values, dtype=float))[::-1][:LINEUP_TOP].sum())


def lineup_delta(strengths: list) -> float:
    """
    Force du dernier match moins la force moyenne des LINEUP_WINDOW derniers
    (liste du plus ancien au plus récent, matchs de la saison en cours ;
    None/NaN = match sans feuille joueur, ignoré dans la moyenne). 0 si la
    liste est vide ou si le dernier match n'a pas de feuille joueur.
    """
    window = np.array([np.nan if s is None else s for s in strengths[-LINEUP_WINDOW:]], dtype=float)
    if len(window) == 0 or np.isnan(window[-1]):
        return 0.0
    return float(window[-1] - np.nanmean(window))


def compute_lineup_deltas(team_games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """
    lineup_delta de chaque équipe AVANT chacun de ses matchs (version batch,
    pour l'entraînement ; même calcul que lineup_delta() appliqué match par
    match, vérifié par check_features.py).

    `team_games` : une ligne par équipe et par match (team_id, nba_game_id,
    game_date, season). `player_games` : feuilles de match par joueur, déjà
    limitées aux matchs de `team_games`. Retourne team_id, nba_game_id,
    lineup_delta.
    """
    tg = team_games[["team_id", "nba_game_id", "game_date", "season"]].sort_values(
        ["team_id", "game_date", "nba_game_id"]
    ).reset_index(drop=True)
    tg["idx"] = tg.groupby(["team_id", "season"]).cumcount()

    # Participants de chaque match passé, rattachés aux LINEUP_WINDOW matchs
    # suivants de la même équipe dans la saison (lag 1 = dernier match joué).
    part = player_games[["player_id", "team_id", "nba_game_id"]].merge(
        tg[["team_id", "nba_game_id", "season", "idx"]], on=["team_id", "nba_game_id"]
    )
    targets = tg.rename(columns={"idx": "target_idx", "nba_game_id": "target_id", "game_date": "target_date"})
    w = pd.concat(
        [part.assign(target_idx=part["idx"] + lag, lag=lag) for lag in range(1, LINEUP_WINDOW + 1)],
        ignore_index=True,
    ).merge(targets, on=["team_id", "season", "target_idx"])

    # Valeur de chaque joueur juste avant la date du match cible.
    values = player_value_timeline(player_games).sort_values("game_date")
    w = pd.merge_asof(
        w.sort_values("target_date"), values.rename(columns={"game_date": "value_date"}),
        left_on="target_date", right_on="value_date", by="player_id", allow_exact_matches=False,
    )

    # lineup_strength() de chaque effectif passé, vectorisé : LINEUP_TOP
    # meilleures valeurs de chaque (équipe, match cible, lag), sommées.
    keys = ["team_id", "target_id", "lag"]
    w = w.sort_values(keys + ["value"], ascending=[True, True, True, False])
    top = w[w.groupby(keys).cumcount() < LINEUP_TOP]
    strength = (
        top.groupby(keys)["value"].sum().unstack("lag")
        .reindex(columns=range(1, LINEUP_WINDOW + 1))
    )
    delta = (strength[1] - strength.mean(axis=1)).fillna(0.0)
    out = tg[["team_id", "nba_game_id"]].merge(
        delta.rename("lineup_delta").rename_axis(["team_id", "nba_game_id"]).reset_index(),
        on=["team_id", "nba_game_id"], how="left",
    )
    out["lineup_delta"] = out["lineup_delta"].fillna(0.0)  # premier match de la saison
    return out


# ---------------------------------------------------------------------------
# Ratings de score — total de points attendu, partagé entre
# feature_engineering.py / team_state.py (batch) et predictor_service.py
# ---------------------------------------------------------------------------
#
# Total d'un match = possessions × points par possession. Deux notes par
# équipe, mises à jour après chaque match comme l'Elo, donc ajustées aux
# adversaires rencontrés (une moyenne brute des derniers matchs ne l'est
# pas) :
# - pace : possessions de ses matchs, au-dessus de la moyenne de la ligue ;
# - scoring : points pour 100 possessions produits dans ses matchs (les
#   deux équipes réunies), au-dessus de la moyenne. Élevé = attaque
#   efficace et/ou défense perméable.
# Le niveau de la ligue (possessions, points pour 100 possessions par
# équipe) suit lui aussi les matchs : le jeu évolue (211 points par match
# en 2016-17, 231 en 2025-26).
#
# Une seule note « scoring » plutôt qu'attaque et défense : pour le TOTAL,
# seule compte la différence attaque − défense de chaque équipe (vérifié le
# 2026-10-21 : séparées, elles donnent les mêmes totaux à l'identique, et
# l'avantage du terrain n'y joue aucun rôle). L'écart vient de la
# probabilité du modèle (expected_margin).
#
# Réglage et évaluation (2026-10-21) : models/tune_score_ratings.py et
# models/score_model.py.


@dataclass(frozen=True)
class ScoreRatingParams:
    """Paramètres réglables des ratings de score (voir models/tune_score_ratings.py)."""
    k_pace: float             # part de l'écart de possessions reportée sur la note de rythme
    k_scoring: float          # idem pour l'écart de points pour 100 possessions
    season_regression: float  # part de régression des notes vers 0 entre 2 saisons
    league_rate: float        # vitesse de suivi du niveau de la ligue

    def as_dict(self) -> dict:
        return asdict(self)


# Réglés le 2026-10-21 sur la MAE du total des saisons 2021-22 -> 2023-24.
# Comme pour l'Elo, les changer impose de régénérer game_features.csv, de
# réentraîner, puis de régénérer team_state.json (deploy puis sync).
DEFAULT_SCORE_RATING_PARAMS = ScoreRatingParams(k_pace=0.04, k_scoring=0.03, season_regression=0.5, league_rate=0.01)

# Point de départ (premier match de l'historique, 2016-17), proche de la
# réalité de l'époque et oublié pendant les saisons de rodage : il n'est pas
# estimé sur les matchs eux-mêmes.
LEAGUE_INITIAL = {"pace": 100.0, "efficiency": 106.0}
NEUTRAL_SCORE_RATING = {"pace": 0.0, "scoring": 0.0}  # équipe moyenne (ou jamais vue)


def game_possessions(df: pd.DataFrame) -> pd.Series:
    """
    Possessions d'un match : moyenne des estimations des deux équipes
    (FGA + 0,44 FTA - OREB + TOV, la formule usuelle). Différente de celle
    de net_rtg_from_boxscore (0,4, une seule équipe), feature du modèle
    qu'on ne modifie pas.
    """
    def team(side: str) -> pd.Series:
        return df[f"{side}_fga"] + 0.44 * df[f"{side}_fta"] - df[f"{side}_oreb"] + df[f"{side}_tov"]
    return (team("home") + team("away")) / 2


def expected_game(league: dict, home: dict, away: dict) -> tuple[float, float]:
    """(possessions, total de points) attendus, à partir des notes d'avant-match."""
    pace = league["pace"] + home["pace"] + away["pace"]
    return pace, pace * (2 * league["efficiency"] + home["scoring"] + away["scoring"]) / 100


def score_rating_errors(
    league: dict, home: dict, away: dict, possessions: float, total_points: float,
) -> tuple[float, float]:
    """Écarts d'UN match à sa prévision : (possessions, points pour 100 possessions des deux équipes)."""
    pace, _ = expected_game(league, home, away)
    efficiency = 2 * league["efficiency"] + home["scoring"] + away["scoring"]
    return possessions - pace, 100 * total_points / possessions - efficiency


def team_rating_update(
    rating: dict, pace_error: float, efficiency_error: float, params: ScoreRatingParams = DEFAULT_SCORE_RATING_PARAMS,
) -> dict:
    """Notes d'une équipe après un match (fonction pure)."""
    return {
        "pace": rating["pace"] + params.k_pace * pace_error,
        "scoring": rating["scoring"] + params.k_scoring * efficiency_error,
    }


def league_update(
    league: dict, pace_errors: float, efficiency_errors: float, params: ScoreRatingParams = DEFAULT_SCORE_RATING_PARAMS,
) -> dict:
    """
    Niveau de la ligue après une JOURNÉE (écarts de ses matchs sommés). Pas
    après chaque match : sinon un match du soir changerait la prévision d'un
    autre match du même soir, ce qui est impossible en production (la
    synchro a lieu le matin).
    """
    return {
        "pace": league["pace"] + params.league_rate * pace_errors,
        "efficiency": league["efficiency"] + params.league_rate * efficiency_errors / 2,
    }


def regress_score_rating(rating: dict, params: ScoreRatingParams = DEFAULT_SCORE_RATING_PARAMS) -> dict:
    """Régression partielle des notes d'une équipe vers la moyenne (0), au changement de saison."""
    return {key: (1 - params.season_regression) * value for key, value in rating.items()}


def score_rating_for_target_season(
    rating: dict, last_known_season, target_season, params: ScoreRatingParams = DEFAULT_SCORE_RATING_PARAMS,
) -> dict:
    """Comme elo_for_target_season() : régression si le match tombe dans une saison postérieure."""
    if is_later_season(target_season, last_known_season):
        return regress_score_rating(rating, params)
    return rating


@dataclass
class ScoreRatingTimeline:
    """Résultat du passage séquentiel sur tout l'historique."""
    per_game: pd.DataFrame  # nba_game_id, exp_possessions, exp_total (AVANT le match)
    final_state: dict       # {team_id: {"pace", "scoring"}} après le dernier match connu
    league: dict            # niveau de la ligue après la dernière journée connue
    params: ScoreRatingParams


def compute_score_rating_timeline(
    df_games: pd.DataFrame, params: ScoreRatingParams = DEFAULT_SCORE_RATING_PARAMS,
) -> ScoreRatingTimeline:
    """
    Notes de rythme et de points match après match, dans l'ordre
    chronologique (même principe que compute_elo_timeline : la prévision
    d'un match est enregistrée AVANT de le prendre en compte ; le niveau de
    la ligue, partagé par tous les matchs, n'est mis à jour qu'en fin de
    journée). Un match sans score ou sans boxscore laisse les notes
    inchangées.
    """
    missing = [f"{side}_{c}" for side in ("home", "away") for c in BOX_SCORE_COLS if f"{side}_{c}" not in df_games]
    if missing:
        raise ValueError(f"Boxscore requis pour les ratings de score, colonnes absentes : {missing}")

    df = df_games.sort_values(["game_date", "nba_game_id"]).reset_index(drop=True)
    dates = df["game_date"].to_numpy()
    seasons = df["season"].to_numpy()
    home_ids = df["home_team_id"].to_numpy()
    away_ids = df["away_team_id"].to_numpy()
    totals = (pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")).to_numpy(dtype=float)
    possessions = game_possessions(df).to_numpy(dtype=float)

    league = dict(LEAGUE_INITIAL)
    ratings: dict = {}
    current_season = None
    current_date = None
    day_errors = [0.0, 0.0]  # écarts (possessions, efficacité) sommés sur la journée en cours
    exp_possessions = np.empty(len(df))
    exp_total = np.empty(len(df))

    for i in range(len(df)):
        if dates[i] != current_date:
            league = league_update(league, *day_errors, params)
            day_errors = [0.0, 0.0]
            current_date = dates[i]
        if seasons[i] != current_season:
            if current_season is not None:
                ratings = {team: regress_score_rating(r, params) for team, r in ratings.items()}
            current_season = seasons[i]

        home = ratings.get(home_ids[i], NEUTRAL_SCORE_RATING)
        away = ratings.get(away_ids[i], NEUTRAL_SCORE_RATING)
        exp_possessions[i], exp_total[i] = expected_game(league, home, away)

        if np.isnan(totals[i]) or not possessions[i] > 0:
            continue
        pace_error, efficiency_error = score_rating_errors(league, home, away, possessions[i], totals[i])
        ratings[home_ids[i]] = team_rating_update(home, pace_error, efficiency_error, params)
        ratings[away_ids[i]] = team_rating_update(away, pace_error, efficiency_error, params)
        day_errors[0] += pace_error
        day_errors[1] += efficiency_error
    league = league_update(league, *day_errors, params)  # dernière journée

    per_game = pd.DataFrame({
        "nba_game_id": df["nba_game_id"],
        "exp_possessions": exp_possessions,
        "exp_total": exp_total,
    })
    return ScoreRatingTimeline(per_game=per_game, final_state=ratings, league=league, params=params)


# ---------------------------------------------------------------------------
# Score prédit — partagé entre models/score_model.py (évaluation, réglage de
# sigma et de la fourchette) et predictor_service.py (score affiché)
# ---------------------------------------------------------------------------
#
# - total des deux équipes : ratings de score ci-dessus (exp_total) ;
# - écart : sigma * Φ⁻¹(p), p étant la probabilité de victoire à domicile
#   du modèle servi. Si l'écart final suit une loi normale de moyenne mu et
#   d'écart-type sigma, alors p = Φ(mu / sigma), donc mu = sigma * Φ⁻¹(p).
#   Le vainqueur du score est donc toujours le favori du modèle ;
# - fourchette : ± une demi-largeur fixe autour des points de chaque équipe,
#   qui contient le vrai score dans 80 % des cas (mesuré en walk-forward).
# Évaluation et choix : voir models/score_model.py.

PROBABILITY_CLIP = 1e-6  # évite Φ⁻¹(0) = -inf


def expected_margin(home_probability, sigma: float):
    """Écart attendu (domicile - extérieur), en points : sigma * Φ⁻¹(p)."""
    p = np.clip(home_probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return sigma * ndtri(p)


def split_score(total, margin):
    """(points domicile, points extérieur) d'un total et d'un écart."""
    return (total + margin) / 2, (total - margin) / 2


def rounded_score(home: float, away: float, home_favored: bool) -> tuple[int, int]:
    """
    Score affiché : entiers, sans égalité (impossible au basket). Arrondis
    égaux (écart attendu de moins d'un point) : le favori prend un point de
    plus. Le vainqueur du score reste celui du modèle.
    """
    h, a = int(round(home)), int(round(away))
    if h == a:
        h, a = (h + 1, a) if home_favored else (h, a + 1)
    return h, a


def score_range(points: float, half_width: float) -> tuple[int, int]:
    """Fourchette affichée autour des points (non arrondis) d'une équipe, en entiers."""
    return int(round(points - half_width)), int(round(points + half_width))


# ---------------------------------------------------------------------------
# Statistiques des joueurs — partagé entre models/player_model.py (batch :
# évaluation, modèle de minutes), team_state.py (état courant de chaque
# joueur) et predictor_service.py (projections d'un match)
# ---------------------------------------------------------------------------
#
# Projection d'un joueur pour un match, « s'il joue » (une espérance) :
# - joueurs attendus : ceux du dernier match de l'équipe dans la saison
#   (même hypothèse que lineup_delta : un absent du dernier match reste
#   absent) ; aucun au premier match de la saison (effectifs inconnus) ;
# - minutes : modèle appris (XGBoost, models/player_model.py) sur
#   MINUTES_FEATURES ;
# - autres stats : minutes × taux par minute attendu pour CE match. Taux du
#   joueur = moyenne pondérée de ses matchs (demi-vie propre à chaque stat),
#   ramenée vers le taux de la ligue de la saison précédente par
#   `prior_minutes` minutes fictives (recrues, petits temps de jeu) ; × rythme
#   du match (possessions attendues par les ratings de score / possessions
#   moyennes des matchs du joueur) ; pour rebonds, passes, interceptions et
#   contres, × (ce que l'adversaire concède / moyenne de la ligue) **
#   opponent_exponent.
#
# Mesures et choix (2026-10-21, walk-forward 2021-22 -> 2025-26) : voir
# models/player_model.py et CLAUDE.md, « Statistiques des joueurs ».

PLAYER_STATS = ("min", "pts", "reb", "ast", "stl", "blk")
COUNT_STATS = PLAYER_STATS[1:]
OPPONENT_STATS = ("reb", "ast", "stl", "blk")


@dataclass(frozen=True)
class PlayerProjectionParams:
    """Paramètres des projections des joueurs (voir models/tune_player_projections.py)."""
    minutes_halflife: float       # demi-vie (en matchs) de la moyenne des minutes
    minutes_long_halflife: float  # idem, tendance longue (entrée du modèle de minutes)
    rates: dict                   # {stat: {"halflife": matchs, "prior_minutes": minutes fictives}}
    opponent_halflife: float      # demi-vie de ce que concède chaque équipe
    opponent_exponent: float      # poids de l'adversaire (0 : ignoré, 1 : proportionnel)

    def as_dict(self) -> dict:
        return asdict(self)


# Réglés le 2026-10-21 sur les saisons 2021-22 -> 2023-24 (erreur
# quadratique, rodage 2016-17 -> 2020-21) : minutes sur les minutes, taux sur
# « minutes réelles × taux ». Les stats rares (interceptions, contres)
# demandent une fenêtre plus longue et plus de lissage. Comme pour l'Elo, les
# changer impose de réentraîner puis de régénérer team_state.json (deploy
# puis sync).
DEFAULT_PLAYER_PARAMS = PlayerProjectionParams(
    minutes_halflife=2.0,
    minutes_long_halflife=10.0,
    rates={
        "pts": {"halflife": 10.0, "prior_minutes": 25.0},
        "reb": {"halflife": 20.0, "prior_minutes": 25.0},
        "ast": {"halflife": 10.0, "prior_minutes": 25.0},
        "stl": {"halflife": 40.0, "prior_minutes": 200.0},
        "blk": {"halflife": 40.0, "prior_minutes": 100.0},
    },
    opponent_halflife=20.0,
    opponent_exponent=0.5,
)


def rate_halflives(params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS) -> list[float]:
    return sorted({r["halflife"] for r in params.rates.values()})


def player_state_cols(params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS) -> list[str]:
    """Colonnes de l'état d'un joueur (player_form_timeline), stockées dans team_state.json."""
    return (
        ["minutes", "minutes_long", "last_minutes", "season_minutes", "last10_minutes", "games"]
        + [f"ew_min_{h:g}" for h in rate_halflives(params)]
        + [f"ew_poss_{h:g}" for h in rate_halflives(params)]
        + [f"ew_{s}" for s in COUNT_STATS]
    )


# Entrées du modèle de minutes, dans cet ordre (player_projection_features).
MINUTES_FEATURES = (
    ["minutes", "minutes_long", "last_minutes", "season_minutes", "last10_minutes", "games",
     "team_games", "rest_days", "is_home", "exp_possessions", "exp_total", "elo_win_probability",
     "roster_size", "roster_minutes"]
    + [f"rate_{s}" for s in COUNT_STATS]
    + [f"formula_{s}" for s in COUNT_STATS]
    + [f"opponent_factor_{s}" for s in OPPONENT_STATS]
)


def player_form_timeline(
    player_games: pd.DataFrame, possessions: pd.Series, params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS,
) -> pd.DataFrame:
    """
    État de chaque joueur APRÈS chacune de ses apparitions (toutes équipes,
    saisons précédentes comprises) : c'est son état pour tout match d'une
    date ultérieure. `possessions` : possessions de chaque match
    (game_possessions), indexées par nba_game_id. Moyennes pondérées
    exponentielles (ewm de pandas, demi-vie en matchs) ; ew_<stat> et
    ew_min_<h> sont des SOMMES pondérées (numérateur et dénominateur d'un
    taux par minute).

    Colonnes : player_id, team_id, nba_game_id, game_date, season, player_name
    (si présente) + player_state_cols(params).
    """
    p = player_games.sort_values(["player_id", "game_date", "nba_game_id"]).reset_index(drop=True)
    p["reb"] = p["oreb"] + p["dreb"]
    p["poss"] = p["nba_game_id"].map(possessions)
    by_player = p.groupby("player_id")

    def ew(col: str, halflife: float, how: str) -> pd.Series:
        window = by_player[col].ewm(halflife=halflife)
        values = window.mean() if how == "mean" else window.sum()
        return values.reset_index(level=0, drop=True).sort_index()

    keep = [c for c in ("player_id", "team_id", "nba_game_id", "game_date", "season", "player_name") if c in p]
    out = p[keep].copy()
    out["minutes"] = ew("min", params.minutes_halflife, "mean")
    out["minutes_long"] = ew("min", params.minutes_long_halflife, "mean")
    out["last_minutes"] = p["min"].astype(float)
    by_season = p.groupby(["player_id", "season"])
    out["season_minutes"] = by_season["min"].cumsum() / (by_season.cumcount() + 1)
    out["last10_minutes"] = by_player["min"].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    out["games"] = (by_player.cumcount() + 1).astype(float)
    for h in rate_halflives(params):
        out[f"ew_min_{h:g}"] = ew("min", h, "sum")
        out[f"ew_poss_{h:g}"] = ew("poss", h, "mean")
    for s in COUNT_STATS:
        out[f"ew_{s}"] = ew(s, params.rates[s]["halflife"], "sum")
    return out


def league_player_rates(player_games: pd.DataFrame) -> dict:
    """Taux par minute de la ligue, par saison : {saison: {stat: total / minutes}}."""
    totals = player_games.assign(reb=player_games["oreb"] + player_games["dreb"]).groupby("season")[
        ["min", *COUNT_STATS]
    ].sum()
    return {
        str(season): {s: float(row[s] / row["min"]) for s in COUNT_STATS}
        for season, row in totals.iterrows()
    }


def team_allowed_long(df_games: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par équipe et par match : stats (OPPONENT_STATS) de son ADVERSAIRE dans ce match."""
    frames = []
    for side, other in (("home", "away"), ("away", "home")):
        frame = df_games[["nba_game_id", "game_date", "season", f"{side}_team_id"]].rename(
            columns={f"{side}_team_id": "team_id"}
        )
        for s in OPPONENT_STATS:
            frame[f"allowed_{s}"] = pd.to_numeric(df_games[f"{other}_{s}"], errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["team_id", "game_date", "nba_game_id"]).reset_index(drop=True)


def league_allowed(df_games: pd.DataFrame) -> dict:
    """Moyenne par équipe et par match de chaque stat concédée, par saison : {saison: {stat: valeur}}."""
    means = team_allowed_long(df_games).groupby("season")[[f"allowed_{s}" for s in OPPONENT_STATS]].mean()
    return {str(season): {s: float(row[f"allowed_{s}"]) for s in OPPONENT_STATS} for season, row in means.iterrows()}


def opponent_allowed_timeline(df_games: pd.DataFrame, params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS) -> pd.DataFrame:
    """
    Ce que chaque équipe concède (moyenne pondérée, demi-vie
    opponent_halflife, toutes saisons), APRÈS chacun de ses matchs.
    Colonnes : team_id, nba_game_id, game_date, opponent_allowed_<stat>.
    """
    long_df = team_allowed_long(df_games)
    out = long_df[["team_id", "nba_game_id", "game_date"]].copy()
    by_team = long_df.groupby("team_id")
    for s in OPPONENT_STATS:
        out[f"opponent_allowed_{s}"] = (
            by_team[f"allowed_{s}"].ewm(halflife=params.opponent_halflife).mean()
            .reset_index(level=0, drop=True).sort_index()
        )
    return out


def previous_season_values(by_season: dict, season) -> dict:
    """Valeurs de la dernière saison ANTÉRIEURE à `season` (la plus ancienne connue à défaut : début de l'historique)."""
    earlier = [s for s in by_season if str(s) < str(season)]
    return by_season[max(earlier)] if earlier else by_season[min(by_season)]


def player_projection_features(df: pd.DataFrame, params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS) -> pd.DataFrame:
    """
    Complète des lignes « joueur attendu pour un match » (une ou plusieurs
    équipes, clés team_id + nba_game_id) avec les taux attendus et les
    entrées du modèle de minutes. Colonnes requises : player_state_cols()
    (état d'avant-match), team_games, rest_days, is_home, exp_possessions,
    exp_total, elo_win_probability, opponent_allowed_<stat>, et les valeurs
    de la ligue league_rate_<stat> / league_allowed_<stat> (saison
    précédente). Une valeur manquante (rythme ou adversaire inconnu) laisse
    le taux inchangé ; le modèle de minutes accepte les NaN.

    Ajoute rate_<stat> (taux du joueur), per_min_<stat> (taux attendu pour
    ce match), formula_<stat> (minutes moyennes × per_min), opponent_factor_<stat>,
    roster_size, roster_minutes.
    """
    out = df.copy()
    for s in OPPONENT_STATS:
        out[f"opponent_factor_{s}"] = (out[f"opponent_allowed_{s}"] / out[f"league_allowed_{s}"]).fillna(1.0)
    for s in COUNT_STATS:
        halflife, prior_minutes = params.rates[s]["halflife"], params.rates[s]["prior_minutes"]
        rate = (out[f"ew_{s}"] + prior_minutes * out[f"league_rate_{s}"]) / (out[f"ew_min_{halflife:g}"] + prior_minutes)
        per_min = rate * (out["exp_possessions"] / out[f"ew_poss_{halflife:g}"]).fillna(1.0)
        if s in OPPONENT_STATS:
            per_min = per_min * out[f"opponent_factor_{s}"] ** params.opponent_exponent
        out[f"rate_{s}"] = rate
        out[f"per_min_{s}"] = per_min
        out[f"formula_{s}"] = out["minutes"] * per_min
    roster = out.groupby(["team_id", "nba_game_id"])["minutes"]
    out["roster_size"] = roster.transform("size").astype(float)
    out["roster_minutes"] = roster.transform("sum")
    return out


def projected_stats(features: pd.DataFrame, minutes) -> pd.DataFrame:
    """Stats projetées (PLAYER_STATS) : minutes du modèle (plancher 0) × taux attendus du match."""
    out = pd.DataFrame({"min": np.maximum(np.asarray(minutes, dtype=float), 0.0)}, index=features.index)
    for s in COUNT_STATS:
        out[s] = out["min"] * features[f"per_min_{s}"]
    return out


def stat_ranges(table: dict, values) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourchettes (bornes entières incluses) d'une stat pour des valeurs
    projetées : tranche de projection (table["edges"], bornes croissantes),
    puis bornes de la tranche (table["bounds"], une paire par tranche,
    models/player_model.fit_interval_table).
    """
    idx = np.searchsorted(np.asarray(table["edges"], dtype=float), np.asarray(values, dtype=float), side="right")
    bounds = np.asarray(table["bounds"], dtype=float)
    return bounds[idx, 0], bounds[idx, 1]


def compute_player_rows(
    df_games: pd.DataFrame, player_games: pd.DataFrame, params: PlayerProjectionParams = DEFAULT_PLAYER_PARAMS,
) -> pd.DataFrame:
    """
    Version batch (entraînement, évaluation) : une ligne par équipe, match et
    joueur attendu (joueurs du match précédent de l'équipe dans la saison),
    avec l'état d'avant-match et player_projection_features(). Même calcul
    que predictor_service.py à partir de team_state.json, vérifié par
    check_features.py.

    `df_games` : matchs avec boxscore d'équipe, exp_possessions / exp_total
    (ratings de score) et home_elo / away_elo d'avant-match (game_features.csv
    ou feature_engineering.build_features). `player_games` : feuilles joueur
    limitées aux matchs de df_games (player_games.restrict_to_games).
    """
    possessions = pd.Series(game_possessions(df_games).to_numpy(), index=df_games["nba_game_id"].to_numpy())
    timeline = player_form_timeline(player_games, possessions, params)
    state_cols = player_state_cols(params)

    # Matchs de chaque équipe : match précédent de la saison, repos, contexte.
    cols = ["nba_game_id", "game_date", "season", "exp_possessions", "exp_total"]
    p_home = elo_expected_home(df_games["home_elo"].to_numpy(dtype=float), df_games["away_elo"].to_numpy(dtype=float), DEFAULT_ELO_PARAMS)
    sides = []
    for side, other, p in (("home", "away", p_home), ("away", "home", 1 - p_home)):
        frame = df_games[cols + [f"{side}_team_id", f"{other}_team_id"]].rename(
            columns={f"{side}_team_id": "team_id", f"{other}_team_id": "opponent_id"}
        )
        frame["is_home"] = 1.0 if side == "home" else 0.0
        frame["elo_win_probability"] = p
        sides.append(frame)
    tg = pd.concat(sides, ignore_index=True).sort_values(["team_id", "game_date", "nba_game_id"]).reset_index(drop=True)
    tg["team_games"] = tg.groupby(["team_id", "season"]).cumcount().astype(float)
    tg["previous_game_id"] = tg.groupby(["team_id", "season"])["nba_game_id"].shift(1)
    tg["rest_days"] = tg.groupby("team_id")["game_date"].diff().dt.days.clip(upper=REST_DAYS_CAP)
    tg = tg.dropna(subset=["previous_game_id"]).astype({"previous_game_id": "int64"})

    # Joueurs attendus = joueurs du match précédent de l'équipe.
    roster = player_games[["player_id", "team_id", "nba_game_id"]].rename(columns={"nba_game_id": "previous_game_id"})
    rows = tg.merge(roster, on=["team_id", "previous_game_id"])

    # État de chaque joueur après son dernier match strictement avant la date du match.
    state = timeline[["player_id", "game_date"] + state_cols].rename(columns={"game_date": "state_date"})
    rows = pd.merge_asof(
        rows.sort_values("game_date"), state.sort_values("state_date"),
        left_on="game_date", right_on="state_date", by="player_id", allow_exact_matches=False,
    ).drop(columns="state_date")

    # Ce que concède l'adversaire, après son match précédent.
    allowed = opponent_allowed_timeline(df_games, params)
    allowed_cols = [f"opponent_allowed_{s}" for s in OPPONENT_STATS]
    allowed[allowed_cols] = allowed.groupby("team_id")[allowed_cols].shift(1)
    rows = rows.merge(
        allowed.drop(columns="game_date").rename(columns={"team_id": "opponent_id"}),
        on=["opponent_id", "nba_game_id"], how="left",
    )

    # Valeurs de la ligue de la saison précédente.
    rates_by_season = league_player_rates(player_games)
    allowed_by_season = league_allowed(df_games)
    for season, idx in rows.groupby("season").groups.items():
        rates = previous_season_values(rates_by_season, season)
        conceded = previous_season_values(allowed_by_season, season)
        for s in COUNT_STATS:
            rows.loc[idx, f"league_rate_{s}"] = rates[s]
        for s in OPPONENT_STATS:
            rows.loc[idx, f"league_allowed_{s}"] = conceded[s]

    rows = rows.sort_values(["game_date", "nba_game_id", "team_id", "player_id"]).reset_index(drop=True)
    return player_projection_features(rows, params)


# ---------------------------------------------------------------------------
# Explication des prédictions — partagée entre train_model.py (importance
# globale de chaque famille) et predictor_service.py (facteurs d'un match)
# ---------------------------------------------------------------------------

# Familles de features : les contributions des colonnes home_/away_/diff_
# d'une même famille sont sommées. Ordre important : un motif plus
# spécifique (win_pct_context, win_pct_last10) doit passer avant le motif
# générique (win_pct) qu'il contient.
FACTOR_FAMILIES = [
    ("elo", "Rating Elo", ("elo",)),
    ("context", "Bilan domicile / extérieur", ("win_pct_context",)),
    # Bilan des 10 derniers matchs et Net Rating récent dans la même famille :
    # très corrélés (0,88), la régression logistique se partage leur effet
    # commun de façon instable (coefficient du bilan récent de +0,04 à -0,13
    # selon les saisons d'entraînement, en miroir de celui du Net Rating).
    # Séparés, ils donneraient des facteurs de signes opposés difficiles à lire.
    ("form", "Forme récente", ("win_pct_last10", "net_rtg_last")),
    ("season", "Bilan de la saison", ("win_pct",)),
    ("offense", "Attaque récente", ("avg_pts_scored_last",)),
    ("defense", "Défense récente", ("avg_pts_allowed_last",)),
    ("rest", "Jours de repos", ("rest_days",)),
    ("lineup", "Effectif", ("lineup_delta",)),
]


def factor_family(column: str) -> tuple[str, str]:
    """(clé, libellé) de la famille d'une colonne de feature."""
    base = column.split("_", 1)[1]  # retire home_/away_/diff_
    for key, label, patterns in FACTOR_FAMILIES:
        if any(base.startswith(p) for p in patterns):
            return key, label
    return "other", "Autres"


def linear_contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Contributions exactes de chaque feature à la sortie d'une régression
    logistique (Pipeline imputer -> scaler -> clf de train_model.make_logreg),
    en log-odds : coefficient × valeur standardisée, valeur manquante
    remplacée comme à la prédiction. Biais + somme des contributions d'un
    match = logit de sa probabilité de victoire à domicile. Une contribution
    nulle correspond à la valeur moyenne de l'entraînement.

    Retourne (contributions de forme (n_matchs, n_features), biais).
    """
    z = model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(X))
    clf = model.named_steps["clf"]
    return z * clf.coef_[0], float(clf.intercept_[0])