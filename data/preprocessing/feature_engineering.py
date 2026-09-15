"""
feature_engineering.py

Calcule les features prédictives à partir de games_clean.csv.

Features "de base" (v1) :
- avg_pts_scored_last{5,10}  : moyenne des points marqués sur les N derniers matchs
- avg_pts_allowed_last{5,10} : moyenne des points encaissés sur les N derniers matchs
- win_pct                    : % de victoires depuis le début de la saison en cours
                               (toutes apparitions confondues, domicile + extérieur)
- win_pct_context            : % de victoires depuis le début de la saison,
                               SPÉCIFIQUE au contexte du match à prédire : pour
                               l'équipe qui reçoit, son % de victoires À DOMICILE ;
                               pour l'équipe qui se déplace, son % de victoires
                               À L'EXTÉRIEUR. Ajoutée EN PLUS de win_pct (pas en
                               remplacement) : un essai en remplacement pur a
                               légèrement dégradé l'accuracy, probablement par
                               dilution de l'échantillon (~20 matchs à domicile
                               par saison au lieu de ~40 au global).
- win_pct_last10             : % de victoires glissant sur les 10 derniers matchs,
                               tous contextes confondus (forme récente générale,
                               indépendante de la saison et du domicile/extérieur)
- rest_days                  : jours de repos depuis le match précédent

Features "avancées" (v2, ajoutées pour dépasser le plateau à ~66-67% d'accuracy) :
- pace_est                   : possessions estimées pour le match (par équipe)
                               formule standard : FGA - OREB + TOV + 0.4 * FTA
- off_rtg / def_rtg          : points marqués/encaissés pour 100 possessions estimées
- net_rtg                    : off_rtg - def_rtg
- net_rtg_last{5,10}         : moyenne glissante du Net Rating (le signal le plus
                               fort en général, bien plus informatif qu'un simple
                               nombre de points par match)

Features "Elo" (v3) :
- elo                        : rating Elo de l'équipe AVANT le match (mis à jour
                               séquentiellement match après match sur tout
                               l'historique, avantage du terrain et écart de
                               score inclus dans la formule de mise à jour).
                               Contrairement à win_pct, tient compte de la force
                               de l'adversaire.

Features différentielles (calculées une fois revenu au format 1-ligne-par-match) :
- diff_win_pct, diff_win_pct_last10, diff_net_rtg_last5, diff_net_rtg_last10,
  diff_rest_days
  -> home_X - away_X. En général plus discriminant pour un modèle qu'une paire
  de colonnes séparées (le modèle n'a pas à "apprendre" la soustraction lui-même).

RÈGLE CRITIQUE (fuite temporelle) : toutes les moyennes/ratios sont calculées
avec shift(1), donc UNIQUEMENT à partir des matchs qui précèdent le match
courant. Le modèle ne doit jamais "voir" le résultat du match qu'il essaie
de prédire, ni des matchs futurs.

Usage :
    cd data
    python preprocessing/feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features_lib import (
    BOX_SCORE_COLS,
    FORM_WINDOW,
    ROLLING_WINDOWS,
    compute_elo_timeline,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "games_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "game_features.csv"

# NOTE : ROLLING_WINDOWS, FORM_WINDOW, BOX_SCORE_COLS et toute la logique
# Elo (constantes + formule de mise à jour) vivent maintenant dans
# features_lib.py, partagé avec predictor_service.py (calcul des features
# "en direct" pour un match à venir). Ne plus dupliquer ces valeurs ici :
# toute modification de fenêtre ou de formule Elo doit se faire dans
# features_lib.py pour rester synchronisée entre entraînement et prédiction.


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    print(f"Chargé : {len(df)} matchs")
    return df


def has_box_score(df: pd.DataFrame) -> bool:
    return all(
        f"{side}_{col}" in df.columns
        for side in ("home", "away")
        for col in BOX_SCORE_COLS
    )


def to_long_format(df: pd.DataFrame, with_box_score: bool) -> pd.DataFrame:
    """
    Transforme le format 'wide' (1 ligne = 1 match, home + away côte à côte)
    en format 'long' (1 ligne = 1 équipe pour 1 match), nécessaire pour
    calculer des stats par équipe avec groupby + rolling.
    """
    base_cols = ["nba_game_id", "game_date", "season"]
    box_cols = BOX_SCORE_COLS if with_box_score else []

    home_cols = base_cols + ["home_team_id", "home_score", "away_score", "home_win"] + [f"home_{c}" for c in box_cols]
    away_cols = base_cols + ["away_team_id", "away_score", "home_score", "home_win"] + [f"away_{c}" for c in box_cols]

    new_names = base_cols + ["team_id", "pts_scored", "pts_allowed", "win"] + box_cols

    home = df[home_cols].copy()
    home.columns = new_names
    home["is_home"] = 1  # ce match était un match à domicile pour cette équipe

    away = df[away_cols].copy()
    away.columns = new_names
    away["win"] = 1 - away["win"]  # l'équipe away gagne quand home_win == 0
    away["is_home"] = 0  # ce match était un match à l'extérieur pour cette équipe

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return long_df


def add_advanced_ratings(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estime Pace / Offensive Rating / Defensive Rating / Net Rating pour
    chaque équipe sur chaque match, à partir de son propre boxscore.

    Approximation assumée (raisonnable pour un projet portfolio) : le
    nombre de possessions de l'équipe sur CE match sert de dénominateur
    à la fois pour son attaque et sa défense, faute d'avoir le boxscore
    complet des deux équipes alignées sur la même ligne à ce stade.
    """
    long_df = long_df.copy()

    poss = long_df["fga"] - long_df["oreb"] + long_df["tov"] + 0.4 * long_df["fta"]
    poss = poss.replace(0, np.nan)  # évite une division par zéro sur des lignes corrompues
    long_df["pace_est"] = poss

    long_df["off_rtg"] = 100 * long_df["pts_scored"] / poss
    long_df["def_rtg"] = 100 * long_df["pts_allowed"] / poss
    long_df["net_rtg"] = long_df["off_rtg"] - long_df["def_rtg"]

    return long_df


def add_rolling_features(long_df: pd.DataFrame, with_box_score: bool) -> pd.DataFrame:
    long_df = long_df.copy()
    grouped = long_df.groupby("team_id")

    for w in ROLLING_WINDOWS:
        long_df[f"avg_pts_scored_last{w}"] = (
            grouped["pts_scored"]
            .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        )
        long_df[f"avg_pts_allowed_last{w}"] = (
            grouped["pts_allowed"]
            .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        )
        if with_box_score:
            long_df[f"net_rtg_last{w}"] = (
                grouped["net_rtg"]
                .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            )

    # 1) Win % global : toutes les apparitions de la saison, domicile et
    # extérieur confondus. C'est la version qui donnait les meilleurs
    # résultats (66,5% / 67,4% accuracy) avant l'ajout du contexte.
    grouped_season = long_df.groupby(["team_id", "season"])
    long_df["win_pct"] = (
        grouped_season["win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # 2) Win % spécifique au contexte domicile/extérieur, AJOUTÉ EN PLUS du
    # win % global (pas en remplacement). Un premier essai en remplacement
    # pur a légèrement dégradé l'accuracy (66,1% / 65,9%) : diviser
    # l'historique par contexte réduit l'échantillon (~20 matchs à domicile
    # par saison au lieu de ~40 au global), donc plus de bruit et plus de
    # lignes manquantes en début de saison. On laisse plutôt le modèle
    # arbitrer entre le signal global (robuste) et le signal contextuel
    # (plus précis mais plus bruité) au lieu de choisir à sa place.
    long_df["win_pct_context"] = pd.NA
    for is_home_flag in (1, 0):
        mask = long_df["is_home"] == is_home_flag
        subset = long_df.loc[mask].sort_values(["team_id", "season", "game_date"])
        grouped_subset = subset.groupby(["team_id", "season"])
        win_pct_subset = grouped_subset["win"].transform(lambda s: s.shift(1).expanding().mean())
        long_df.loc[subset.index, "win_pct_context"] = win_pct_subset
    long_df["win_pct_context"] = long_df["win_pct_context"].astype(float)

    # Forme récente : win % glissant sur les FORM_WINDOW derniers matchs,
    # sans remise à zéro par saison (capte une bonne/mauvaise dynamique
    # même en tout début de saison, contrairement à win_pct).
    long_df["win_pct_last10"] = (
        grouped["win"]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    )

    # Jours de repos depuis le match précédent (toutes saisons confondues,
    # l'intersaison compte comme du repos, donc on plafonnera plus tard
    # si besoin dans train_model.py).
    long_df["rest_days"] = (
        grouped["game_date"]
        .transform(lambda s: s.diff().dt.days)
    )

    return long_df


def compute_elo_ratings(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le rating Elo pré-match de chaque équipe (home_elo/away_elo),
    pour l'entraînement.

    La logique de calcul (formule Elo, MOV, régression inter-saison) vit
    désormais dans features_lib.compute_elo_timeline(), partagée avec
    predictor_service.py. Ici on ne garde que l'adaptation au format
    attendu par le reste du pipeline batch (1 ligne par match).

    Fuite temporelle : compute_elo_timeline() enregistre le rating de
    chaque équipe AVANT le match, donc le modèle ne voit jamais le
    résultat du match courant à travers son propre Elo.
    """
    timeline = compute_elo_timeline(df_games)
    return timeline.per_game


def back_to_wide(df_games: pd.DataFrame, long_df: pd.DataFrame, with_box_score: bool) -> pd.DataFrame:
    """Rejoint les features calculées sur le format 1-ligne-par-match."""
    feature_cols = ["nba_game_id", "team_id", "win_pct", "win_pct_context", "win_pct_last10", "rest_days"]
    for w in ROLLING_WINDOWS:
        feature_cols += [f"avg_pts_scored_last{w}", f"avg_pts_allowed_last{w}"]
        if with_box_score:
            feature_cols.append(f"net_rtg_last{w}")

    feats = long_df[feature_cols]

    def rename_for(side: str) -> dict:
        mapping = {"team_id": f"{side}_team_id"}
        for col in feature_cols:
            if col in ("nba_game_id", "team_id"):
                continue
            mapping[col] = f"{side}_{col}"
        return mapping

    home_feats = feats.rename(columns=rename_for("home"))
    away_feats = feats.rename(columns=rename_for("away"))

    merged = df_games.merge(
        home_feats, on=["nba_game_id", "home_team_id"], how="left"
    ).merge(
        away_feats, on=["nba_game_id", "away_team_id"], how="left"
    )
    return merged


def add_diff_features(df: pd.DataFrame, with_box_score: bool) -> pd.DataFrame:
    """
    Features différentielles home - away : généralement plus discriminantes
    qu'une paire de colonnes séparées pour un modèle linéaire ou arbre.
    """
    df = df.copy()
    df["diff_win_pct"] = df["home_win_pct"] - df["away_win_pct"]
    df["diff_win_pct_context"] = df["home_win_pct_context"] - df["away_win_pct_context"]
    df["diff_win_pct_last10"] = df["home_win_pct_last10"] - df["away_win_pct_last10"]
    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]
    df["diff_elo"] = df["home_elo"] - df["away_elo"]
    for w in ROLLING_WINDOWS:
        df[f"diff_avg_pts_scored_last{w}"] = df[f"home_avg_pts_scored_last{w}"] - df[f"away_avg_pts_scored_last{w}"]
        if with_box_score:
            df[f"diff_net_rtg_last{w}"] = df[f"home_net_rtg_last{w}"] - df[f"away_net_rtg_last{w}"]
    return df


def main():
    df_games = load_clean(INPUT_PATH)
    with_box_score = has_box_score(df_games)
    if not with_box_score:
        print(
            "[INFO] Colonnes boxscore (fga/fta/oreb/tov) absentes -> "
            "Pace/Net Rating désactivés. Relance fetch_games.py + clean_data.py "
            "pour les activer."
        )

    long_df = to_long_format(df_games, with_box_score)
    if with_box_score:
        long_df = add_advanced_ratings(long_df)
    long_df = add_rolling_features(long_df, with_box_score)

    result = back_to_wide(df_games, long_df, with_box_score)

    elo_df = compute_elo_ratings(df_games)
    result = result.merge(elo_df, on="nba_game_id", how="left")

    result = add_diff_features(result, with_box_score)

    feature_cols = [c for c in result.columns if c.startswith(("home_", "away_", "diff_")) and c not in (
        "home_team_id", "away_team_id", "home_team_abbr", "away_team_abbr", "home_score", "away_score"
    )]

    print("\nValeurs manquantes par feature (normal en début de saison) :")
    print(result[feature_cols].isna().sum())

    print(f"\nTotal matchs : {len(result)}")
    print(result[["nba_game_id", "game_date"] + feature_cols[:8]].head(10))

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSauvegardé : {OUTPUT_PATH} ({len(feature_cols)} features)")


if __name__ == "__main__":
    main()