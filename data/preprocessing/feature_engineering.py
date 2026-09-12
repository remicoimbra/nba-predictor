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

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "games_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "game_features.csv"

ROLLING_WINDOWS = [5, 10]
FORM_WINDOW = 10  # fenêtre pour le win_pct "forme récente"

# --- Configuration Elo ---
# Valeurs standards façon FiveThirtyEight, adaptées à la NBA :
ELO_INITIAL = 1500.0
ELO_K = 20.0              # vitesse d'ajustement du rating après un match
ELO_HOME_ADVANTAGE = 100  # bonus (en points d'Elo) accordé à l'équipe qui reçoit
ELO_SEASON_REGRESSION = 0.25  # part de régression vers la moyenne entre 2 saisons

# Colonnes de boxscore nécessaires pour estimer Pace / Net Rating.
# Si absentes (ancien games_clean.csv généré avant la mise à jour de
# fetch_games.py), on désactive simplement les features avancées.
BOX_SCORE_COLS = ["fga", "fta", "oreb", "tov"]


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
    Calcule un rating Elo par équipe, mis à jour match après match dans
    l'ordre chronologique global (contrairement aux autres features, qui
    se calculent équipe par équipe indépendamment, l'Elo doit être traité
    de façon strictement séquentielle car un match met à jour les DEUX
    équipes en même temps).

    Contrairement à win_pct, l'Elo tient compte de la force de
    l'adversaire : battre une équipe forte fait gagner plus de points
    qu'battre une équipe faible, et l'inverse pour une défaite.

    Fuite temporelle : on enregistre le rating de chaque équipe AVANT le
    match (home_elo/away_elo), puis on met à jour APRÈS avoir enregistré,
    donc le modèle ne voit jamais le résultat du match courant à travers
    son propre Elo.

    Rating initial : ELO_INITIAL pour toute nouvelle équipe.
    Mise à jour : formule Elo classique + multiplicateur d'écart de score
    (MOV, façon FiveThirtyEight) : une victoire large ajuste plus le
    rating qu'une victoire d'un point, mais l'effet est amorti quand
    l'écart de rating pré-match était déjà favorable (une grosse équipe
    qui écrase une petite n'est pas "surprenante").
    Entre 2 saisons : régression partielle vers la moyenne (ELO_SEASON_REGRESSION),
    pour refléter les mouvements d'effectif l'été sans perdre toute la
    mémoire de la saison précédente.
    """
    df = df_games.sort_values(["game_date", "nba_game_id"]).reset_index(drop=True).copy()

    # Cast explicite : on ne veut pas dépendre du dtype tel qu'il arrive de
    # games_clean.csv (objet/string possible selon la version du fichier).
    # Sans ça, `row.home_score - row.away_score` peut lever un TypeError
    # ("unsupported operand type(s) for -: 'str' and 'str'") si une des
    # deux colonnes n'est pas numérique.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")

    n_bad = df[["home_score", "away_score", "home_win"]].isna().any(axis=1).sum()
    if n_bad:
        print(f"[WARN] {n_bad} matchs avec score/résultat non numérique -> ignorés pour l'Elo (rating inchangé)")

    # On extrait les colonnes en tableaux numpy typés (float64/object) avant
    # la boucle plutôt que d'itérer avec itertuples(). itertuples() type
    # chaque valeur en "Scalar" générique (str | bytes | date | complex | ...)
    # aux yeux d'un vérificateur de type statique (Pylance/Pyright), ce qui
    # déclenche des faux positifs sur les opérations arithmétiques même
    # quand le dtype réel est numérique. Les tableaux numpy ont un dtype
    # concret (float64 ici, grâce au to_numeric plus haut), donc plus
    # d'ambiguïté de type, et c'est aussi plus rapide qu'itertuples.
    seasons = df["season"].to_numpy()
    home_ids = df["home_team_id"].to_numpy()
    away_ids = df["away_team_id"].to_numpy()
    home_scores = df["home_score"].to_numpy(dtype=float)
    away_scores = df["away_score"].to_numpy(dtype=float)
    home_wins = df["home_win"].to_numpy(dtype=float)

    elo = {}
    current_season = None
    home_elo_pre = []
    away_elo_pre = []

    for i in range(len(df)):
        season = seasons[i]
        home_id = home_ids[i]
        away_id = away_ids[i]

        if season != current_season:
            if current_season is not None:
                for team_id in elo:
                    elo[team_id] = ELO_INITIAL + (1 - ELO_SEASON_REGRESSION) * (elo[team_id] - ELO_INITIAL)
            current_season = season

        r_home = elo.get(home_id, ELO_INITIAL)
        r_away = elo.get(away_id, ELO_INITIAL)

        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        home_score = home_scores[i]
        away_score = away_scores[i]
        home_win = home_wins[i]

        if np.isnan(home_score) or np.isnan(away_score) or np.isnan(home_win):
            # Ligne corrompue : on garde le rating pré-match tel quel (déjà
            # enregistré ci-dessus) mais on ne met à jour ni home_team_id
            # ni away_team_id, faute de résultat exploitable.
            continue

        # Probabilité de victoire attendue pour l'équipe à domicile (avantage du terrain inclus).
        expected_home = 1 / (1 + 10 ** (-((r_home + ELO_HOME_ADVANTAGE) - r_away) / 400))
        actual_home = 1.0 if home_win == 1 else 0.0

        margin = abs(home_score - away_score)
        elo_diff_for_winner = (r_home + ELO_HOME_ADVANTAGE - r_away) if actual_home == 1.0 else (r_away - (r_home + ELO_HOME_ADVANTAGE))
        mov_multiplier = ((margin + 3) ** 0.8) / (7.5 + 0.006 * max(elo_diff_for_winner, 0))

        delta = ELO_K * mov_multiplier * (actual_home - expected_home)
        elo[home_id] = r_home + delta
        elo[away_id] = r_away - delta

    df = df.copy()
    df["home_elo"] = home_elo_pre
    df["away_elo"] = away_elo_pre
    return df[["nba_game_id", "home_elo", "away_elo"]]


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