"""
team_state.py

Construit team_state.json : un instantané de l'état "courant" de chaque
équipe (Elo, forme récente, résultats de la saison), à partir de
games_clean.csv.

C'est le pont entre le pipeline batch (feature_engineering.py, qui calcule
l'historique complet pour l'ENTRAÎNEMENT) et l'API (predictor_service.py,
qui a besoin de l'état COURANT de chaque équipe pour PRÉDIRE un match à
venir). Sans ce fichier, predictor_service.py devrait recharger et
retraiter tout l'historique (8000+ matchs) à chaque requête.

Ce module est destiné à être appelé par nba_sync_service.py (job de synchro
quotidien, pas encore implémenté), mais est autonome et exécutable seul :

Usage :
    cd data
    python preprocessing/team_state.py
"""

import json
from pathlib import Path

import pandas as pd

import numpy as np

from features_lib import BOX_SCORE_COLS, compute_elo_timeline, elo_update

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "games_clean.csv"
OUTPUT_PATH = PROCESSED_DIR / "team_state.json"

MAX_RECENT_GAMES = 10  # suffit à rolling_mean_last_n pour last5/last10
ELO_HISTORY_LENGTH = 30  # points de la courbe d'évolution Elo affichée par le front


def has_box_score(df: pd.DataFrame) -> bool:
    return all(
        f"{side}_{col}" in df.columns
        for side in ("home", "away")
        for col in BOX_SCORE_COLS
    )


def _to_long_format(df: pd.DataFrame, with_box_score: bool) -> pd.DataFrame:
    """
    Même transformation wide -> long que feature_engineering.py (1 ligne =
    1 équipe pour 1 match), gardée volontairement séparée : ici on n'a
    besoin que des colonnes utiles à la construction de team_state, pas de
    tout le pipeline de features.
    """
    base_cols = ["nba_game_id", "game_date", "season"]
    box_cols = BOX_SCORE_COLS if with_box_score else []

    home_cols = (
        base_cols + ["home_team_id", "away_team_id", "home_score", "away_score", "home_win", "home_elo_before", "home_elo_after"]
        + [f"home_{c}" for c in box_cols]
    )
    away_cols = (
        base_cols + ["away_team_id", "home_team_id", "away_score", "home_score", "home_win", "away_elo_before", "away_elo_after"]
        + [f"away_{c}" for c in box_cols]
    )
    new_names = base_cols + ["team_id", "opponent_id", "pts_scored", "pts_allowed", "win", "elo_before", "elo_after"] + box_cols

    home = df[home_cols].copy()
    home.columns = new_names
    home["is_home"] = 1

    away = df[away_cols].copy()
    away.columns = new_names
    away["win"] = 1 - away["win"]
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    return long_df


def _with_elo_before_after(df_games: pd.DataFrame, per_game: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute à chaque match le rating Elo des deux équipes AVANT (fourni par
    compute_elo_timeline) et APRÈS le match (recalculé avec elo_update, la
    même formule). Sert uniquement à l'affichage (courbe Elo, forme récente
    dans le front) : ces colonnes ne sont pas des features du modèle.
    """
    df = df_games.merge(per_game, on="nba_game_id", how="left")
    df = df.rename(columns={"home_elo": "home_elo_before", "away_elo": "away_elo_before"})

    r_home = df["home_elo_before"].to_numpy(dtype=float)
    r_away = df["away_elo_before"].to_numpy(dtype=float)
    home_win = pd.to_numeric(df["home_win"], errors="coerce").to_numpy(dtype=float)
    home_score = pd.to_numeric(df["home_score"], errors="coerce").to_numpy(dtype=float)
    away_score = pd.to_numeric(df["away_score"], errors="coerce").to_numpy(dtype=float)

    home_after = r_home.copy()
    away_after = r_away.copy()
    for i in range(len(df)):
        # Ligne corrompue : rating inchangé, comme dans compute_elo_timeline.
        if np.isnan([r_home[i], r_away[i], home_win[i], home_score[i], away_score[i]]).any():
            continue
        home_after[i], away_after[i] = elo_update(r_home[i], r_away[i], home_win[i], home_score[i], away_score[i])

    df["home_elo_after"] = home_after
    df["away_elo_after"] = away_after
    return df


def build_team_state(df_games: pd.DataFrame) -> dict:
    """
    Construit l'état courant de chaque équipe à partir de l'historique
    complet des matchs.

    Retourne un dict prêt à être sérialisé en JSON :
    {
      "as_of": {"last_game_date": "...", "season": "...", "generated_at": "..."},
      "teams": {
        "<team_id>": {
          "team_id": int,
          "elo": float,               # état final -> features_lib.current_elo_for_prediction()
          "last_game_date": "...",    # -> features_lib.rest_days()
          "recent_games": [...],      # jusqu'à MAX_RECENT_GAMES, plus ancien -> plus récent
                                       # -> features_lib.rolling_mean_last_n() / net_rtg_from_boxscore()
                                       # (+ opponent_id, elo_before : affichage seulement)
          "elo_history": [...],       # {date, elo} après chacun des ELO_HISTORY_LENGTH derniers
                                       # matchs (affichage seulement)
          "season_results": {"all": [...], "home": [...], "away": [...]}
                                       # 0/1, saison courante uniquement -> features_lib.win_pct()
        },
        ...
      }
    }
    """
    with_box_score = has_box_score(df_games)
    timeline = compute_elo_timeline(df_games)
    long_df = _to_long_format(_with_elo_before_after(df_games, timeline.per_game), with_box_score)
    current_season = df_games.loc[df_games["game_date"].idxmax(), "season"]

    teams = {}
    for _, team_games in long_df.groupby("team_id"):
        team_games = team_games.sort_values("game_date")

        # Extraction en tableaux numpy typés AVANT la boucle (même approche
        # que compute_elo_timeline dans features_lib.py) : une valeur
        # scalaire issue de .iterrows()/.iat est typée "Scalar" générique
        # par pandas-stubs (union incluant complex, bytes...), ce qui fait
        # échouer int()/float() aux yeux d'un vérificateur de type statique
        # même quand le dtype réel est numérique. Un tableau numpy a un
        # dtype concret (int64/float64), donc plus d'ambiguïté.
        team_id_int = int(team_games["team_id"].to_numpy(dtype=int)[0])
        dates = team_games["game_date"].to_numpy()
        is_home_arr = team_games["is_home"].to_numpy(dtype=int)
        pts_scored_arr = team_games["pts_scored"].to_numpy(dtype=float)
        pts_allowed_arr = team_games["pts_allowed"].to_numpy(dtype=float)
        win_arr = team_games["win"].to_numpy(dtype=int)
        opponent_arr = team_games["opponent_id"].to_numpy(dtype=int)
        elo_before_arr = team_games["elo_before"].to_numpy(dtype=float)
        elo_after_arr = team_games["elo_after"].to_numpy(dtype=float)
        box_arrs = (
            {c: team_games[c].to_numpy(dtype=float) for c in BOX_SCORE_COLS}
            if with_box_score else {}
        )

        # --- Forme récente : les MAX_RECENT_GAMES derniers matchs, tous contextes ---
        n = len(team_games)
        start = max(0, n - MAX_RECENT_GAMES)
        recent_games = []
        for i in range(start, n):
            game = {
                "date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "is_home": int(is_home_arr[i]),
                "pts_scored": float(pts_scored_arr[i]),
                "pts_allowed": float(pts_allowed_arr[i]),
                "win": int(win_arr[i]),
                "opponent_id": int(opponent_arr[i]),
                "elo_before": round(float(elo_before_arr[i]), 1),
            }
            for c, arr in box_arrs.items():
                game[c] = float(arr[i])
            recent_games.append(game)

        # --- Résultats de la SAISON EN COURS (pour win_pct / win_pct_context) ---
        season_mask = (team_games["season"] == current_season).to_numpy()
        season_results = {
            "all": win_arr[season_mask].tolist(),
            "home": win_arr[season_mask & (is_home_arr == 1)].tolist(),
            "away": win_arr[season_mask & (is_home_arr == 0)].tolist(),
        }

        elo_history = [
            {"date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"), "elo": round(float(elo_after_arr[i]), 1)}
            for i in range(max(0, n - ELO_HISTORY_LENGTH), n)
        ]

        last_game_date = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") if n else None

        teams[str(team_id_int)] = {
            "team_id": team_id_int,
            "elo": float(timeline.final_state.get(team_id_int, 1500.0)),
            "last_game_date": last_game_date,
            "recent_games": recent_games,
            "season_results": season_results,
            "elo_history": elo_history,
        }

    return {
        "as_of": {
            "last_game_date": df_games["game_date"].max().strftime("%Y-%m-%d"),
            "season": str(current_season),
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
        },
        "teams": teams,
    }


def main():
    df_games = pd.read_csv(INPUT_PATH, parse_dates=["game_date"])
    print(f"Chargé : {len(df_games)} matchs")

    state = build_team_state(df_games)
    print(f"Saison courante détectée : {state['as_of']['season']}")
    print(f"Équipes traitées : {len(state['teams'])}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"Sauvegardé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()