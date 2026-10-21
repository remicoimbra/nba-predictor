"""
check_features.py

Contrôles automatiques du pipeline de features, à relancer après toute
modification de features_lib.py, feature_engineering.py, team_state.py ou
predictor_service.py :

Les contrôles portent sur TOUTES les features calculées
(train_model.candidate_feature_cols : celles que l'API affiche sur la fiche
match), pas seulement sur celles du modèle (feature_columns.json), et sur
les colonnes du score prédit (SCORE_COLS : ratings de score d'avant-match).

0. game_features.csv est à jour : identique aux features recalculées à
   partir de games_clean.csv avec le code actuel (sinon train_model.py
   s'entraîne sur d'anciennes features).
1. Cohérence entraînement / prédiction : pour un échantillon de matchs de
   la saison la plus récente, on reconstruit team_state.json à partir des
   seuls matchs antérieurs, puis on calcule les features comme l'API
   (PredictorService.build_feature_row). Elles doivent être identiques à
   celles du pipeline d'entraînement. Les 10 premiers matchs de la saison
   sont toujours inclus : ils passent par le changement de saison
   (régression Elo, bilans remis à zéro). Pour ces mêmes matchs,
   l'explication servie doit être exacte : biais + somme des facteurs =
   logit de la probabilité (à l'arrondi de l'API près). Le total attendu de
   l'API doit être celui du pipeline (exp_total), et le score et les
   fourchettes servis ceux des formules de features_lib appliquées aux
   features d'entraînement, avec le vainqueur prédit en tête.
2. Absence de fuite temporelle : on inverse le résultat d'un match (scores
   et boxscores échangés) et on change son total (+10 points pour chaque
   équipe : l'inversion seule laisse le total intact, donc les ratings de
   score), puis on recalcule toutes les features. Celles de ce match et des
   matchs antérieurs ne doivent pas bouger ; celles des matchs suivants des
   deux équipes, si, total attendu compris.
3. Absence de fuite par l'effectif : on retire de la feuille de match le
   meilleur joueur de l'équipe à domicile (comme s'il avait été absent).
   Aucune feature de ce match ni des matchs antérieurs ne doit bouger ; le
   lineup_delta du match suivant de l'équipe, si.
4. Absence de fuite dans les statistiques des joueurs : on gonfle la ligne
   du joueur le plus utilisé de l'équipe à domicile dans un match (minutes,
   points, rebonds...). Les entrées des projections (PLAYER_COLS) de ce
   match et des matchs antérieurs ne doivent pas bouger ; celles de ce
   joueur au match suivant de son équipe, si.

Les contrôles 1 et 2 portent aussi sur les projections des joueurs : mêmes
joueurs attendus, mêmes entrées (PLAYER_COLS) et mêmes stats et fourchettes
servies côté API que côté batch (features_lib.compute_player_rows), premiers
matchs de la saison compris (aucun joueur attendu) ; inverser un résultat
ne change aucune entrée antérieure et change celles des matchs suivants.

Usage :
    cd data
    python preprocessing/check_features.py
Code de sortie 1 si un contrôle échoue.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_features
from features_lib import (
    COUNT_STATS,
    MINUTES_FEATURES,
    compute_player_rows,
    expected_margin,
    game_score,
    rounded_score,
    score_range,
    split_score,
)
from player_games import load_player_games, restrict_to_games
from team_state import build_team_state

DATA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_DIR.parent / "api"))
sys.path.insert(0, str(DATA_DIR / "models"))
from app.services.predictor_service import FEATURE_COLUMNS_PATH, PredictorService  # noqa: E402
from train_model import candidate_feature_cols  # noqa: E402

CLEAN_PATH = DATA_DIR / "processed" / "games_clean.csv"
FEATURES_PATH = DATA_DIR / "processed" / "game_features.csv"

SCORE_COLS = ["exp_possessions", "exp_total"]  # colonnes du score prédit, pas des features du modèle
# Entrées des projections des joueurs : modèle de minutes + taux attendus du match.
PLAYER_COLS = list(dict.fromkeys(MINUTES_FEATURES + [f"per_min_{s}" for s in COUNT_STATS]))
PLAYER_KEYS = ["team_id", "nba_game_id", "player_id"]
INFLATED = {"min": 10, "pts": 15, "oreb": 3, "dreb": 3, "ast": 4, "stl": 2, "blk": 2}  # contrôle 4
TOLERANCE = 1e-9
# Facteurs, biais et probabilité sont arrondis à 4 décimales par l'API.
EXPLANATION_TOLERANCE = 1e-3
SEASON_START_GAMES = 10  # premiers matchs de la saison, toujours contrôlés
RANDOM_GAMES = 40
LEAK_GAMES = 5
MIN_SEASON_GAMES = 200  # saison contrôlée : la plus récente avec au moins autant de matchs


def same_value(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return abs(float(a) - float(b)) <= TOLERANCE


def differing_columns(row_a, row_b, columns) -> list[str]:
    return [c for c in columns if not same_value(row_a[c], row_b[c])]


def check_features_file_up_to_date(batch: pd.DataFrame, columns: list[str], model_columns: list[str]) -> bool:
    saved = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"]).set_index("nba_game_id")
    fresh = batch.set_index("nba_game_id")
    if set(saved.index) != set(fresh.index):
        print(f"  ÉCHEC : {len(saved)} matchs dans game_features.csv, {len(fresh)} recalculés")
        return False
    missing = [c for c in columns if c not in saved.columns]
    if missing:
        print(f"  ÉCHEC : colonnes absentes de game_features.csv : {missing}")
        return False
    unknown = [c for c in model_columns if c not in columns]
    if unknown:
        print(f"  ÉCHEC : colonnes de feature_columns.json inconnues du pipeline : {unknown} (relancer train_model.py)")
        return False
    a = saved.loc[fresh.index, columns].to_numpy(dtype=float)
    b = fresh[columns].to_numpy(dtype=float)
    mismatch = ~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b)))
    if mismatch.any():
        cols = [columns[j] for j in np.where(mismatch.any(axis=0))[0]]
        print(f"  ÉCHEC : {int(mismatch.any(axis=1).sum())} matchs diffèrent ({cols[:5]}...) : relancer feature_engineering.py")
        return False
    print(f"  OK : {len(fresh)} matchs, {len(columns)} colonnes identiques (dont les {len(model_columns)} features du modèle)")
    return True


def batch_score(service: PredictorService, batch_row: pd.Series) -> tuple:
    """Score et fourchettes attendus d'après les features d'entraînement (modèle et paramètres du service)."""
    model_cols = service._feature_columns
    p = float(service._model.predict_proba(batch_row[model_cols].to_frame().T.astype(float))[0, 1])
    margin = float(expected_margin(p, service._score_model["sigma"]))
    home, away = split_score(batch_row["exp_total"], margin)
    half_width = service._score_model["interval"]["half_width"]
    return rounded_score(home, away, p >= 0.5), list(score_range(home, half_width)), list(score_range(away, half_width))


def player_batch_rows(batch: pd.DataFrame, clean: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Entrées des projections des joueurs côté entraînement (batch = build_features(clean, players))."""
    return compute_player_rows(batch, restrict_to_games(players, clean))


def projections_by_player(projections: list) -> dict:
    return {p["player_id"]: p["stats"] for p in projections}


def player_gaps(service: PredictorService, live: pd.DataFrame, batch_rows: pd.DataFrame, served: list) -> str | None:
    """
    Écart entre les joueurs attendus d'une équipe côté API (`live`,
    PredictorService.player_inputs) et côté batch : mêmes joueurs, mêmes
    entrées, et stats/fourchettes servies (`served`) égales à celles
    calculées à partir des entrées du batch. None si tout est identique.
    """
    live_ids = set(live["player_id"].astype(int)) if not live.empty else set()
    batch_ids = set(batch_rows["player_id"].astype(int))
    if live_ids != batch_ids:
        return f"joueurs attendus différents ({len(live_ids)} API, {len(batch_ids)} batch)"
    if not live_ids:
        return None if not served else "projections servies sans joueur attendu"
    a = live.set_index(live["player_id"].astype(int)).sort_index()
    b = batch_rows.set_index(batch_rows["player_id"].astype(int)).sort_index()
    diff = [c for c in PLAYER_COLS if any(not same_value(x, y) for x, y in zip(a[c], b[c]))]
    if diff:
        return f"entrées différentes : {diff[:5]}"
    expected = projections_by_player(service.player_projections(b.assign(name="")))
    if projections_by_player(served) != expected:
        return "stats ou fourchettes servies différentes du calcul batch"
    return None


def check_train_live_consistency(
    clean: pd.DataFrame, players: pd.DataFrame, batch: pd.DataFrame, columns: list[str], player_batch: pd.DataFrame,
) -> bool:
    counts = clean["season"].value_counts()
    season = max(s for s, n in counts.items() if n >= MIN_SEASON_GAMES)
    games = clean[clean["season"] == season].sort_values(["game_date", "nba_game_id"])
    rest = games.iloc[SEASON_START_GAMES:]
    sample = pd.concat([
        games.iloc[:SEASON_START_GAMES],
        rest.sample(n=min(RANDOM_GAMES, len(rest)), random_state=0),
    ])
    batch_rows = batch.set_index("nba_game_id")

    failures = []
    total_failures = []
    score_failures = []
    player_failures = []
    player_sides = 0  # équipes avec au moins un joueur attendu
    worst_gap = 0.0  # écart max |biais + somme des facteurs - logit(p)|
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "team_state.json"
        service = None
        for game_date, day_games in sample.groupby("game_date"):
            # Feuilles joueur complètes : build_team_state ne garde que celles
            # des matchs qu'on lui passe.
            state = build_team_state(clean[clean["game_date"] < game_date], players)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
            if service is None:
                service = PredictorService(team_state_path=state_path)
            else:
                service.reload_team_state()

            for game in day_games.itertuples():
                teams = (int(game.home_team_id), int(game.away_team_id))
                as_of = game_date.strftime("%Y-%m-%d")
                live = service.build_feature_row(*teams, as_of_date=as_of)
                feature_cols = [c for c in columns if c not in SCORE_COLS]
                absent = [c for c in feature_cols if c not in live]
                diff = absent or differing_columns(live, batch_rows.loc[game.nba_game_id], feature_cols)
                if diff:
                    failures.append((game.nba_game_id, game_date.date(), diff))
                total = service.expected_total(*teams, as_of_date=as_of)
                if not same_value(total, batch_rows.loc[game.nba_game_id, "exp_total"]):
                    total_failures.append((game.nba_game_id, total, batch_rows.loc[game.nba_game_id, "exp_total"]))

                pred = service.predict(*teams, as_of_date=as_of)
                p = pred.home_win_probability
                explained = pred.base_value + sum(f["contribution"] for f in pred.factors)
                worst_gap = max(worst_gap, abs(explained - math.log(p / (1 - p))))

                served = (
                    (pred.predicted_home_score, pred.predicted_away_score),
                    pred.predicted_home_score_range,
                    pred.predicted_away_score_range,
                )
                home_ahead = served[0][0] > served[0][1]
                if served != batch_score(service, batch_rows.loc[game.nba_game_id]) or home_ahead != (pred.predicted_winner == "home"):
                    score_failures.append((game.nba_game_id, served, pred.predicted_winner))

                inputs = service.player_inputs(*teams, as_of_date=as_of)
                game_rows = player_batch[player_batch["nba_game_id"] == game.nba_game_id]
                for side, team_id in zip(("home", "away"), teams):
                    gap = player_gaps(service, inputs[side], game_rows[game_rows["team_id"] == team_id], pred.players[side])
                    player_sides += bool(pred.players[side])
                    if gap:
                        player_failures.append((game.nba_game_id, side, gap))

    ok = True
    if failures:
        print(f"  ÉCHEC : {len(failures)} matchs sur {len(sample)} diffèrent entre entraînement et API")
        for game_id, day, cols in failures[:5]:
            print(f"    {game_id} ({day}) : {cols}")
        ok = False
    else:
        print(f"  OK : {len(sample)} matchs de {season} ({sample['game_date'].nunique()} journées), features identiques à {TOLERANCE}")
    if worst_gap > EXPLANATION_TOLERANCE:
        print(f"  ÉCHEC : biais + facteurs s'écartent du logit de la probabilité servie (jusqu'à {worst_gap:.4f})")
        ok = False
    else:
        print(f"  OK : biais + facteurs = logit(p) pour ces matchs (écart max {worst_gap:.1e}, arrondis de l'API)")
    if total_failures:
        print(f"  ÉCHEC : total attendu de l'API différent de exp_total : {total_failures[:5]}")
        ok = False
    else:
        print(f"  OK : total attendu de l'API = exp_total du pipeline à {TOLERANCE}")
    if score_failures:
        print(f"  ÉCHEC : score ou fourchettes servis différents du calcul d'entraînement, ou vainqueur : {score_failures[:5]}")
        ok = False
    else:
        print("  OK : score et fourchettes servis = formules de features_lib sur les features d'entraînement, vainqueur prédit en tête")
    if player_failures:
        print(f"  ÉCHEC : projections des joueurs différentes entre API et batch : {player_failures[:5]}")
        ok = False
    else:
        print(
            f"  OK : projections des joueurs identiques API / batch ({player_sides} équipes avec joueurs attendus, "
            f"{2 * len(sample) - player_sides} sans : premiers matchs de saison)"
        )
    return ok


def flipped_result(clean: pd.DataFrame, game_id) -> pd.DataFrame:
    """
    Copie de l'historique où le match `game_id` a le résultat inverse (stats
    domicile/extérieur échangées) et un autre total (+10 points par équipe).
    """
    modified = clean.copy()
    row = modified["nba_game_id"] == game_id
    for home_col in clean.columns:
        away_col = "away_" + home_col.removeprefix("home_")
        if not home_col.startswith("home_") or home_col in ("home_team_id", "home_team_abbr", "home_win"):
            continue
        if away_col in clean.columns:
            modified.loc[row, [home_col, away_col]] = clean.loc[row, [away_col, home_col]].to_numpy()
    modified.loc[row, "home_win"] = 1 - clean.loc[row, "home_win"]
    modified.loc[row, ["home_score", "away_score"]] += 10
    return modified


def leak_candidates(clean: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """LEAK_GAMES matchs d'une saison du milieu de l'historique, au moins 30 jours avant sa fin."""
    seasons = sorted(clean["season"].unique())
    season = seasons[len(seasons) // 2]
    games = clean[clean["season"] == season]
    candidates = games[games["game_date"] < games["game_date"].max() - pd.Timedelta(days=30)]
    return season, candidates.sample(n=LEAK_GAMES, random_state=1)


def changed(base: pd.DataFrame, feats: pd.DataFrame, index, columns: list[str]) -> np.ndarray:
    """Masque (matchs × colonnes) des features qui diffèrent entre les deux calculs."""
    a = base.loc[index, columns].to_numpy(dtype=float)
    b = feats.loc[index, columns].to_numpy(dtype=float)
    return ~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b)))


def player_changes(base_rows: pd.DataFrame, rows: pd.DataFrame, game_date) -> tuple[list[str], bool]:
    """
    (entrées modifiées pour les joueurs des matchs jusqu'à `game_date` inclus,
    au moins une entrée modifiée après) entre deux calculs de compute_player_rows.
    """
    a = base_rows.set_index(PLAYER_KEYS)
    b = rows.set_index(PLAYER_KEYS)
    before_a, before_b = a.index[a["game_date"] <= game_date], b.index[b["game_date"] <= game_date]
    if set(before_a) != set(before_b):
        return ["joueurs attendus"], False
    leaked = changed(a, b, before_a, PLAYER_COLS).any(axis=0)
    later = a.index[a["game_date"] > game_date].intersection(b.index)
    return [PLAYER_COLS[j] for j in np.where(leaked)[0]], bool(changed(a, b, later, PLAYER_COLS).any())


def check_no_leakage(
    clean: pd.DataFrame, players: pd.DataFrame, batch: pd.DataFrame, columns: list[str], player_batch: pd.DataFrame,
) -> bool:
    season, picked = leak_candidates(clean)
    base = batch.set_index("nba_game_id")

    ok = True
    for game in picked.itertuples():
        flipped = flipped_result(clean, game.nba_game_id)
        feats = build_features(flipped, players)
        player_leaked, player_propagated = player_changes(player_batch, player_batch_rows(feats, flipped, players), game.game_date)
        feats = feats.set_index("nba_game_id")
        dates = feats["game_date"]
        teams = (game.home_team_id, game.away_team_id)

        before = feats.index[dates <= game.game_date]  # le match lui-même, les matchs antérieurs et ceux du même soir
        leaked = changed(base, feats, before, columns)
        later = feats.index[(dates > game.game_date) & (feats["home_team_id"].isin(teams) | feats["away_team_id"].isin(teams))]
        propagated = changed(base, feats, later, columns).any(axis=0)

        if leaked.any():
            cols = [columns[j] for j in np.where(leaked.any(axis=0))[0]]
            print(f"  ÉCHEC : modifier {game.nba_game_id} change les features de matchs antérieurs ou de lui-même : {cols}")
            ok = False
        elif not propagated[columns.index("exp_total")]:
            print(f"  ÉCHEC : modifier {game.nba_game_id} ne change pas le total attendu des matchs suivants (contrôle inopérant ?)")
            ok = False
        if player_leaked:
            print(f"  ÉCHEC : modifier {game.nba_game_id} change les entrées des joueurs de matchs antérieurs ou de lui-même : {player_leaked}")
            ok = False
        elif not player_propagated:
            print(f"  ÉCHEC : modifier {game.nba_game_id} ne change les entrées d'aucun joueur des matchs suivants")
            ok = False
    if ok:
        print(
            f"  OK : {LEAK_GAMES} matchs de {season} inversés (total modifié), aucune feature ni entrée de joueur "
            "antérieure modifiée, matchs suivants modifiés"
        )
    return ok


def check_no_lineup_leakage(clean: pd.DataFrame, players: pd.DataFrame, batch: pd.DataFrame, columns: list[str]) -> bool:
    season, picked = leak_candidates(clean)
    base = batch.set_index("nba_game_id")

    ok = True
    for game in picked.itertuples():
        team = game.home_team_id
        roster = players[(players["nba_game_id"] == game.nba_game_id) & (players["team_id"] == team)]
        best = roster.loc[game_score(roster).idxmax(), "player_id"]
        removed = players.drop(roster.index[roster["player_id"] == best])
        feats = build_features(clean, removed).set_index("nba_game_id")
        dates = feats["game_date"]

        leaked = changed(base, feats, feats.index[dates <= game.game_date], columns)
        team_later = feats[(dates > game.game_date) & ((feats["home_team_id"] == team) | (feats["away_team_id"] == team))]
        following = team_later.sort_values("game_date").iloc[0]
        side = "home" if following["home_team_id"] == team else "away"
        propagated = not same_value(following[f"{side}_lineup_delta"], base.loc[following.name, f"{side}_lineup_delta"])

        if leaked.any():
            cols = [columns[j] for j in np.where(leaked.any(axis=0))[0]]
            print(f"  ÉCHEC : retirer le joueur {best} de {game.nba_game_id} change des features antérieures ou du match : {cols}")
            ok = False
        elif not propagated:
            print(f"  ÉCHEC : retirer le joueur {best} de {game.nba_game_id} ne change pas l'effectif du match suivant")
            ok = False
    if ok:
        print(f"  OK : {LEAK_GAMES} matchs de {season}, meilleur joueur retiré : aucune feature antérieure modifiée, effectif du match suivant modifié")
    return ok


def check_no_player_leakage(clean: pd.DataFrame, players: pd.DataFrame, batch: pd.DataFrame, player_batch: pd.DataFrame) -> bool:
    season, picked = leak_candidates(clean)
    keyed = player_batch.set_index(PLAYER_KEYS)

    ok = True
    for game in picked.itertuples():
        team = game.home_team_id
        roster = players[(players["nba_game_id"] == game.nba_game_id) & (players["team_id"] == team)]
        best = roster.loc[roster["min"].idxmax(), "player_id"]
        inflated = players.copy()
        line = (inflated["nba_game_id"] == game.nba_game_id) & (inflated["player_id"] == best)
        for col, extra in INFLATED.items():
            inflated.loc[line, col] += extra
        rows = player_batch_rows(batch, clean, inflated)
        leaked, _ = player_changes(player_batch, rows, game.game_date)

        # Ligne du joueur au match suivant de son équipe : elle doit avoir changé.
        following = keyed.loc[
            (keyed.index.get_level_values("team_id") == team) & (keyed["game_date"] > game.game_date)
        ].sort_values("game_date")
        following = following[following["game_date"] == following["game_date"].min()]
        key = (team, following.index.get_level_values("nba_game_id")[0], best)
        propagated = key in following.index and bool(
            changed(keyed, rows.set_index(PLAYER_KEYS), [key], PLAYER_COLS).any()
        )

        if leaked:
            print(f"  ÉCHEC : gonfler la ligne du joueur {best} dans {game.nba_game_id} change des entrées antérieures ou du match : {leaked}")
            ok = False
        elif not propagated:
            print(f"  ÉCHEC : gonfler la ligne du joueur {best} dans {game.nba_game_id} ne change pas ses entrées au match suivant")
            ok = False
    if ok:
        print(
            f"  OK : {LEAK_GAMES} matchs de {season}, ligne du joueur le plus utilisé gonflée : aucune entrée "
            "antérieure modifiée, ses entrées au match suivant modifiées"
        )
    return ok


def main():
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["game_date"])
    players = load_player_games()
    with open(FEATURE_COLUMNS_PATH, encoding="utf-8") as f:
        model_columns = json.load(f)
    batch = build_features(clean, players)
    columns = candidate_feature_cols(batch) + SCORE_COLS
    player_batch = player_batch_rows(batch, clean, players)

    print("0. game_features.csv à jour")
    results = [check_features_file_up_to_date(batch, columns, model_columns)]
    print("1. Cohérence entraînement / prédiction")
    results.append(check_train_live_consistency(clean, players, batch, columns, player_batch))
    print("2. Absence de fuite temporelle")
    results.append(check_no_leakage(clean, players, batch, columns, player_batch))
    print("3. Absence de fuite par l'effectif")
    results.append(check_no_lineup_leakage(clean, players, batch, columns))
    print("4. Absence de fuite dans les statistiques des joueurs")
    results.append(check_no_player_leakage(clean, players, batch, player_batch))

    if not all(results):
        sys.exit(1)
    print("Tous les contrôles passent.")


if __name__ == "__main__":
    main()
