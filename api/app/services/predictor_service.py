"""
predictor_service.py

Sert de vraies prédictions pour un match à venir (home_team_id vs
away_team_id), à partir de :
- data/processed/team_state.json  : état courant de chaque équipe (Elo,
  forme récente, résultats de la saison) — voir data/preprocessing/team_state.py
- data/preprocessing/features_lib.py : mêmes formules que celles utilisées
  à l'entraînement (Elo, rolling stats, win%) — source de vérité partagée
- data/models/saved_models/xgboost_v1.pkl : modèle entraîné
- data/models/saved_models/feature_columns.json : ordre exact des colonnes
  attendu par le modèle (généré par train_model.py)

Contrairement à feature_engineering.py (qui recalcule tout l'historique
pour l'entraînement), ce module ne calcule QUE la ligne de features du
match demandé, à partir de l'état déjà résumé dans team_state.json — pas
besoin de recharger 8000+ matchs à chaque requête.

Point d'attention (cf. CLAUDE.md) : team_state.json doit être régénéré
régulièrement (nba_sync_service.py, job quotidien) pour rester à jour.
Une prédiction faite avec un team_state.json périmé utilisera l'Elo/la
forme d'il y a plusieurs jours.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

# --- Rendre features_lib.py importable depuis l'API sans dupliquer le code ---
# api/ et data/ sont deux dossiers frères du projet : plutôt que de copier
# features_lib.py dans l'API (risque de divergence avec la version utilisée
# à l'entraînement), on ajoute data/preprocessing/ au sys.path. Si le projet
# devient un vrai package installable plus tard, remplacer par un import
# package normal (pip install -e .) supprimera ce bricolage.
#
# NOTE Pylance/Pyright : "Import could not be resolved" est attendu ici et
# sans conséquence à l'exécution. L'analyse statique lit le fichier sans
# l'exécuter, donc elle ne voit jamais le sys.path.insert() ci-dessous —
# elle ne peut pas savoir que ce module existera au runtime. C'est un faux
# positif inhérent à ce pattern (import dont le chemin est résolu
# dynamiquement), pas un bug.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
sys.path.insert(0, str(DATA_DIR / "preprocessing"))

from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    ELO_INITIAL,
    FORM_WINDOW,
    ROLLING_WINDOWS,
    elo_for_target_season,
    is_later_season,
    net_rtg_from_boxscore,
    rest_days,
    rolling_mean_last_n,
    season_for_date,
    win_pct,
)

logger = logging.getLogger(__name__)

TEAM_STATE_PATH = DATA_DIR / "processed" / "team_state.json"
MODEL_PATH = DATA_DIR / "models" / "saved_models" / "xgboost_v1.pkl"
FEATURE_COLUMNS_PATH = DATA_DIR / "models" / "saved_models" / "feature_columns.json"


class PredictorNotReadyError(RuntimeError):
    """Levée si un fichier requis (modèle, team_state, colonnes) est manquant."""


class UnknownTeamError(ValueError):
    """Levée si un des deux team_id n'existe pas dans team_state.json."""


@dataclass
class GamePrediction:
    home_team_id: int
    away_team_id: int
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str          # "home" ou "away"
    predicted_home_score: float
    predicted_away_score: float
    features_used: dict            # utile pour debug / affichage détaillé côté front


class PredictorService:
    """
    Charge le modèle et team_state.json UNE FOIS (au démarrage de l'API),
    puis sert des prédictions à la demande. team_state.json n'est relu que
    si sa date de modification a changé (reload_team_state_if_changed(),
    appelé par get_predictor_service()) : la synchro quotidienne remplace
    le fichier sur le VPS sans redémarrer l'API.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        team_state_path: Path = TEAM_STATE_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    ):
        self.model_path = model_path
        self.team_state_path = team_state_path
        self.feature_columns_path = feature_columns_path
        # Typé Any plutôt que None : le modèle chargé (XGBClassifier via
        # joblib) n'a pas de type statique connu de Pylance ici, et un
        # attribut initialisé à None sans annotation ferait inférer
        # "toujours None" au vérificateur de type, qui refuserait alors
        # tout appel de méthode dessus (ex: predict_proba) même après un
        # _load() réussi qui l'a réassigné à une vraie instance.
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._team_state: dict = {}
        self._team_state_mtime: float | None = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise PredictorNotReadyError(
                f"Modèle introuvable : {self.model_path}. "
                "Lancer data/models/train_model.py."
            )
        if not self.feature_columns_path.exists():
            raise PredictorNotReadyError(
                f"feature_columns.json introuvable : {self.feature_columns_path}. "
                "Relancer train_model.py (version à jour) pour le générer."
            )
        if not self.team_state_path.exists():
            raise PredictorNotReadyError(
                f"team_state.json introuvable : {self.team_state_path}. "
                "Lancer data/preprocessing/team_state.py."
            )

        self._model = joblib.load(self.model_path)
        with open(self.feature_columns_path, encoding="utf-8") as f:
            self._feature_columns = json.load(f)
        self.reload_team_state()

    def reload_team_state(self):
        """Relit team_state.json (après une synchro)."""
        mtime = self.team_state_path.stat().st_mtime
        with open(self.team_state_path, encoding="utf-8") as f:
            self._team_state = json.load(f)
        self._team_state_mtime = mtime

    def reload_team_state_if_changed(self):
        """
        Relit team_state.json si le fichier a été remplacé depuis le dernier
        chargement. En cas d'échec (fichier absent ou illisible), on garde
        l'état déjà en mémoire : mieux vaut une prédiction basée sur l'état
        de la veille qu'une API en erreur.
        """
        try:
            if self.team_state_path.stat().st_mtime != self._team_state_mtime:
                self.reload_team_state()
                logger.info("team_state.json rechargé (as_of : %s)", self._team_state.get("as_of"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Rechargement de team_state.json impossible, état précédent conservé")

    # ------------------------------------------------------------------
    # Construction des features pour UN match (home vs away)
    # ------------------------------------------------------------------

    def _get_team(self, team_id: int) -> dict:
        team = self._team_state["teams"].get(str(team_id))
        if team is None:
            raise UnknownTeamError(
                f"Équipe {team_id} absente de team_state.json "
                "(nouvelle franchise ? relancer fetch_games.py + team_state.py)."
            )
        return team

    def _team_side_features(self, team: dict, is_home: bool, as_of_date) -> dict:
        """
        Calcule les features d'UNE équipe pour le match, dans le rôle
        home ou away. Utilise exactement les mêmes fonctions que le
        pipeline batch (features_lib.py), appliquées à l'état résumé de
        team_state.json plutôt qu'à l'historique complet.

        Valeurs manquantes (ex: équipe en tout début de saison, moins de
        5 matchs joués) : on laisse `None` -> converti en NaN plus bas.
        XGBoost gère nativement les NaN (pas besoin d'imputer une valeur
        arbitraire qui biaiserait la prédiction).
        """
        recent = team["recent_games"]  # déjà limité à 10, plus ancien -> plus récent
        pts_scored = [g["pts_scored"] for g in recent]
        pts_allowed = [g["pts_allowed"] for g in recent]
        wins_recent = [g["win"] for g in recent]

        has_box_score = bool(recent) and "fga" in recent[0]
        net_rtg_series = (
            [
                net_rtg_from_boxscore(
                    g["pts_scored"], g["pts_allowed"], g["fga"], g["oreb"], g["tov"], g["fta"]
                )
                for g in recent
            ]
            if has_box_score else []
        )
        # net_rtg_from_boxscore peut renvoyer None (possessions estimées à 0,
        # ligne corrompue) : à exclure des moyennes glissantes.
        net_rtg_series = [v for v in net_rtg_series if v is not None]

        # Saison du match à prédire, déduite de SA date (pas de celle de
        # team_state.json) : si le match tombe dans une saison postérieure,
        # on reproduit ce que fait le pipeline d'entraînement au changement
        # de saison — régression Elo vers la moyenne ET remise à zéro de
        # win_pct/win_pct_context (calculés par saison dans
        # feature_engineering.py). win_pct_last10 et les moyennes glissantes
        # ne sont pas remis à zéro, comme à l'entraînement.
        known_season = self._team_state["as_of"]["season"]
        target_season = season_for_date(as_of_date)
        elo = elo_for_target_season(
            rating=team["elo"],
            last_known_season=known_season,
            target_season=target_season,
        )

        if is_later_season(target_season, known_season):
            season_results = {"all": [], "home": [], "away": []}
        else:
            season_results = team["season_results"]
        context_results = season_results["home" if is_home else "away"]

        features = {
            "win_pct": win_pct(season_results["all"]),
            "win_pct_context": win_pct(context_results),
            "win_pct_last10": win_pct(wins_recent[-FORM_WINDOW:]),
            "rest_days": rest_days(team["last_game_date"], as_of_date),
            "elo": elo,
        }
        for w in ROLLING_WINDOWS:
            features[f"avg_pts_scored_last{w}"] = rolling_mean_last_n(pts_scored, w)
            features[f"avg_pts_allowed_last{w}"] = rolling_mean_last_n(pts_allowed, w)
            if has_box_score:
                features[f"net_rtg_last{w}"] = rolling_mean_last_n(net_rtg_series, w)

        return features

    def build_feature_row(self, home_team_id: int, away_team_id: int, as_of_date=None) -> dict:
        """
        Construit le dict complet des features pour le match home vs away,
        avec les préfixes home_/away_/diff_ exactement comme dans
        feature_engineering.py (add_diff_features / back_to_wide).
        """
        as_of_date = as_of_date or self._team_state["as_of"]["last_game_date"]

        home_team = self._get_team(home_team_id)
        away_team = self._get_team(away_team_id)

        home_feats = self._team_side_features(home_team, is_home=True, as_of_date=as_of_date)
        away_feats = self._team_side_features(away_team, is_home=False, as_of_date=as_of_date)

        row = {}
        for key, value in home_feats.items():
            row[f"home_{key}"] = value
        for key, value in away_feats.items():
            row[f"away_{key}"] = value

        # Features différentielles : mêmes noms que add_diff_features() dans
        # feature_engineering.py. On ne différencie que les clés présentes
        # des deux côtés (net_rtg_last* absent si pas de boxscore).
        for key in home_feats:
            if key in away_feats and home_feats[key] is not None and away_feats[key] is not None:
                row[f"diff_{key}"] = home_feats[key] - away_feats[key]
            elif key in away_feats:
                row[f"diff_{key}"] = None

        return row

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(self, home_team_id: int, away_team_id: int, as_of_date=None) -> GamePrediction:
        row = self.build_feature_row(home_team_id, away_team_id, as_of_date)

        # Ordre des colonnes IMPOSÉ par feature_columns.json (généré à
        # l'entraînement) : ne jamais reconstruire cet ordre à la main ici,
        # c'est exactement le piège identifié plus tôt (désynchronisation
        # entraînement/prédiction).
        missing_cols = [c for c in self._feature_columns if c not in row]
        if missing_cols:
            raise PredictorNotReadyError(
                f"Colonnes manquantes dans les features calculées : {missing_cols}. "
                "features_lib.py et feature_columns.json ont probablement divergé."
            )

        # astype(float) : convertit les None en NaN. Sans ça, une colonne
        # dont la seule valeur est None est typée "object" et XGBoost refuse
        # le DataFrame (cas réel : win_pct en tout début de saison).
        X = pd.DataFrame(
            [[row[c] for c in self._feature_columns]], columns=self._feature_columns
        ).astype(float)

        proba = self._model.predict_proba(X)[0]
        home_win_proba = float(proba[1])
        away_win_proba = float(proba[0])

        # Score approximatif : simple heuristique (le modèle n'est PAS
        # entraîné à prédire un score, seulement l'issue). On combine la
        # forme offensive récente d'une équipe avec la forme défensive
        # récente de l'adversaire. À affiner plus tard si un modèle de
        # régression dédié est ajouté (cf. étape "optionnel" du CLAUDE.md).
        home_off = row.get("home_avg_pts_scored_last10") or row.get("home_avg_pts_scored_last5")
        away_def = row.get("away_avg_pts_allowed_last10") or row.get("away_avg_pts_allowed_last5")
        away_off = row.get("away_avg_pts_scored_last10") or row.get("away_avg_pts_scored_last5")
        home_def = row.get("home_avg_pts_allowed_last10") or row.get("home_avg_pts_allowed_last5")

        predicted_home_score = (
            (home_off + away_def) / 2 if home_off is not None and away_def is not None else 110.0
        )
        predicted_away_score = (
            (away_off + home_def) / 2 if away_off is not None and home_def is not None else 108.0
        )

        return GamePrediction(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_win_probability=round(home_win_proba, 4),
            away_win_probability=round(away_win_proba, 4),
            predicted_winner="home" if home_win_proba >= 0.5 else "away",
            predicted_home_score=round(predicted_home_score, 1),
            predicted_away_score=round(predicted_away_score, 1),
            features_used=row,
        )


# Instance unique réutilisée par l'API (chargement du modèle une seule fois
# au démarrage, pas à chaque requête) — à importer dans main.py / routers.
_service: Optional[PredictorService] = None


def get_predictor_service() -> PredictorService:
    global _service
    if _service is None:
        _service = PredictorService()
    else:
        _service.reload_team_state_if_changed()
    return _service