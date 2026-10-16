# %% [markdown]
# # Exploration de nba_api
#
# Objectif : comprendre la structure exacte des données renvoyées par nba_api
# AVANT d'écrire data/collectors/fetch_games.py en dur.
#
# Prérequis (dans ton venv api/ ou un venv dédié pour data/) :
#   pip install nba_api pandas
#
# nba_api tape sur stats.nba.com, qui n'est pas officiel : évite de spammer
# les requêtes, mets des pauses entre les appels, et si un appel timeout,
# relance simplement (ça arrive souvent, ce n'est pas ton code qui est cassé).

# %%
import time
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Liste des équipes
#
# On commence par ça car on a besoin de mapper nba_team_id <-> nom/abréviation
# pour remplir la table `teams` de ton schéma.

# %%
from nba_api.stats.static import teams

nba_teams = teams.get_teams()
df_teams = pd.DataFrame(nba_teams)
print(df_teams.shape)
print(df_teams.head())
print(df_teams.columns.tolist())

# Sauvegarde tout de suite en CSV pour pouvoir la réutiliser sans rappeler l'API
df_teams.to_csv("teams_static.csv", index=False)

# %% [markdown]
# ## 2. Historique des matchs d'une saison (LeagueGameFinder)
#
# C'est l'endpoint qui va nourrir ta table `games`.
# Attention : LeagueGameFinder renvoie UNE LIGNE PAR ÉQUIPE PAR MATCH
# (donc 2 lignes par match : une pour l'équipe home, une pour l'away).
# Il faudra les recombiner en une seule ligne par match dans clean_data.py.

# %%
from nba_api.stats.endpoints import leaguegamefinder

# Format saison NBA : "2023-24"
season = "2023-24"

gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable=season,
    league_id_nullable="00",  # 00 = NBA (vs G-League, WNBA...)
)
df_games_raw = gamefinder.get_data_frames()[0]

print(df_games_raw.shape)
print(df_games_raw.columns.tolist())
print(df_games_raw.head(10))

df_games_raw.to_csv(f"games_raw_{season}.csv", index=False)

# Pause polie avant le prochain appel
time.sleep(1)

# %% [markdown]
# ## 3. Vérifier le format "une ligne par équipe" et comment le dédupliquer
#
# Colonnes clés à observer : GAME_ID, TEAM_ID, MATCHUP, WL, PTS
# MATCHUP contient "@" pour away et "vs." pour home -> c'est comme ça
# qu'on distingue home_team_id / away_team_id.

# %%
sample_game_id = df_games_raw["GAME_ID"].iloc[0]
print(df_games_raw[df_games_raw["GAME_ID"] == sample_game_id])

# %% [markdown]
# ## 4. Matchs du jour (pour remplacer /games/today plus tard)
#
# Endpoint différent, plus léger, pensé pour le "live"/scoreboard.

# %%
from nba_api.stats.endpoints import scoreboardv2

today_board = scoreboardv2.ScoreboardV2()
frames = today_board.get_data_frames()

# ScoreboardV2 renvoie plusieurs DataFrames : GameHeader, LineScore, etc.
for i, f in enumerate(frames):
    print(f"--- frame {i} : shape {f.shape} ---")
    print(f.columns.tolist())

# %% [markdown]
# ## Prochaines questions à répondre en regardant les sorties ci-dessus :
#
# 1. Quelles colonnes de LeagueGameFinder correspondent exactement aux champs
#    de ta table `games` (home_score, away_score, game_date, season, status) ?
# 2. Combien de temps met l'appel LeagueGameFinder pour une saison entière ?
#    (ça conditionne si tu peux tout récupérer d'un coup ou par lots)
# 3. Le WL (Win/Loss) et le MATCHUP suffisent-ils à reconstruire home/away,
#    ou faut-il croiser avec un autre endpoint ?
# 4. ScoreboardV2 fonctionne-t-il hors saison / la nuit (pas de matchs) ?
#    -> gérer le cas "liste vide" dans /games/today.
#
# Une fois ces réponses en main, on écrit fetch_games.py avec une logique
# propre : boucle sur les saisons, sleep entre les appels, retry en cas
# d'erreur réseau, export vers CSV (ou direct en BDD une fois qu'elle existe).