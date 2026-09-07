import pandas as pd 
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os 

load_dotenv()
db_host = os.getenv("DB_HOST")
db_password = os.getenv("DB_PASSWORD")
db_user = os.getenv("DB_USER")
db_port = os.getenv("DB_PORT")

pg_conn = psycopg2.connect(
    host=db_host,
    dbname="postgres",
    user=db_user,
    password=db_password,
    port=db_port
)

matches_df = pd.read_sql_query("SELECT * FROM matches ORDER BY date", pg_conn)

home_rows = matches_df[['match_id', 'date', 'home_team_api', 'away_team_api', 'home_team_goal', 'away_team_goal']].copy()
home_rows.columns = ['match_id', 'date', 'team_id', 'opponent_id', 'goals_scored', 'goals_conceded']
home_rows['is_home'] = True

away_rows = matches_df[['match_id', 'date', 'away_team_api', 'home_team_api', 'away_team_goal', 'home_team_goal']].copy()
away_rows.columns = ['match_id', 'date', 'team_id', 'opponent_id', 'goals_scored', 'goals_conceded']
away_rows['is_home'] = False

team_matches = pd.concat([home_rows, away_rows]).sort_values('date')

conditions = [
    team_matches['goals_scored'] > team_matches['goals_conceded'],
    team_matches['goals_scored'] < team_matches['goals_conceded']
]
choices = [
    3,
    0
]

team_matches['points'] = np.select(conditions, choices, default=1)

team_matches['points_form'] = (
    team_matches.groupby('team_id')['points']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

team_matches['goals_scored_form'] = (
    team_matches.groupby('team_id')['goals_scored']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

team_matches['goals_conceded_form'] = (
    team_matches.groupby('team_id')['goals_conceded']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

sample_team = team_matches[team_matches['team_id'] == 8559].head(10)
# print(sample_team[['date', 'goals_scored', 'goals_conceded', 'points', 'points_form']])

home_matches = team_matches[team_matches['is_home'] == True].copy()
away_matches = team_matches[team_matches['is_home'] == False].copy()

# .shift(1) -> to start calculations using values down by one row to prevent data leakeage ( where you use the real value to calculate it's predicted value)
# window -> to select how many rows far back we use
# min_periods -> min number of matches to have a form
home_matches['home_form'] = (
    home_matches.groupby('team_id')['points']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

home_matches['home_goals'] = (
    home_matches.groupby('team_id')['goals_scored']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

home_matches['home_conceded'] = (
    home_matches.groupby('team_id')['goals_conceded']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

away_matches['away_form'] = (
    away_matches.groupby('team_id')['points']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

away_matches['away_goals'] = (
    away_matches.groupby('team_id')['goals_scored']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

away_matches['away_conceded'] = (
    away_matches.groupby('team_id')['goals_conceded']
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

# print(home_matches[['match_id', 'team_id', 'home_form']].head(10))
# print(team_matches[['match_id', 'team_id']].dtypes)
# print(home_matches[['match_id', 'team_id']].dtypes)

team_matches = team_matches.merge(
    home_matches[['match_id', 'team_id', 'home_form', 'home_goals', 'home_conceded']],
    on=['match_id', 'team_id'],
    how='left'
)

team_matches = team_matches.merge(
    away_matches[['match_id', 'team_id', 'away_form', 'away_goals', 'away_conceded']],
    on=['match_id', 'team_id'],
    how='left'
)


team_matches['h2h_form'] = (
    team_matches.groupby(['team_id', 'opponent_id'])['points']
    .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
)

h2h_sample = team_matches[(team_matches['team_id'] == 8191) & (team_matches['opponent_id'] == 8456)]
print(h2h_sample[['date', 'points', 'h2h_form']])