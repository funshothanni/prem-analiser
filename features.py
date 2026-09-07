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
print(sample_team[['date', 'goals_scored', 'goals_conceded', 'points', 'points_form']])