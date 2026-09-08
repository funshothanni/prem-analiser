import pandas as pd
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

pg_conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname="postgres",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)

matches_df = pd.read_sql_query(
    """
    SELECT
        m.*,
        home.team_name AS home_team,
        away.team_name AS away_team
    FROM matches m
    JOIN teams home
        ON m.home_team_api = home.team_api_id
    JOIN teams away
        ON m.away_team_api = away.team_api_id
    ORDER BY m.date
    """,
    pg_conn
)

matches_df['total_goals'] = (
    matches_df['home_team_goal']
    + matches_df['away_team_goal']
)

season_scoring = (
    matches_df.groupby('season')
    .agg(
        matches=('match_id', 'count'),
        total_goals=('total_goals', 'sum'),
        avg_goals=('total_goals', 'mean')
    )
    .reset_index()
)

conditions = [
    matches_df['home_team_goal'] > matches_df['away_team_goal'],
    matches_df['home_team_goal'] < matches_df['away_team_goal']
]

choices = ['Home Win', 'Away Win']

matches_df['result'] = np.select(
    conditions,
    choices,
    default='Draw'
)

season_results = (
    matches_df.groupby(['season', 'result'])
    .size()
    .unstack(fill_value=0)
)

total_matches = season_results.sum(axis=1)

season_results['Home Win %'] = (
    season_results['Home Win']
    / total_matches
    * 100
)

season_results['Away Win %'] = (
    season_results['Away Win']
    / total_matches
    * 100
)

season_results['Draw %'] = (
    season_results['Draw']
    / total_matches
    * 100
)

home_performance = matches_df[
    ['season', 'home_team', 'home_team_goal', 'away_team_goal']
].copy()

home_performance.columns = [
    'season',
    'team',
    'goals_scored',
    'goals_conceded'
]

away_performance = matches_df[
    ['season', 'away_team', 'away_team_goal', 'home_team_goal']
].copy()

away_performance.columns = [
    'season',
    'team',
    'goals_scored',
    'goals_conceded'
]

team_performance = pd.concat(
    [home_performance, away_performance],
    ignore_index=True
)

conditions = [
    team_performance['goals_scored'] > team_performance['goals_conceded'],
    team_performance['goals_scored'] < team_performance['goals_conceded']
]

choices = [3, 0]

team_performance['points'] = np.select(
    conditions,
    choices,
    default=1
)

team_summary = (
    team_performance.groupby('team')
    .agg(
        matches=('team', 'size'),
        goals_scored=('goals_scored', 'sum'),
        goals_conceded=('goals_conceded', 'sum'),
        points=('points', 'sum')
    )
    .reset_index()
)

team_summary['goal_difference'] = (
    team_summary['goals_scored']
    - team_summary['goals_conceded']
)

team_summary = team_summary.sort_values(
    'points',
    ascending=False
)

team_summary['points_per_match'] = (
    team_summary['points']
    / team_summary['matches']
)

team_summary = team_summary.sort_values(
    'points_per_match',
    ascending=False
)

home_points = (
    home_performance.assign(
        points=np.select(
            [
                home_performance['goals_scored'] > home_performance['goals_conceded'],
                home_performance['goals_scored'] < home_performance['goals_conceded']
            ],
            [3, 0],
            default=1
        )
    )
    .groupby('team')
    .agg(
        home_matches=('team', 'size'),
        home_points=('points', 'sum')
    )
)

home_points['home_ppm'] = (
    home_points['home_points']
    / home_points['home_matches']
)

away_points = (
    away_performance.assign(
        points=np.select(
            [
                away_performance['goals_scored'] > away_performance['goals_conceded'],
                away_performance['goals_scored'] < away_performance['goals_conceded']
            ],
            [3, 0],
            default=1
        )
    )
    .groupby('team')
    .agg(
        away_matches=('team', 'size'),
        away_points=('points', 'sum')
    )
)

away_points['away_ppm'] = (
    away_points['away_points']
    / away_points['away_matches']
)

home_away_summary = home_points.join(away_points)

home_away_summary['home_advantage'] = (
    home_away_summary['home_ppm']
    - home_away_summary['away_ppm']
)

home_away_summary = home_away_summary.sort_values(
    'home_advantage',
    ascending=False
)

season_scoring.to_csv(
    'data_outputs/season_scoring.csv',
    index=False
)

season_results.to_csv(
    'data_outputs/season_results.csv'
)

team_summary.to_csv(
    'data_outputs/team_summary.csv',
    index=False
)

home_away_summary.to_csv(
    'data_outputs/home_away_summary.csv'
)

print("Analysis files exported successfully.")