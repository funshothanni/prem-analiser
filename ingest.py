import sqlite3
import psycopg2
from dotenv import load_dotenv
import os

#opens file and returns connection object
sqlite_conn = sqlite3.connect("database.sqlite")
#object to run queries
sqlite_cursor = sqlite_conn.cursor()

sqlite_cursor.execute("""
    SELECT team_api_id, team_long_name FROM Team 
    WHERE team_api_id IN (
        SELECT home_team_api_id FROM Match WHERE league_id = 1729
        UNION
        SELECT away_team_api_id FROM Match WHERE league_id = 1729
    )
""")
rows = sqlite_cursor.fetchall()

load_dotenv()
db_host = os.getenv("DB_HOST")
db_password = os.getenv("DB_PASSWORD")
db_user = os.getenv("DB_USER")
db_port = os.getenv("DB_PORT")

# Connect to Supabase Postgres (destination)
pg_conn = psycopg2.connect(
    host=db_host,
    dbname="postgres",
    user=db_user,
    password=db_password,
    port=db_port
)
pg_cursor = pg_conn.cursor()

for team_id, team_name in rows:
    pg_cursor.execute("INSERT INTO teams (team_api_id, team_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (team_id, team_name)
    )

pg_conn.commit()


sqlite_cursor.execute("""
    SELECT id, league_id, season, stage, date, match_api_id,
           home_team_api_id, away_team_api_id, home_team_goal, away_team_goal
    FROM Match
    WHERE league_id = 1729
""")
match_rows = sqlite_cursor.fetchall()

for match_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal in match_rows:
    pg_cursor.execute("""INSERT INTO matches (match_id, league_id, season, stage, date, match_api_id, home_team_api, away_team_api, home_team_goal, away_team_goal) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", (match_id, league_id, season, stage, date, match_api_id, home_team_api_id,
                        away_team_api_id, home_team_goal, away_team_goal))

pg_conn.commit()
print(f"Inserted {len(match_rows)} matches!")
