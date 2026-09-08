import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load database credentials from .env
load_dotenv()

db_host = os.getenv("DB_HOST")
db_password = os.getenv("DB_PASSWORD")
db_user = os.getenv("DB_USER")
db_port = os.getenv("DB_PORT")


# Connect to PostgreSQL
pg_conn = psycopg2.connect(
    host=db_host,
    dbname="postgres",
    user=db_user,
    password=db_password,
    port=db_port
)


# Load Premier League matches in chronological order
matches_df = pd.read_sql_query(
    "SELECT * FROM matches ORDER BY date",
    pg_conn
)


# Create one row per team per match
home_rows = matches_df[
    [
        "match_id",
        "date",
        "home_team_api",
        "away_team_api",
        "home_team_goal",
        "away_team_goal",
    ]
].copy()

home_rows.columns = [
    "match_id",
    "date",
    "team_id",
    "opponent_id",
    "goals_scored",
    "goals_conceded",
]

home_rows["is_home"] = True

away_rows = matches_df[
    [
        "match_id",
        "date",
        "away_team_api",
        "home_team_api",
        "away_team_goal",
        "home_team_goal",
    ]
].copy()

away_rows.columns = [
    "match_id",
    "date",
    "team_id",
    "opponent_id",
    "goals_scored",
    "goals_conceded",
]

away_rows["is_home"] = False


team_matches = pd.concat(
    [home_rows, away_rows]
).sort_values("date")


# Assign league points for each team in each match
conditions = [
    team_matches["goals_scored"] > team_matches["goals_conceded"],
    team_matches["goals_scored"] < team_matches["goals_conceded"],
]

choices = [3, 0]

team_matches["points"] = np.select(
    conditions,
    choices,
    default=1
)


# Calculate overall form from the previous 5 matches
# shift(1) prevents the current match result from being used
# in its own pre-match features
team_matches["points_form"] = (
    team_matches.groupby("team_id")["points"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

team_matches["goals_scored_form"] = (
    team_matches.groupby("team_id")["goals_scored"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

team_matches["goals_conceded_form"] = (
    team_matches.groupby("team_id")["goals_conceded"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)


# Separate home and away matches
home_matches = team_matches[
    team_matches["is_home"] == True
].copy()

away_matches = team_matches[
    team_matches["is_home"] == False
].copy()


# Calculate home-specific form from the previous 5 home matches
home_matches["home_form"] = (
    home_matches.groupby("team_id")["points"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

home_matches["home_goals"] = (
    home_matches.groupby("team_id")["goals_scored"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

home_matches["home_conceded"] = (
    home_matches.groupby("team_id")["goals_conceded"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)


# Calculate away-specific form from the previous 5 away matches
away_matches["away_form"] = (
    away_matches.groupby("team_id")["points"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

away_matches["away_goals"] = (
    away_matches.groupby("team_id")["goals_scored"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)

away_matches["away_conceded"] = (
    away_matches.groupby("team_id")["goals_conceded"]
    .transform(
        lambda x: x.shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
    )
)


# Merge home-specific features back into team_matches
team_matches = team_matches.merge(
    home_matches[
        [
            "match_id",
            "team_id",
            "home_form",
            "home_goals",
            "home_conceded",
        ]
    ],
    on=["match_id", "team_id"],
    how="left",
)


# Merge away-specific features back into team_matches
team_matches = team_matches.merge(
    away_matches[
        [
            "match_id",
            "team_id",
            "away_form",
            "away_goals",
            "away_conceded",
        ]
    ],
    on=["match_id", "team_id"],
    how="left",
)


# Carry each team's most recent home/away form forward
team_matches = team_matches.sort_values(
    ["team_id", "date"]
)

venue_features = [
    "home_form",
    "home_goals",
    "home_conceded",
    "away_form",
    "away_goals",
    "away_conceded",
]

for feature in venue_features:
    team_matches[feature] = (
        team_matches.groupby("team_id")[feature]
        .ffill()
    )


# Create home-team and away-team versions of the features
home_features = team_matches.add_prefix("home_")
home_features = home_features.rename(
    columns={"home_match_id": "match_id"}
)

away_features = team_matches.add_prefix("away_")
away_features = away_features.rename(
    columns={"away_match_id": "match_id"}
)


# Merge the current home team's historical features
matches_with_form = matches_df.merge(
    home_features,
    left_on=["match_id", "home_team_api"],
    right_on=["match_id", "home_team_id"],
    how="left",
)


# Merge the current away team's historical features
matches_with_form = matches_with_form.merge(
    away_features,
    left_on=["match_id", "away_team_api"],
    right_on=["match_id", "away_team_id"],
    how="left",
)


# Create the match-result target:
# H = home win, A = away win, D = draw
conditions = [
    matches_with_form["home_team_goal"]
    > matches_with_form["away_team_goal"],

    matches_with_form["home_team_goal"]
    < matches_with_form["away_team_goal"],
]

choices = ["H", "A"]

matches_with_form["result"] = np.select(
    conditions,
    choices,
    default="D",
)


# Historical features used by the prediction model
feature_columns = [
    "home_points_form",
    "home_goals_scored_form",
    "home_goals_conceded_form",
    "home_home_form",
    "home_home_goals",
    "home_home_conceded",
    "home_away_form",
    "home_away_goals",
    "home_away_conceded",

    "away_points_form",
    "away_goals_scored_form",
    "away_goals_conceded_form",
    "away_home_form",
    "away_home_goals",
    "away_home_conceded",
    "away_away_form",
    "away_away_goals",
    "away_away_conceded",
]


# Use all seasons before 2015/16 as training data
train_mask = (
    matches_with_form["season"] != "2015/2016"
)

# Use the final season as unseen test data
test_mask = (
    matches_with_form["season"] == "2015/2016"
)


# Model input features
X_train = matches_with_form.loc[
    train_mask,
    feature_columns,
]

X_test = matches_with_form.loc[
    test_mask,
    feature_columns,
]


# Actual match results
y_train = matches_with_form.loc[
    train_mask,
    "result",
]

y_test = matches_with_form.loc[
    test_mask,
    "result",
]


# Fill missing historical values using medians
# learned only from the training data
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


# Train a simple multiclass logistic regression model
model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_imputed,
    y_train,
)


# Predict outcomes for the unseen 2015/16 season
y_pred = model.predict(
    X_test_imputed
)


# Evaluate the model
accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(f"Model Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)


# Close the database connection
pg_conn.close()
