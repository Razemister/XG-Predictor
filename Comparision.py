from pathlib import Path
import duckdb as db
from Logistic_Regression import logistic_regression

DATA_DIR = Path(__file__).parent / "Data"

shots = db.read_csv(str(DATA_DIR / "Shots_Data.csv"))


shot_features = db.sql("""
    SELECT
        shot_id,
        sqrt(power(120 - location_x, 2) + power(location_y - 40, 2)) AS distance,
        atan2(
            8 * (120 - location_x),
            power(120 - location_x, 2) + power(location_y - 40, 2) - 16
        ) AS angle,
        CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END AS goal
    FROM shots
""").df().set_index("shot_id")

X = shot_features[["distance", "angle"]]
y = shot_features["goal"]

linear_data = logistic_regression(X, y)
print(linear_data)