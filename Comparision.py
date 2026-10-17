from pathlib import Path
import duckdb as db
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

linear_data = logistic_regression(X_train, y_train, X_test, y_test)