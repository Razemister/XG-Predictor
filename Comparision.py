from pathlib import Path
import duckdb as db
from Logistic_Regression import logistic_regression
from Freeze_Frame import freeze_frame_log, freeze_frame_gb

DATA_DIR = Path(__file__).parent / "Data"

shots = db.read_csv(str(DATA_DIR / "Shots_Data.csv"))


shot_features = db.sql("""
    SELECT
        *,
        sqrt(power(120 - location_x, 2) + power(location_y - 40, 2)) AS distance,
        atan2(
            8 * (120 - location_x),
            power(120 - location_x, 2) + power(location_y - 40, 2) - 16
        ) AS angle,
        CASE WHEN shot_outcome = 'Goal' THEN 1 ELSE 0 END AS goal
    FROM shots
""").df().set_index("shot_id", drop=False)

y = shot_features["goal"]
X_baseline = shot_features[["distance", "angle"]]
X_freeze_frame = shot_features.drop(columns="goal")

linear_data = logistic_regression(X_baseline, y)
freeze_frame_log_data = freeze_frame_log(X_freeze_frame, y)
freeze_frame_gb_data = freeze_frame_gb(X_freeze_frame, y)

print(f"Baseline logistic:     log loss {linear_data[0]:.4f} | ROC AUC {linear_data[1]:.4f}")
print(f"Freeze-frame logistic: log loss {freeze_frame_log_data[0]:.4f} | ROC AUC {freeze_frame_log_data[1]:.4f}")
print(f"    best config: {freeze_frame_log_data[2]}")
print(f"Freeze-frame GB:       log loss {freeze_frame_gb_data[0]:.4f} | ROC AUC {freeze_frame_gb_data[1]:.4f}")
print(f"    best config: {freeze_frame_gb_data[2]}")
