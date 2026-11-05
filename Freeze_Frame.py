from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = Path(__file__).parent / "Data"
LOG_OUT_PATH = DATA_DIR / "Freeze_Frame_Log_Prediction.csv"
GB_OUT_PATH = DATA_DIR / "Freeze_Frame_GB_Prediction.csv"

DROP_COLUMNS = [
    "shot_aerial_won",
    "shot_end_location",
    "shot_id",
    "shot_outcome",
    "shot_statsbomb_xg",
    "under_pressure",
]
ONE_HOT_COLUMNS = ["shot_body_part"]
ORDINAL_COLUMNS = ["shot_first_time", "shot_technique", "shot_type"]
NUMERIC_COLUMNS = [
    "location_x",
    "location_y",
    "distance",
    "angle",
    "gk_distance_from_line",
    "gk_angle_from_center",
    "pressure_score",
]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"log_loss": "neg_log_loss", "roc_auc": "roc_auc"}


def preprocessor(scale_numeric):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("drop", "drop", DROP_COLUMNS),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLUMNS),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLUMNS),
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
    ])


def tune(pipeline, param_grid, X, y):
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=CV,
        scoring=SCORING,
        refit="log_loss",
        n_jobs=-1,
    )
    search.fit(X, y)

    results = search.cv_results_
    best = search.best_index_
    params = {name.removeprefix("model__"): value for name, value in search.best_params_.items()}
    return search.best_estimator_, params, -results["mean_test_log_loss"][best], results["mean_test_roc_auc"][best]


def save_predictions(model, X, y, out_path):
    xg = cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]
    predictions = X.copy()
    predictions["goal"] = y
    predictions["xg"] = xg
    predictions.to_csv(out_path)


def freeze_frame_log(X, y):
    pipeline = Pipeline([
        ("prep", preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, solver="liblinear")),
    ])
    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0, 1],
    }
    model, params, log_loss, roc_auc = tune(pipeline, param_grid, X, y)
    save_predictions(model, X, y, LOG_OUT_PATH)
    return [log_loss, roc_auc, params]


def freeze_frame_gb(X, y):
    pipeline = Pipeline([
        ("prep", preprocessor(scale_numeric=False)),
        ("model", HistGradientBoostingClassifier(early_stopping=False, random_state=42)),
    ])
    param_grid = {
        "model__learning_rate": [0.03, 0.1],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 6],
        "model__min_samples_leaf": [20, 100],
    }
    model, params, log_loss, roc_auc = tune(pipeline, param_grid, X, y)
    save_predictions(model, X, y, GB_OUT_PATH)
    return [log_loss, roc_auc, params]
