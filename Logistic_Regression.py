from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
import pandas

OUT_PATH = Path(__file__).parent / "Data" / "Logistic_Prediction.csv"

def save_predictions(X, y, xg):
    predictions = X.copy()
    predictions["goal"] = y
    predictions["xg"] = xg
    predictions.to_csv(OUT_PATH)

def logistic_regression(X, y):
    model = LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(model, X, y, cv=cv, scoring=["neg_log_loss", "roc_auc"])

    xg = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    save_predictions(X, y, xg)

    return [-scores["test_neg_log_loss"].mean(), scores["test_roc_auc"].mean()]
