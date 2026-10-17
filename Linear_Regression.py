from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import pandas

OUT_PATH = Path(__file__).parent / "Data" / "Linear_Prediction.csv"

def save_predictions(X_test, y_test, xg):
    predictions = X_test.copy()
    predictions["goal"] = y_test
    predictions["xg"] = xg
    predictions.to_csv(OUT_PATH)

def linear_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    xg = model.predict_proba(X_test)[:, 1]

    save_predictions(X_test, y_test, xg)

    return [log_loss(y_test, xg), roc_auc_score(y_test, xg)]
