from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(df: pd.DataFrame, score_column: str = "final_score", threshold: float = 50.0) -> dict:
    if "fraud_label" not in df.columns:
        return {"available": False, "message": "Для расчета метрик нужна колонка fraud_label."}
    y_true = pd.to_numeric(df["fraud_label"], errors="coerce").fillna(0).astype(int)
    y_pred = (pd.to_numeric(df[score_column], errors="coerce").fillna(0) >= threshold).astype(int)
    result = {"available": True, "accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0), "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
    try:
        result["roc_auc"] = roc_auc_score(y_true, df[score_column])
    except ValueError:
        result["roc_auc"] = None
    return result
