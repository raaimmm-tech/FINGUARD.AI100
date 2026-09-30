from __future__ import annotations

import pandas as pd

STATUSES = ["New", "Investigating", "Confirmed fraud", "False positive", "Closed"]


def initialize_cases(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "case_status" not in result.columns:
        result["case_status"] = "New"
    if "case_comment" not in result.columns:
        result["case_comment"] = ""
    if "alert_priority" not in result.columns:
        result["alert_priority"] = result["risk_category"].map({"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}).fillna("P4")
    return result


def update_case(df: pd.DataFrame, transaction_id: str, status: str, comment: str = "") -> pd.DataFrame:
    if status not in STATUSES:
        raise ValueError("Неизвестный статус кейса.")
    result = df.copy()
    mask = result["transaction_id"].astype(str) == str(transaction_id)
    result.loc[mask, "case_status"] = status
    result.loc[mask, "case_comment"] = comment
    return result
