from __future__ import annotations

import pandas as pd

EXPECTED_REGIONS = {"Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe", "Pavlodar", "Taraz", "Ust-Kamenogorsk", "Moscow", "Dubai", "Istanbul"}


def check_data_quality(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Ожидался pandas DataFrame.")

    required = ["transaction_id", "client_id", "datetime", "amount", "region"]
    missing_columns = [column for column in required if column not in df.columns]
    duplicate_ids = int(df["transaction_id"].duplicated().sum()) if "transaction_id" in df.columns else 0
    missing_values = int(df.isna().sum().sum())
    negative_amounts = 0
    invalid_dates = 0
    unknown_regions = 0

    if "amount" in df.columns:
        amounts = pd.to_numeric(df["amount"], errors="coerce")
        negative_amounts = int((amounts < 0).sum())
        missing_values += int(amounts.isna().sum())
    if "datetime" in df.columns:
        invalid_dates = int(pd.to_datetime(df["datetime"], errors="coerce").isna().sum())
    if "region" in df.columns:
        unknown_regions = int((~df["region"].astype(str).isin(EXPECTED_REGIONS)).sum())

    issues = []
    if missing_columns:
        issues.append(f"Отсутствуют колонки: {', '.join(missing_columns)}")
    if duplicate_ids:
        issues.append(f"Дубликаты transaction_id: {duplicate_ids}")
    if missing_values:
        issues.append(f"Пропущенные значения: {missing_values}")
    if negative_amounts:
        issues.append(f"Отрицательные суммы: {negative_amounts}")
    if invalid_dates:
        issues.append(f"Некорректные даты: {invalid_dates}")
    if unknown_regions:
        issues.append(f"Неизвестные регионы: {unknown_regions}")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_ids": duplicate_ids,
        "missing_values": missing_values,
        "negative_amounts": negative_amounts,
        "invalid_dates": invalid_dates,
        "unknown_regions": unknown_regions,
        "issues": issues,
        "status": "Passed" if not issues else "Warning",
    }
