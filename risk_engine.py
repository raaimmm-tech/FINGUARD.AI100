from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["transaction_id", "client_id", "datetime", "amount", "region", "device_new", "recipient_new"]
WEIGHTS = {"amount": 25, "time": 15, "device": 15, "recipient": 20, "region": 15, "frequency": 10}
RISK_COLORS = {"Low": "#2ECC71", "Medium": "#F1C40F", "High": "#E67E22", "Critical": "#E74C3C"}


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def _parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "да", "y"])


def compute_client_stats(df: pd.DataFrame) -> dict:
    result = {}
    for client_id, group in df.groupby("client_id"):
        mode = group["region"].mode()
        result[client_id] = {"avg_amount": float(group["amount"].mean()), "home_region": str(mode.iloc[0]) if not mode.empty else "Unknown"}
    return result


def _amount_score(amount: float, average: float) -> tuple[float, str | None]:
    if average <= 0 or amount <= average * 1.5:
        return 0.0, None
    ratio = amount / average
    score = min(WEIGHTS["amount"], (ratio - 1.5) / 8.5 * WEIGHTS["amount"])
    return round(score, 1), f"Сумма превышает средний чек клиента в {ratio:.1f} раза"


def _time_score(dt: pd.Timestamp) -> tuple[float, str | None]:
    if 0 <= dt.hour < 6:
        return 15.0, f"Операция проведена ночью ({dt.hour:02d}:00)"
    if 6 <= dt.hour < 8 or 22 <= dt.hour <= 23:
        return 7.5, f"Операция проведена в нетипичное время ({dt.hour:02d}:00)"
    return 0.0, None


def _frequency_score(count: int) -> tuple[float, str | None]:
    if count >= 10:
        return 10.0, f"Высокая частота: {count} операций за последний час"
    if count >= 5:
        return 6.0, f"Повышенная частота: {count} операций за последний час"
    if count >= 3:
        return 3.0, f"Несколько операций подряд: {count} за последний час"
    return 0.0, None


def _boolean_score(value: bool, weight: float, reason: str) -> tuple[float, str | None]:
    return (weight, reason) if value else (0.0, None)


def _add_frequency_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["client_id", "datetime"]).reset_index(drop=True)
    counts = np.zeros(len(result), dtype=int)
    for _, group in result.groupby("client_id"):
        indexes = group.index.to_numpy()
        times = group["datetime"].to_numpy(dtype="datetime64[m]")
        for position, index in enumerate(indexes):
            start = times[position] - np.timedelta64(60, "m")
            counts[index] = int(((times[: position + 1] >= start) & (times[: position + 1] <= times[position])).sum())
    result["tx_count_last_hour"] = counts
    return result


def categorize_risk(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Medium"
    if score < 75:
        return "High"
    return "Critical"


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Датасет пустой.")
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing)}")

    result = df.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    if result["datetime"].isna().any():
        raise ValueError("Некорректные значения datetime.")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    if result["amount"].isna().any() or (result["amount"] < 0).any():
        raise ValueError("amount должен содержать неотрицательные числа.")
    for column in ["device_new", "recipient_new"]:
        result[column] = _parse_bool(result[column])

    result = _add_frequency_column(result)
    stats = compute_client_stats(result)
    scores, categories, reasons = [], [], []
    components_storage = {"amount_score": [], "time_score": [], "device_score": [], "recipient_score": [], "region_score": [], "frequency_score": []}

    for _, row in result.iterrows():
        client_stats = stats[row["client_id"]]
        amount_score, amount_reason = _amount_score(float(row["amount"]), client_stats["avg_amount"])
        time_score, time_reason = _time_score(row["datetime"])
        device_score, device_reason = _boolean_score(bool(row["device_new"]), 15, "Новое или неизвестное устройство")
        recipient_score, recipient_reason = _boolean_score(bool(row["recipient_new"]), 20, "Новый получатель")
        region_score, region_reason = _boolean_score(str(row["region"]) != client_stats["home_region"], 15, "Регион отличается от привычного региона клиента")
        frequency_score, frequency_reason = _frequency_score(int(row["tx_count_last_hour"]))
        components = [amount_score, time_score, device_score, recipient_score, region_score, frequency_score]
        score = round(min(100.0, sum(components)), 1)
        row_reasons = [reason for reason in [amount_reason, time_reason, device_reason, recipient_reason, region_reason, frequency_reason] if reason]
        scores.append(score)
        categories.append(categorize_risk(score))
        reasons.append(row_reasons or ["Явных признаков риска не обнаружено"])
        for key, value in zip(components_storage, components):
            components_storage[key].append(value)

    result["risk_score"] = scores
    result["risk_category"] = categories
    result["risk_reasons"] = reasons
    for key, values in components_storage.items():
        result[key] = values
    return result
