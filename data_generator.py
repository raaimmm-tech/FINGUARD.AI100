from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

REGIONS = ["Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe", "Pavlodar", "Taraz", "Ust-Kamenogorsk", "Moscow", "Dubai", "Istanbul"]
LOCAL_REGIONS = REGIONS[:8]
CHANNELS = ["mobile_app", "internet_banking", "atm", "pos", "card_present"]
CLIENT_NAMES = ["Айгерим Сатпаева", "Нурлан Ахметов", "Дана Ержанова", "Ерлан Смагулов", "Жанна Тулегенова", "Санжар Оспанов", "Асель Байжанова", "Тимур Касымов", "Гульнара Абенова", "Азамат Нурланов", "Мадина Кенжебаева", "Алишер Досов"]
FRAUD_TYPES = ["Night transaction", "New device attack", "New recipient", "International transfer", "Rapid burst", "Large amount", "Account takeover", "Money mule"]


def _random_moment(start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


def generate_synthetic_transactions(n: int = 700, anomaly_rate: float = 0.10, seed: int | None = 42) -> pd.DataFrame:
    if n < 1:
        raise ValueError("Количество транзакций должно быть больше нуля.")
    if not 0 <= anomaly_rate <= 1:
        raise ValueError("Доля аномалий должна быть от 0 до 1.")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    clients = []
    for index in range(max(10, min(100, n // 20))):
        client_id = f"CL{1000 + index}"
        clients.append({
            "client_id": client_id,
            "client_name": random.choice(CLIENT_NAMES),
            "home_region": random.choice(LOCAL_REGIONS),
            "avg_amount": round(float(np.random.uniform(15_000, 250_000)), 2),
            "known_devices": [f"DEV-{client_id}-1", f"DEV-{client_id}-2"],
            "known_recipients": [f"RCP-{client_id}-{i}" for i in range(1, 4)],
        })

    end = datetime.now()
    start = end - timedelta(days=30)
    anomaly_flags = [True] * int(n * anomaly_rate) + [False] * (n - int(n * anomaly_rate))
    random.shuffle(anomaly_flags)
    rows = []

    for index, is_anomaly in enumerate(anomaly_flags):
        client = random.choice(clients)
        moment = _random_moment(start, end)
        fraud_type = "Normal"
        expected_amount = client["avg_amount"]
        device_new = False
        recipient_new = False
        region = client["home_region"]
        amount = max(500, np.random.normal(expected_amount, expected_amount * 0.25))

        if is_anomaly:
            fraud_type = random.choice(FRAUD_TYPES)
            if fraud_type == "Night transaction":
                moment = moment.replace(hour=random.choice([0, 1, 2, 3, 4, 23]), minute=random.randint(0, 59))
                amount = expected_amount * np.random.uniform(1.2, 5)
            elif fraud_type == "New device attack":
                device_new = True
                amount = expected_amount * np.random.uniform(1.5, 8)
            elif fraud_type == "New recipient":
                recipient_new = True
                amount = expected_amount * np.random.uniform(1.5, 8)
            elif fraud_type == "International transfer":
                region = random.choice(["Moscow", "Dubai", "Istanbul"])
                amount = expected_amount * np.random.uniform(2, 10)
            elif fraud_type == "Rapid burst":
                amount = expected_amount * np.random.uniform(1, 4)
            elif fraud_type == "Large amount":
                amount = expected_amount * np.random.uniform(8, 18)
            elif fraud_type == "Account takeover":
                device_new = True
                recipient_new = True
                region = random.choice(["Moscow", "Dubai", "Istanbul"])
                moment = moment.replace(hour=random.choice([0, 1, 2, 3, 23]), minute=random.randint(0, 59))
                amount = expected_amount * np.random.uniform(5, 15)
            elif fraud_type == "Money mule":
                recipient_new = True
                amount = expected_amount * np.random.uniform(2, 7)
            if fraud_type not in ["Night transaction", "International transfer"] and random.random() < 0.65:
                device_new = True
            if fraud_type not in ["New recipient", "Money mule"] and random.random() < 0.70:
                recipient_new = True
            if region == client["home_region"] and random.random() < 0.75:
                region = random.choice(["Moscow", "Dubai", "Istanbul"])

        amount = round(max(500, amount), 2)
        rows.append({
            "transaction_id": f"TXN{100000 + index}",
            "client_id": client["client_id"],
            "client_name": client["client_name"],
            "datetime": moment,
            "amount": amount,
            "currency": "KZT",
            "region": region,
            "home_region": client["home_region"],
            "channel": random.choice(CHANNELS),
            "device_id": f"DEV-NEW-{random.randint(10000, 99999)}" if device_new else random.choice(client["known_devices"]),
            "device_new": device_new,
            "recipient_id": f"RCP-NEW-{random.randint(10000, 99999)}" if recipient_new else random.choice(client["known_recipients"]),
            "recipient_new": recipient_new,
            "fraud_label": int(is_anomaly),
            "fraud_type": fraud_type,
            "expected_amount": round(expected_amount, 2),
        })

    result = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    result["amount_ratio"] = (result["amount"] / result["expected_amount"].replace(0, np.nan)).fillna(0).round(2)
    result["is_night"] = result["datetime"].dt.hour.between(0, 5)
    result["shared_device_count"] = result.groupby("device_id")["client_id"].transform("nunique")
    return result
