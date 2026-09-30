from __future__ import annotations

import os
from typing import Any

import requests


def local_explanation(tx: Any) -> str:
    reasons = tx.get("risk_reasons", []) if hasattr(tx, "get") else []
    reasons_text = "; ".join(str(reason) for reason in reasons)
    return f"Операция получила risk score {float(tx.get('risk_score', 0)):.1f}. Основные факторы: {reasons_text}. Требуется проверка аналитиком; автоматическая блокировка не выполняется."


def perplexity_explanation(tx: Any) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return local_explanation(tx)
    prompt = ("Ты anti-fraud аналитик. Сформируй краткое объяснение на русском языке для специалиста банка. " "Не придумывай факты, не рекомендуй автоматическую блокировку и используй только обезличенные признаки. " f"Данные операции: risk_score={tx.get('risk_score')}, risk_category={tx.get('risk_category')}, amount={tx.get('amount')}, region={tx.get('region')}, device_new={tx.get('device_new')}, recipient_new={tx.get('recipient_new')}, reasons={tx.get('risk_reasons')}." )
    response = requests.post("https://api.perplexity.ai/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "sonar", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
