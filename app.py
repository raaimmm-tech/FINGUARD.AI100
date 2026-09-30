from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from case_manager import STATUSES, initialize_cases, update_case
from data_generator import generate_synthetic_transactions
from data_quality import check_data_quality
from ml_model import add_ml_scores
from model_metrics import calculate_metrics
from risk_engine import RISK_COLORS, analyze_dataframe

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")

RISK_ORDER = ["Low", "Medium", "High", "Critical"]

TEXT = {
    "ru": {
        "title": "FinGuard AI", "subtitle": "Интеллектуальная explainable anti-fraud платформа для банков и финтеха",
        "nav_overview": "Обзор", "nav_data": "Данные", "nav_alerts": "Alert Center", "nav_analytics": "Аналитика",
        "nav_model": "Модель и метрики", "nav_settings": "Настройки",
        "source": "Источник данных", "demo": "Сгенерировать демо-датасет", "upload": "Загрузить CSV / Excel",
        "settings": "Настройки", "language": "Язык интерфейса", "theme": "Светлая тема",
        "filters": "Фильтры", "risk": "Категория риска", "region": "Регион", "client": "Клиент", "period": "Период",
        "export": "Экспорт", "csv": "Скачать CSV", "pdf": "Скачать PDF отчет",
        "transactions": "Транзакций", "turnover": "Оборот", "avg": "Средний риск", "critical": "Critical операции",
        "share": "Доля Critical", "prevented": "Потенциально предотвращенный ущерб",
        "distribution": "Распределение риска", "timeline": "Risk score во времени", "drivers": "Факторы риска",
        "region_chart": "Риск по регионам", "registry": "Реестр транзакций", "details": "Карточка операции",
        "reasons": "Причины риска", "breakdown": "Детализация risk score",
        "upload_title": "Загрузите данные для анализа", "upload_hint": "Поддерживаются CSV, XLSX и XLS",
        "empty": "Сгенерируйте демо-датасет или загрузите файл, чтобы начать.",
        "no_data": "По выбранным фильтрам данных нет.",
        "new_device": "Новое устройство", "new_recipient": "Новый получатель", "date": "Дата и время",
        "amount": "Сумма", "score": "Risk score", "channel": "Канал", "region_label": "Регион",
        "yes": "Да", "no": "Нет", "ready": "Анализ выполнен", "error": "Ошибка", "live": "Система онлайн",
        "quality_title": "Проверка качества данных", "quality_status": "Статус", "quality_rows": "Строк",
        "quality_issues": "Обнаруженные проблемы", "quality_ok": "Проблем не обнаружено",
        "alert_center_title": "Alert Center — операции повышенного риска",
        "alert_priority": "Приоритет", "alert_status": "Статус кейса", "alert_comment": "Комментарий аналитика",
        "alert_save": "Сохранить статус", "alert_saved": "Статус обновлен",
        "model_title": "Качество модели (на синтетических данных)",
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC",
        "confusion": "Confusion Matrix", "metrics_unavailable": "Метрики недоступны: нет колонки fraud_label.",
        "final_score": "Итоговый score (rule + ML)", "ml_score": "ML anomaly score", "legacy_score": "Rule-based score",
        "fraud_type": "Тип сценария", "ai_tip_title": "Рекомендации аналитику",
        "tip_new_device": "Свяжитесь с клиентом для подтверждения операции с нового устройства.",
        "tip_new_recipient": "Проверьте историю переводов новому получателю перед подтверждением.",
        "tip_region": "Уточните у клиента, находится ли он в указанном регионе.",
        "tip_amount": "Сопоставьте сумму операции с обычным поведением клиента.",
        "tip_time": "Оцените вероятность мошенничества с учетом ночного времени операции.",
        "tip_frequency": "Проверьте, не является ли клиент жертвой автоматизированной атаки.",
        "tip_default": "Явных признаков риска не выявлено, дополнительных действий не требуется.",
        "demo_note": "Все данные являются синтетическими и используются только для демонстрации.",
    },
    "kk": {
        "title": "FinGuard AI", "subtitle": "Банктер мен финтех үшін интеллектуалды explainable anti-fraud платформасы",
        "nav_overview": "Шолу", "nav_data": "Деректер", "nav_alerts": "Alert Center", "nav_analytics": "Аналитика",
        "nav_model": "Модель және метрикалар", "nav_settings": "Баптаулар",
        "source": "Деректер көзі", "demo": "Демо-деректерді жасау", "upload": "CSV / Excel жүктеу",
        "settings": "Баптаулар", "language": "Интерфейс тілі", "theme": "Жарық тақырып",
        "filters": "Сүзгілер", "risk": "Тәуекел санаты", "region": "Аймақ", "client": "Клиент", "period": "Кезең",
        "export": "Экспорт", "csv": "CSV жүктеу", "pdf": "PDF есебін жүктеу",
        "transactions": "Транзакциялар", "turnover": "Айналым", "avg": "Орташа тәуекел", "critical": "Critical операциялар",
        "share": "Critical үлесі", "prevented": "Потенциалды сақталған ұйым",
        "distribution": "Тәуекел бөлінісі", "timeline": "Уақыт бойынша risk score", "drivers": "Тәуекел факторлары",
        "region_chart": "Аймақтар бойынша тәуекел", "registry": "Транзакциялар тізілімі", "details": "Операция карточкасы",
        "reasons": "Тәуекел себептері", "breakdown": "Risk score бөлінісі",
        "upload_title": "Талдау үшін деректерді жүктеңіз", "upload_hint": "CSV, XLSX және XLS қолданылады",
        "empty": "Бастау үшін демо жасаңыз немесе файл жүктеңіз.",
        "no_data": "Таңдалған сүзгілер бойынша деректер жоқ.",
        "new_device": "Жаңа құрылғы", "new_recipient": "Жаңа алушы", "date": "Күні мен уақыты",
        "amount": "Сома", "score": "Risk score", "channel": "Арна", "region_label": "Аймақ",
        "yes": "Иә", "no": "Жоқ", "ready": "Талдау аяқталды", "error": "Қате", "live": "Жүйе онлайн",
        "quality_title": "Деректер сапасын тексеру", "quality_status": "Мәртебе", "quality_rows": "Жолдар",
        "quality_issues": "Анықталған мәселелер", "quality_ok": "Мәселелер табылмады",
        "alert_center_title": "Alert Center — жоғары тәуекелді операциялар",
        "alert_priority": "Басымдық", "alert_status": "Кейс мәртебесі", "alert_comment": "Аналитик түсініктемесі",
        "alert_save": "Мәртебені сақтау", "alert_saved": "Мәртебе жаңартылды",
        "model_title": "Модель сапасы (синтетикалық деректерде)",
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC",
        "confusion": "Confusion Matrix", "metrics_unavailable": "Метрикалар қолжетімсіз: fraud_label бағаны жоқ.",
        "final_score": "Қорытынды score (rule + ML)", "ml_score": "ML anomaly score", "legacy_score": "Rule-based score",
        "fraud_type": "Сценарий түрі", "ai_tip_title": "Аналитикке ұсыныстар",
        "tip_new_device": "Жаңа құрылғыдан операцияны растау үшін клиентпен байланысыңыз.",
        "tip_new_recipient": "Растаудан бұрын жаңа алушыға аударым тарихын тексеріңіз.",
        "tip_region": "Клиенттен көрсетілген аймақта екенін нақтылаңыз.",
        "tip_amount": "Операция сомасын клиенттің әдеттегі мінез-құлқымен салыстырыңыз.",
        "tip_time": "Операцияның түнгі уақытын ескере отырып алаяқтық ықтималдығын бағалаңыз.",
        "tip_frequency": "Клиент автоматтандырылған шабуылдың құрбаны емес пе екенін тексеріңіз.",
        "tip_default": "Тәуекел белгілері анықталмады, қосымша әрекет қажет емес.",
        "demo_note": "Барлық деректер синтетикалық және тек демонстрация үшін пайдаланылады.",
    },
    "en": {
        "title": "FinGuard AI", "subtitle": "Explainable AI anti-fraud platform for banks and fintech",
        "nav_overview": "Overview", "nav_data": "Data", "nav_alerts": "Alert Center", "nav_analytics": "Analytics",
        "nav_model": "Model & Metrics", "nav_settings": "Settings",
        "source": "Data source", "demo": "Generate demo dataset", "upload": "Upload CSV / Excel",
        "settings": "Settings", "language": "Interface language", "theme": "Light theme",
        "filters": "Filters", "risk": "Risk category", "region": "Region", "client": "Client", "period": "Period",
        "export": "Export", "csv": "Download CSV", "pdf": "Download PDF report",
        "transactions": "Transactions", "turnover": "Turnover", "avg": "Average risk", "critical": "Critical operations",
        "share": "Critical share", "prevented": "Potential loss prevented",
        "distribution": "Risk distribution", "timeline": "Risk score over time", "drivers": "Risk drivers",
        "region_chart": "Risk by region", "registry": "Transaction registry", "details": "Transaction details",
        "reasons": "Risk reasons", "breakdown": "Risk score breakdown",
        "upload_title": "Upload data for analysis", "upload_hint": "CSV, XLSX and XLS are supported",
        "empty": "Generate a demo dataset or upload a file to start.",
        "no_data": "No data for selected filters.",
        "new_device": "New device", "new_recipient": "New recipient", "date": "Date and time",
        "amount": "Amount", "score": "Risk score", "channel": "Channel", "region_label": "Region",
        "yes": "Yes", "no": "No", "ready": "Analysis completed", "error": "Error", "live": "System online",
        "quality_title": "Data quality check", "quality_status": "Status", "quality_rows": "Rows",
        "quality_issues": "Detected issues", "quality_ok": "No issues detected",
        "alert_center_title": "Alert Center — high risk operations",
        "alert_priority": "Priority", "alert_status": "Case status", "alert_comment": "Analyst comment",
        "alert_save": "Save status", "alert_saved": "Status updated",
        "model_title": "Model quality (synthetic data)",
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC",
        "confusion": "Confusion Matrix", "metrics_unavailable": "Metrics unavailable: fraud_label column missing.",
        "final_score": "Final score (rule + ML)", "ml_score": "ML anomaly score", "legacy_score": "Rule-based score",
        "fraud_type": "Scenario type", "ai_tip_title": "Analyst recommendations",
        "tip_new_device": "Contact the client to confirm the operation from a new device.",
        "tip_new_recipient": "Review transfer history to the new recipient before confirming.",
        "tip_region": "Verify with the client whether they are in the stated region.",
        "tip_amount": "Compare the transaction amount with the client's usual behavior.",
        "tip_time": "Assess fraud likelihood considering the night-time transaction.",
        "tip_frequency": "Check whether the client may be a victim of an automated attack.",
        "tip_default": "No clear risk indicators found, no further action required.",
        "demo_note": "All data is synthetic and used for demonstration purposes only.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "ru"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "page" not in st.session_state:
    st.session_state.page = "overview"


def t(key: str) -> str:
    return TEXT[st.session_state.lang].get(key, key)


def apply_theme() -> None:
    if st.session_state.theme == "dark":
        css = """
        <style>
        [data-testid='stAppViewContainer']{background:#07111f;color:#f4fbff}
        [data-testid='stSidebar']{background:#0d1b2b}
        [data-testid='stHeader']{background:#07111f}
        .hero{background:linear-gradient(135deg,#102d4d,#0e5a62);border:1px solid #23758a;padding:22px;border-radius:18px;margin-bottom:16px}
        .hero h1{margin:0;color:#f4fbff}.hero p{color:#a8d9e4;margin:5px 0 0}
        [data-testid='stMetric']{background:#10243a;border:1px solid #24445b;border-radius:12px;padding:10px}
        .note{background:#12243a;border-left:3px solid #30d5c8;padding:8px 12px;border-radius:6px;color:#a8d9e4;font-size:13px}
        </style>
        """
    else:
        css = """
        <style>
        [data-testid='stAppViewContainer']{background:#f3f8fb;color:#123047}
        [data-testid='stSidebar']{background:#e8f2f6}
        [data-testid='stHeader']{background:#f3f8fb}
        .hero{background:linear-gradient(135deg,#dff4fa,#d8f1e8);border:1px solid #8ecfd0;padding:22px;border-radius:18px;margin-bottom:16px}
        .hero h1{margin:0;color:#123047}.hero p{color:#356579;margin:5px 0 0}
        [data-testid='stMetric']{background:#fff;border:1px solid #d7e7ee;border-radius:12px;padding:10px}
        .note{background:#e7f6f5;border-left:3px solid #14a99b;padding:8px 12px;border-radius:6px;color:#14615a;font-size:13px}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp1251")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Поддерживаются только CSV, XLSX и XLS.")


def run_pipeline(raw: pd.DataFrame) -> None:
    st.session_state.raw_df = raw
    analyzed = analyze_dataframe(raw)
    analyzed = add_ml_scores(analyzed)
    analyzed = initialize_cases(analyzed)
    st.session_state.df = analyzed


def get_reason_tips(reasons: list[str]) -> list[str]:
    mapping = {
        "устройства": t("tip_new_device"), "device": t("tip_new_device"),
        "получателю": t("tip_new_recipient"), "recipient": t("tip_new_recipient"),
        "регион": t("tip_region"), "region": t("tip_region"),
        "Сумма": t("tip_amount"), "amount": t("tip_amount"), "Сома": t("tip_amount"),
        "ночью": t("tip_time"), "ночное": t("tip_time"), "night": t("tip_time"), "түнгі": t("tip_time"),
        "частот": t("tip_frequency"), "frequency": t("tip_frequency"), "жиілік": t("tip_frequency"),
    }
    tips = []
    for reason in reasons:
        for key, tip in mapping.items():
            if key.lower() in str(reason).lower() and tip not in tips:
                tips.append(tip)
    return tips or [t("tip_default")]


def style_table(data: pd.DataFrame):
    def mark(row):
        color = RISK_COLORS.get(row["risk_category"], "#ffffff")
        return [f"background-color:{color}35"] * len(row)

    columns = ["transaction_id", "client_id", "client_name", "datetime", "amount", "region", "fraud_type", "risk_score", "ml_anomaly_score", "final_score", "risk_category"]
    columns = [column for column in columns if column in data.columns]
    formats = {"amount": "{:,.0f}", "risk_score": "{:.1f}", "ml_anomaly_score": "{:.1f}", "final_score": "{:.1f}"}
    return data[columns].style.apply(mark, axis=1).format(formats)


def make_pdf(data: pd.DataFrame, metrics: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm, bottomMargin=10 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("FinGuard AI — Anti-Fraud Report", styles["Title"]), Spacer(1, 4 * mm)]

    total = len(data)
    turnover = data["amount"].sum() if total else 0
    critical = int((data["risk_category"] == "Critical").sum()) if total else 0
    prevented = data.loc[data["risk_category"].isin(["High", "Critical"]), "amount"].sum() if total else 0
    summary = f"Transactions: {total:,} | Turnover: {turnover:,.0f} KZT | Critical: {critical} | Potential loss prevented: {prevented:,.0f} KZT"
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 3 * mm))

    if metrics.get("available"):
        model_line = f"Accuracy: {metrics['accuracy']:.2f} | Precision: {metrics['precision']:.2f} | Recall: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f}"
        story.append(Paragraph(model_line, styles["Normal"]))
        story.append(Spacer(1, 4 * mm))

    columns = ["transaction_id", "client_id", "datetime", "amount", "region", "fraud_type", "final_score", "risk_category"]
    columns = [column for column in columns if column in data.columns]
    rows = [[column.replace("_", " ").title() for column in columns]]
    for _, row in data[columns].head(100).iterrows():
        rows.append([str(row[column])[:24] for column in columns])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b5d")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf6f8")]), ("PADDING", (0, 0), (-1, -1), 4)]))
    story.append(table)
    doc.build(story)
    return buffer.getvalue()


apply_theme()

with st.sidebar:
    st.markdown("## 🛡️ FinGuard AI")
    st.caption(t("subtitle"))

    language_options = {"Русский": "ru", "Қазақша": "kk", "English": "en"}
    selected_language = st.selectbox(t("language"), list(language_options), index=list(language_options.values()).index(st.session_state.lang))
    new_lang = language_options[selected_language]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    theme_label = "☀️ " + t("theme") if st.session_state.theme == "dark" else "🌙 Dark theme"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown(f'<div class="note">{t("demo_note")}</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader(t("source"))
    if st.button("⚡ " + t("demo"), use_container_width=True, type="primary"):
        try:
            run_pipeline(generate_synthetic_transactions(n=700, anomaly_rate=0.12))
            st.success(t("ready"))
        except Exception as exc:
            st.error(f"{t('error')}: {exc}")

    uploaded = st.file_uploader(t("upload"), type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            run_pipeline(read_uploaded_file(uploaded))
            st.success(t("ready"))
        except Exception as exc:
            st.error(f"{t('error')}: {exc}")

    st.divider()
    st.subheader(t("nav_overview"))
    nav_map = {"overview": t("nav_overview"), "data": t("nav_data"), "alerts": t("nav_alerts"), "analytics": t("nav_analytics"), "model": t("nav_model")}
    for page_key, page_label in nav_map.items():
        if st.button(page_label, use_container_width=True, type="primary" if st.session_state.page == page_key else "secondary"):
            st.session_state.page = page_key
            st.rerun()

    df = st.session_state.df
    if df is not None:
        st.divider()
        st.subheader(t("filters"))
        selected_risks = st.multiselect(t("risk"), RISK_ORDER, default=RISK_ORDER, key="risk_filter")
        selected_regions = st.multiselect(t("region"), sorted(df["region"].astype(str).unique()), default=sorted(df["region"].astype(str).unique()), key="region_filter")
        selected_clients = st.multiselect(t("client"), sorted(df["client_id"].astype(str).unique()), key="client_filter")
        min_date, max_date = df["datetime"].min().date(), df["datetime"].max().date()
        selected_period = st.date_input(t("period"), value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        selected_risks, selected_regions, selected_clients, selected_period = RISK_ORDER, [], [], None

st.markdown(f'<div class="hero"><h1>🛡️ {t("title")}</h1><p>{t("subtitle")}</p></div>', unsafe_allow_html=True)

if st.session_state.df is None:
    st.info(t("empty"))
    st.stop()

df = st.session_state.df
filtered = df[df["risk_category"].isin(selected_risks) & df["region"].astype(str).isin(selected_regions)].copy()
if selected_clients:
    filtered = filtered[filtered["client_id"].astype(str).isin(selected_clients)]
if isinstance(selected_period, (tuple, list)) and len(selected_period) == 2:
    filtered = filtered[(filtered["datetime"].dt.date >= selected_period[0]) & (filtered["datetime"].dt.date <= selected_period[1])]

if filtered.empty:
    st.warning(t("no_data"))
    st.stop()

count = len(filtered)
turnover = float(filtered["amount"].sum())
avg_risk = float(filtered["final_score"].mean())
critical_count = int((filtered["risk_category"] == "Critical").sum())
critical_share = critical_count / count * 100
prevented = float(filtered.loc[filtered["risk_category"].isin(["High", "Critical"]), "amount"].sum())

if st.session_state.page == "overview":
    kpis = st.columns(5)
    kpis[0].metric(t("transactions"), f"{count:,}")
    kpis[1].metric(t("turnover"), f"{turnover:,.0f} KZT")
    kpis[2].metric(t("avg"), f"{avg_risk:.1f}/100")
    kpis[3].metric(t("critical"), f"{critical_count:,}")
    kpis[4].metric(t("share"), f"{critical_share:.1f}%")
    st.metric(t("prevented"), f"{prevented:,.0f} KZT")

    left, right = st.columns(2)
    with left:
        distribution = filtered["risk_category"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        distribution.columns = ["risk_category", "count"]
        figure = px.bar(distribution, x="risk_category", y="count", color="risk_category", color_discrete_map=RISK_COLORS, title=t("distribution"))
        figure.update_layout(showlegend=False, height=340)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        timeline = filtered.sort_values("datetime")
        figure = px.scatter(timeline, x="datetime", y="final_score", color="risk_category", color_discrete_map=RISK_COLORS, hover_data=["transaction_id", "client_id", "amount", "fraud_type"], title=t("timeline"))
        figure.update_layout(height=340)
        st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        driver_counts = {
            t("amount"): int((filtered["amount"] > filtered["amount"].quantile(.85)).sum()),
            t("new_device"): int(filtered["device_new"].sum()),
            t("new_recipient"): int(filtered["recipient_new"].sum()),
            t("region_label"): int((filtered["region"].astype(str) != filtered.get("home_region", filtered["region"]).astype(str)).sum()),
        }
        drivers = pd.DataFrame({"driver": list(driver_counts), "count": list(driver_counts.values())}).sort_values("count", ascending=False)
        figure = px.bar(drivers, x="count", y="driver", orientation="h", color="count", title=t("drivers"), color_continuous_scale="Tealgrn")
        figure.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        region_risk = filtered.groupby("region", as_index=False)["final_score"].mean().sort_values("final_score", ascending=False)
        figure = px.bar(region_risk, x="region", y="final_score", color="final_score", title=t("region_chart"), color_continuous_scale="RdYlGn_r")
        figure.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)

elif st.session_state.page == "data":
    st.subheader(t("upload_title"))
    st.caption(t("upload_hint"))
    quality = check_data_quality(st.session_state.raw_df)
    q1, q2, q3 = st.columns(3)
    q1.metric(t("quality_rows"), f"{quality['rows']:,}")
    q2.metric(t("quality_status"), quality["status"])
    q3.metric("Duplicates", quality["duplicate_ids"])
    st.markdown(f"**{t('quality_issues')}:**")
    if quality["issues"]:
        for issue in quality["issues"]:
            st.markdown(f"- ⚠ {issue}")
    else:
        st.success(t("quality_ok"))
    st.dataframe(st.session_state.raw_df.head(50), use_container_width=True, height=450)

elif st.session_state.page == "alerts":
    st.subheader(t("alert_center_title"))
    alerts = filtered[filtered["risk_category"].isin(["High", "Critical"])].sort_values("final_score", ascending=False)
    if alerts.empty:
        st.info(t("no_data"))
    else:
        for _, alert in alerts.head(30).iterrows():
            with st.expander(f"{alert['transaction_id']} | {t('score')}: {alert['final_score']:.1f} | {alert['fraud_type']} | {alert['case_status']}"):
                c1, c2 = st.columns([1.6, 1])
                with c1:
                    st.markdown(f"**{t('client')}:** {alert.get('client_name', '-')} ({alert['client_id']})")
                    st.markdown(f"**{t('date')}:** {alert['datetime']}")
                    st.markdown(f"**{t('amount')}:** {alert['amount']:,.0f} {alert.get('currency', 'KZT')}")
                    st.markdown(f"**{t('region_label')}:** {alert['region']}")
                    st.markdown(f"**{t('fraud_type')}:** {alert['fraud_type']}")
                    st.markdown(f"**{t('reasons')}:**")
                    for reason in alert["risk_reasons"]:
                        st.markdown(f"- {reason}")
                    st.markdown(f"**{t('ai_tip_title')}:**")
                    for tip in get_reason_tips(alert["risk_reasons"]):
                        st.markdown(f"- 💡 {tip}")
                with c2:
                    status = st.selectbox(t("alert_status"), STATUSES, index=STATUSES.index(alert["case_status"]) if alert["case_status"] in STATUSES else 0, key=f"status_{alert['transaction_id']}")
                    comment = st.text_area(t("alert_comment"), value=alert.get("case_comment", ""), key=f"comment_{alert['transaction_id']}")
                    if st.button(t("alert_save"), key=f"save_{alert['transaction_id']}"):
                        st.session_state.df = update_case(st.session_state.df, alert["transaction_id"], status, comment)
                        st.success(t("alert_saved"))
                        st.rerun()

elif st.session_state.page == "analytics":
    st.subheader(t("registry"))
    st.dataframe(style_table(filtered), use_container_width=True, height=430)

    export_data = filtered.copy()
    export_data["risk_reasons"] = export_data["risk_reasons"].apply(lambda values: " | ".join(values))
    metrics = calculate_metrics(filtered)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ " + t("csv"), export_data.to_csv(index=False).encode("utf-8-sig"), "finguard_report.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("🧾 " + t("pdf"), make_pdf(filtered, metrics), "finguard_report.pdf", "application/pdf", use_container_width=True)

    st.subheader(t("details"))
    selected_tx = st.selectbox("Transaction", filtered["transaction_id"].tolist())
    tx = filtered[filtered["transaction_id"] == selected_tx].iloc[0]
    info, gauge = st.columns([1.7, 1])
    with info:
        st.markdown(f"**{t('client')}:** {tx.get('client_name', '-')} ({tx['client_id']})")
        st.markdown(f"**{t('date')}:** {tx['datetime']}")
        st.markdown(f"**{t('amount')}:** {tx['amount']:,.0f} {tx.get('currency', 'KZT')}")
        st.markdown(f"**{t('region_label')}:** {tx['region']} | **{t('channel')}:** {tx.get('channel', '-')}")
        st.markdown(f"**{t('new_device')}:** {t('yes') if tx['device_new'] else t('no')} | **{t('new_recipient')}:** {t('yes') if tx['recipient_new'] else t('no')}")
        st.markdown(f"**{t('fraud_type')}:** {tx.get('fraud_type', '-')}")
        st.markdown(f"**{t('legacy_score')}:** {tx['risk_score']:.1f} | **{t('ml_score')}:** {tx['ml_anomaly_score']:.1f} | **{t('final_score')}:** {tx['final_score']:.1f}")
        st.markdown(f"**{t('breakdown')}:**")
        for label, key in [(t("amount"), "amount_score"), (t("date"), "time_score"), (t("new_device"), "device_score"), (t("new_recipient"), "recipient_score"), (t("region_label"), "region_score"), ("Frequency", "frequency_score")]:
            st.progress(min(1.0, float(tx.get(key, 0)) / 25), text=f"{label}: {tx.get(key, 0):.1f}")
        st.markdown(f"**{t('reasons')}:**")
        for reason in tx["risk_reasons"]:
            st.markdown(f"- {reason}")
        st.markdown(f"**{t('ai_tip_title')}:**")
        for tip in get_reason_tips(tx["risk_reasons"]):
            st.markdown(f"- 💡 {tip}")
    with gauge:
        figure = go.Figure(go.Indicator(mode="gauge+number", value=float(tx["final_score"]), title={"text": t("final_score")}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS.get(tx["risk_category"], "#167d8d")}, "steps": [{"range": [0, 25], "color": "#d9f5e5"}, {"range": [25, 50], "color": "#fff3c4"}, {"range": [50, 75], "color": "#ffe0c2"}, {"range": [75, 100], "color": "#ffd4d4"}]}))
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(figure, use_container_width=True)

elif st.session_state.page == "model":
    st.subheader(t("model_title"))
    metrics = calculate_metrics(filtered)
    if not metrics.get("available"):
        st.warning(t("metrics_unavailable"))
    else:
        m = st.columns(5)
        m[0].metric(t("accuracy"), f"{metrics['accuracy']:.2f}")
        m[1].metric(t("precision"), f"{metrics['precision']:.2f}")
        m[2].metric(t("recall"), f"{metrics['recall']:.2f}")
        m[3].metric(t("f1"), f"{metrics['f1']:.2f}")
        m[4].metric(t("roc_auc"), f"{metrics['roc_auc']:.2f}" if metrics["roc_auc"] else "-")
        st.markdown(f"**{t('confusion')}:**")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual: Normal", "Actual: Fraud"], columns=["Predicted: Normal", "Predicted: Fraud"])
        st.dataframe(cm, use_container_width=True)
