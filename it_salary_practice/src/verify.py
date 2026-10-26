"""Проверка достоверности: пересчёт показателей, опубликованных Stack Overflow.

Опубликованные значения выписаны вручную со страниц
https://survey.stackoverflow.co/2025/developers, /work и /methodology (октябрь 2026).
Если наш пересчёт по скачанному файлу совпадает с ними, значит файл тот же,
что использовали авторы опроса, и загрузка прошла без искажений.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "so_survey_2025.csv"
OUT = ROOT / "reports" / "tables" / "verify.csv"

US = "United States of America"
UK = "United Kingdom of Great Britain and Northern Ireland"

# (показатель, столбец, значение ответа, опубликовано в %, только профессионалы)
# график размера компании на сайте построен по профессиональным разработчикам
SHARES = [
    ("Профессиональные разработчики", "MainBranch", "I am a developer by profession", 76.2, False),
    ("Возраст 18–24", "Age", "18-24 years old", 18.7, False),
    ("Возраст 25–34", "Age", "25-34 years old", 33.6, False),
    ("Возраст 35–44", "Age", "35-44 years old", 26.9, False),
    ("Возраст 45–54", "Age", "45-54 years old", 12.8, False),
    ("Бакалавриат", "EdLevel", "Bachelor’s degree (B.A., B.S., B.Eng., etc.)", 42.1, False),
    ("Магистратура", "EdLevel", "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)", 26.2, False),
    ("Полностью удалённо", "RemoteWork", "Remote", 32.4, False),
    ("Только офис", "RemoteWork", "In-person", 17.9, False),
    ("Компания 20–99 чел.", "OrgSize", "20 to 99 employees", 20.8, True),
    ("Компания 10 000+ чел.", "OrgSize", "10,000 or more employees", 14.2, True),
]
# (показатель, страна или None для всех, роль, опубликованная медиана, USD)
MEDIANS = [
    ("Медиана: Senior executive", None, "Senior executive (C-suite, VP, etc.)", 139218),
    ("Медиана: Engineering manager", None, "Engineering manager", 130000),
    ("Медиана: Cloud infrastructure engineer", None, "Cloud infrastructure engineer", 103112.5),
    ("Медиана: Architect", None, "Architect, software or solutions", 102000),
    ("Медиана: AI/ML engineer", None, "AI/ML engineer", 89427),
    ("Медиана: DevOps engineer", None, "DevOps engineer or professional", 87011),
    ("США: Senior executive", US, "Senior executive (C-suite, VP, etc.)", 225000),
    ("США: Engineering manager", US, "Engineering manager", 200000),
    ("США: Developer, back-end", US, "Developer, back-end", 175000),
    ("Великобритания: Engineering manager", UK, "Engineering manager", 136141),
    ("Великобритания: Developer, full-stack", UK, "Developer, full-stack", 85428.5),
    ("Германия: Engineering manager", "Germany", "Engineering manager", 118000),
    ("Индия: Engineering manager", "India", "Engineering manager", 52000),
]


def main():
    df = pd.read_csv(RAW, low_memory=False)
    rows = [("Число стран", "шт.", 177, df["Country"].nunique()),
            ("Анкет с доходом", "шт.", 23928, int(df["ConvertedCompYearly"].notna().sum()))]
    pro = df[df["MainBranch"] == "I am a developer by profession"]
    for name, col, answer, published, pro_only in SHARES:
        ours = (pro if pro_only else df)[col].value_counts(normalize=True).get(answer, 0) * 100
        rows.append((name, "%", published, round(ours, 1)))
    paid = df[df["ConvertedCompYearly"].notna()]
    for name, country, role, published in MEDIANS:
        part = paid if country is None else paid[paid["Country"] == country]
        rows.append((name, "USD", published, part.loc[part["DevType"] == role, "ConvertedCompYearly"].median()))

    table = pd.DataFrame(rows, columns=["indicator", "unit", "published", "ours"])
    table["diff"] = table["ours"] - table["published"]
    table["rel_diff_pct"] = (table["diff"] / table["published"] * 100).round(2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    exact = (table["diff"].abs() < 0.051).sum()
    print(f"\nточное совпадение: {exact} из {len(table)}; "
          f"максимальное относительное расхождение {table['rel_diff_pct'].abs().max():.2f} %")


if __name__ == "__main__":
    main()
