"""Отбор анкет, очистка дохода от выбросов и построение признаков."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db" / "survey.db"

PRO = "I am a developer by profession"
BASE_COUNTRY = "United States of America"
SALARY_BOUNDS = (1_000, 1_000_000)  # правдоподобный годовой доход, USD
MZ_THRESHOLD = 3.5                  # порог модифицированного z (Iglewicz, Hoaglin)
MIN_GROUP = 30                      # страны с меньшим числом анкет объединяются в одну группу
LANGS = {"lang_python": "Python", "lang_sql": "SQL", "lang_javascript": "JavaScript",
         "lang_typescript": "TypeScript", "lang_java": "Java", "lang_csharp": "C#",
         "lang_go": "Go", "lang_rust": "Rust", "lang_cpp": "C++", "lang_php": "PHP",
         "lang_kotlin": "Kotlin", "lang_bash": "Bash/Shell (all shells)"}

# верхняя граница возрастного интервала; для 65+ границы нет
AGE_UPPER = {"18-24 years old": 24, "25-34 years old": 34, "35-44 years old": 44,
             "45-54 years old": 54, "55-64 years old": 64}

SAMPLE_SQL = """
SELECT * FROM v_respondent
 WHERE main_branch = ? AND employment = 'Employed' AND comp_usd IS NOT NULL
"""


def load_sample(db=DB):
    """Работающие профессиональные разработчики, указавшие доход."""
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(SAMPLE_SQL, con, params=(PRO,))


def modified_z(values):
    """Модифицированный z-показатель: 0,6745 * (x - медиана) / MAD."""
    mad = (values - values.median()).abs().median()
    return 0.6745 * (values - values.median()) / (mad if mad > 0 else np.nan)


def remove_outliers(df, bounds=SALARY_BOUNDS, threshold=MZ_THRESHOLD, min_group=MIN_GROUP):
    """Удаляет неправдоподобный доход: сначала по абсолютным границам,
    затем по отклонению логарифма дохода от медианы своей страны."""
    low, high = bounds
    in_bounds = df[df["comp_usd"].between(low, high)].copy()
    sizes = in_bounds["country"].map(in_bounds["country"].value_counts())
    group = in_bounds["country"].where(sizes >= min_group, "Other")
    z = np.log(in_bounds["comp_usd"]).groupby(group).transform(modified_z)
    kept = in_bounds[~(z.abs() > threshold)]
    report = {"n_input": len(df),
              "below_bound": int((df["comp_usd"] < low).sum()),
              "above_bound": int((df["comp_usd"] > high).sum()),
              "country_outliers": int((z.abs() > threshold).sum()),
              "n_output": len(kept)}
    return kept, report


def clean_experience(df):
    """Логический контроль: стаж работы не может начаться раньше 14 лет,
    а программирование — раньше 5 лет. Невозможные значения заменяются пропуском."""
    upper = df["age_group"].map(AGE_UPPER)
    bad_work = df["work_exp"] > upper - 14
    bad_code = df["years_code"] > upper - 5
    cleaned = df.copy()
    cleaned.loc[bad_work, "work_exp"] = np.nan
    cleaned.loc[bad_code, "years_code"] = np.nan
    return cleaned, int(bad_work.sum() + bad_code.sum())


class FeatureBuilder:
    """Превращает строки v_respondent в числовые признаки для PCA и регрессии.

    profile() — 20 признаков профиля специалиста, countries() — фиктивные
    переменные страны. Список стран запоминается в fit() по обучающей выборке,
    чтобы тестовая выборка кодировалась тем же набором столбцов.
    """

    def __init__(self, min_country=50, base_country=BASE_COUNTRY):
        self.min_country = min_country
        self.base_country = base_country
        self.countries_ = None

    def fit(self, df):
        counts = df["country"].value_counts()
        frequent = counts[counts >= self.min_country].index
        self.countries_ = sorted(c for c in frequent if c != self.base_country)
        return self

    def profile(self, df):
        x = pd.DataFrame(index=df.index)
        x["work_exp"] = df["work_exp"]
        x["years_code"] = df["years_code"]
        x["age"] = df["age_mid"]
        x["edu_level"] = df["ed_rank"]
        x["org_size_log"] = np.log10(df["org_employees"])
        x["remote"] = df["remote_rank"]
        x["manager"] = df["is_manager"]
        langs = df["languages"].fillna("").str.split(";")
        known = df["languages"].notna()
        x["n_languages"] = langs.str.len().where(known)
        for name, lang in LANGS.items():
            x[name] = langs.apply(lambda items: float(lang in items)).where(known)
        return x.astype(float)

    def countries(self, df):
        if self.countries_ is None:
            raise RuntimeError("сначала вызовите fit()")
        country = df["country"]
        dummies = pd.DataFrame(index=df.index)
        for c in self.countries_:
            dummies[f"country_{c}"] = (country == c).astype(int)
        dummies["country_Other"] = (~country.isin(self.countries_ + [self.base_country])).astype(int)
        return dummies

    @staticmethod
    def target(df):
        """Целевая переменная — натуральный логарифм годового дохода."""
        return np.log(df["comp_usd"])
