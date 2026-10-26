"""Первичный описательный анализ: воронка отбора, пропуски, статистики, частоты."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from features import DB, PRO, FeatureBuilder, clean_experience, load_sample, remove_outliers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tables"

NUMERIC = {"comp_usd": "Годовой доход, USD", "ln_salary": "ln(доход)",
           "work_exp": "Стаж работы, лет", "years_code": "Стаж программирования, лет",
           "age_mid": "Возраст (середина интервала), лет", "ed_rank": "Уровень образования, код 0–5",
           "org_employees": "Размер компании, чел.", "remote_rank": "Формат работы, код 0–3",
           "n_languages": "Число языков, шт."}
CATEGORICAL = ["age_group", "ed_level", "org_size", "remote_work", "dev_type", "country"]


def funnel(con):
    steps = [
        ("Все анкеты", "1 = 1"),
        ("Профессиональные разработчики", f"main_branch = '{PRO}'"),
        ("…работают по найму (Employed)", f"main_branch = '{PRO}' AND employment = 'Employed'"),
        ("…указали доход", f"main_branch = '{PRO}' AND employment = 'Employed' AND comp_usd IS NOT NULL"),
    ]
    return [(name, con.execute(f"SELECT COUNT(*) FROM respondent WHERE {cond}").fetchone()[0])
            for name, cond in steps]


def describe_numeric(df):
    rows = []
    for col, label in NUMERIC.items():
        s = df[col]
        mode = s.mode()
        rows.append({"variable": label, "n": int(s.count()), "missing": int(s.isna().sum()),
                     "min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
                     "mode": mode.iloc[0] if len(mode) else np.nan,
                     "var": s.var(), "std": s.std(), "skew": s.skew()})
    return pd.DataFrame(rows)


def frequencies(df, col, top=12):
    counts = df[col].fillna("(пропуск)").value_counts()
    if len(counts) > top:
        counts = pd.concat([counts.head(top), pd.Series({"Прочие": counts.iloc[top:].sum()})])
    table = counts.rename("n").to_frame()
    table["share_pct"] = (table["n"] / table["n"].sum() * 100).round(1)
    return table.rename_axis(col).reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        steps = funnel(con)
        full = pd.read_sql_query("SELECT * FROM v_respondent", con)

    sample = load_sample()
    kept, report = remove_outliers(sample)
    kept, n_blanked = clean_experience(kept)
    profile = FeatureBuilder().profile(kept)
    kept = kept.assign(ln_salary=np.log(kept["comp_usd"]), n_languages=profile["n_languages"])
    steps += [("…доход в границах 1 000–1 000 000 USD", report["n_input"] - report["below_bound"] - report["above_bound"]),
              ("…без выбросов внутри страны", report["n_output"]),
              ("…без пропусков в признаках профиля", len(profile.dropna()))]
    pd.DataFrame(steps, columns=["step", "n"]).to_csv(OUT / "funnel.csv", index=False)

    cols = ["age_group", "ed_level", "country", "dev_type", "org_size", "remote_work",
            "is_manager", "work_exp", "years_code", "comp_usd", "languages"]
    missing = pd.DataFrame({"all_pct": full[cols].isna().mean() * 100,
                            "sample_pct": sample[cols].isna().mean() * 100}).round(1)
    missing.rename_axis("column").reset_index().to_csv(OUT / "missing.csv", index=False)

    stats = describe_numeric(kept)
    stats.to_csv(OUT / "numeric_stats.csv", index=False)
    for col in CATEGORICAL:
        frequencies(kept, col).to_csv(OUT / f"freq_{col}.csv", index=False)
    manager = kept["is_manager"].map({1: "Руководитель", 0: "Исполнитель"})
    frequencies(manager.to_frame("role"), "role").to_csv(OUT / "freq_role.csv", index=False)

    summary = {"funnel": dict(steps), "outliers": report,
               "countries_in_sample": int(kept["country"].nunique()),
               "duplicate_ids": int(full["respondent_id"].duplicated().sum()),
               "experience_blanked": n_blanked,
               "exp_gt_code_in_sample": int((kept["work_exp"] > kept["years_code"]).sum())}
    (OUT / "describe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    pd.set_option("display.width", 160)
    print(pd.DataFrame(steps, columns=["шаг", "анкет"]).to_string(index=False))
    print(missing.to_string())
    print(stats.round(2).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
