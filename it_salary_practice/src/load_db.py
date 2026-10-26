"""Развёртывание SQLite-базы: создание схемы и загрузка анкет из CSV."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "so_survey_2025.csv"
DB = ROOT / "db" / "survey.db"
SCHEMA = ROOT / "sql" / "01_schema.sql"

# числовые коды для порядковых ответов (хранятся в справочниках БД)
AGE = {"18-24 years old": 21, "25-34 years old": 29.5, "35-44 years old": 39.5,
       "45-54 years old": 49.5, "55-64 years old": 59.5, "65 years or older": 67}
EDU = {"Primary/elementary school": 0, "Secondary school": 1,
       "Some college/university study without earning a degree": 2,
       "Associate degree": 2, "Bachelor’s degree": 3, "Master’s degree": 4,
       "Professional degree": 5}
ORG = {"Just me - I am a freelancer, sole proprietor, etc.": 1,
       "Less than 20 employees": 10, "20 to 99 employees": 60,
       "100 to 499 employees": 300, "500 to 999 employees": 750,
       "1,000 to 4,999 employees": 3000, "5,000 to 9,999 employees": 7500,
       "10,000 or more employees": 15000}
REMOTE = {"In-person": 0, "Hybrid (some remote, leans heavy to in-person)": 1,
          "Your choice (very flexible, you can come in when you want or just as needed)": 2,
          "Hybrid (some in-person, leans heavy to flexibility)": 2, "Remote": 3}


def edu_rank(name):
    # в анкете после названия идут примеры в скобках, поэтому сравниваем по началу строки
    for key, rank in EDU.items():
        if name.startswith(key):
            return rank
    return None


def fill_lookup(con, table, values, code=None):
    """Заполняет справочник и возвращает словарь «текст ответа -> id»."""
    names = sorted(v for v in pd.unique(values) if isinstance(v, str))
    rows = [(i, name) + ((code(name),) if code else ()) for i, name in enumerate(names, start=1)]
    marks = ", ".join("?" * len(rows[0]))
    con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
    return {name: i for i, name, *_ in rows}


def main():
    df = pd.read_csv(RAW, low_memory=False)
    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA.read_text(encoding="utf-8"))

    ids = {
        "country": fill_lookup(con, "country", df["Country"]),
        "age": fill_lookup(con, "age_group", df["Age"], AGE.get),
        "edu": fill_lookup(con, "ed_level", df["EdLevel"], edu_rank),
        "org": fill_lookup(con, "org_size", df["OrgSize"], ORG.get),
        "remote": fill_lookup(con, "remote_mode", df["RemoteWork"], REMOTE.get),
        "dev": fill_lookup(con, "dev_type", df["DevType"]),
    }
    langs = df["LanguageHaveWorkedWith"].dropna().str.split(";").explode()
    ids["lang"] = fill_lookup(con, "language", langs)

    resp = pd.DataFrame({
        "respondent_id": df["ResponseId"],
        "main_branch": df["MainBranch"],
        "employment": df["Employment"],
        "age_group_id": df["Age"].map(ids["age"]),
        "ed_level_id": df["EdLevel"].map(ids["edu"]),
        "country_id": df["Country"].map(ids["country"]),
        "dev_type_id": df["DevType"].map(ids["dev"]),
        "org_size_id": df["OrgSize"].map(ids["org"]),
        "remote_mode_id": df["RemoteWork"].map(ids["remote"]),
        "is_manager": df["ICorPM"].map({"People manager": 1, "Individual contributor": 0}),
        "work_exp": df["WorkExp"],
        "years_code": df["YearsCode"],
        "comp_total": df["CompTotal"],
        "currency": df["Currency"],
        "comp_usd": df["ConvertedCompYearly"],
    })
    int_cols = [c for c in resp.columns if c.endswith("_id")] + ["is_manager"]
    resp[int_cols] = resp[int_cols].astype("Int64")
    # пропуски pandas (NaN, <NA>) превращаем в NULL
    rows = resp.astype(object).where(resp.notna(), None).itertuples(index=False, name=None)
    con.executemany(f"INSERT INTO respondent ({', '.join(resp.columns)}) "
                    f"VALUES ({', '.join('?' * resp.shape[1])})", rows)

    links = langs.reset_index().rename(columns={"index": "row", "LanguageHaveWorkedWith": "lang"})
    links["respondent_id"] = df.loc[links["row"], "ResponseId"].to_numpy()
    links["language_id"] = links["lang"].map(ids["lang"])
    links = links.drop_duplicates(["respondent_id", "language_id"])
    con.executemany("INSERT INTO respondent_language VALUES (?, ?)",
                    links[["respondent_id", "language_id"]].itertuples(index=False, name=None))
    con.commit()

    for table in ["respondent", "respondent_language", "country", "age_group", "ed_level",
                  "org_size", "remote_mode", "dev_type", "language"]:
        print(f"{table:20s} {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>7}")
    con.close()


if __name__ == "__main__":
    main()
