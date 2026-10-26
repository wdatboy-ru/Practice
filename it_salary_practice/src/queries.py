"""Выполнение запросов из sql/02_queries.sql и сохранение результатов в CSV."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db" / "survey.db"
QUERIES = ROOT / "sql" / "02_queries.sql"
OUT = ROOT / "reports" / "tables"


def read_queries(path):
    """Делит SQL-файл на именованные запросы по строкам «-- name: ...»."""
    parts = re.split(r"^-- name: (\w+)\s*$", path.read_text(encoding="utf-8"), flags=re.M)
    return dict(zip(parts[1::2], (sql.strip() for sql in parts[2::2])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        for name, sql in read_queries(QUERIES).items():
            result = pd.read_sql_query(sql, con)
            result.to_csv(OUT / f"sql_{name}.csv", index=False)
            print(f"--- {name}: {len(result)} строк")
            print(result.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
