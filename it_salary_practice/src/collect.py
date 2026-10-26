"""Сбор данных: загрузка открытого набора Stack Overflow Developer Survey 2025.

Набор публикуется Stack Overflow под лицензией ODbL. Ссылка на CSV взята
из разметки страницы https://survey.stackoverflow.co (блок Dataset 2025).
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE_URL = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2025/"
# имя файла в репозитории Stack Overflow -> имя у нас
FILES = {"results.csv": "so_survey_2025.csv", "schema.csv": "so_schema_2025.csv"}
# хеш того же файла, скачанного в сентябре 2026 для курсовой (IT_Salary_PCA/data/raw/*.dvc)
KNOWN_MD5 = "7c8869f4cf7bb35135af0ab87d939e48"
CHUNK = 1 << 20


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    # пишем во временный файл, чтобы оборванная загрузка не выглядела как готовая
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while block := resp.read(CHUNK):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {target.name}: {done / total:6.1%}", end="")
    print()
    part.rename(target)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta = {"source": BASE_URL, "license": "ODbL 1.0",
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    for remote, local in FILES.items():
        target = RAW / local
        if not target.exists():
            download(BASE_URL + remote, target)
        meta["files"][local] = {"url": BASE_URL + remote, "bytes": target.stat().st_size,
                                "md5": md5sum(target)}

    survey = meta["files"]["so_survey_2025.csv"]
    survey["md5_matches_coursework"] = survey["md5"] == KNOWN_MD5
    df = pd.read_csv(RAW / "so_survey_2025.csv", low_memory=False)
    survey["rows"], survey["columns"] = df.shape
    schema = pd.read_csv(RAW / "so_schema_2025.csv")
    meta["files"]["so_schema_2025.csv"]["questions"] = int(schema["qid"].nunique())

    (RAW / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"анкет: {df.shape[0]}, столбцов: {df.shape[1]}")
    print(f"md5: {survey['md5']} (совпадает с курсовой: {survey['md5_matches_coursework']})")
    if not survey["md5_matches_coursework"]:
        sys.exit("файл отличается от ранее скачанного — проверьте источник")


if __name__ == "__main__":
    main()
