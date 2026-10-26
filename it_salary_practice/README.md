# it_salary_practice

Учебная практика: использование метода главных компонент (PCA) для определения
основных параметров, распределяющих заработок ИТ-специалиста.

Данные: Stack Overflow Developer Survey 2025, лицензия ODbL
(https://survey.stackoverflow.co/2025). Сырые файлы и база в репозиторий
не входят, их создают скрипты ниже.

## Окружение

    uv venv --python 3.12 .venv
    uv pip install -r requirements.txt

## Запуск по шагам

    python src/collect.py     # загрузка набора, MD5, data/raw/meta.json
    python src/verify.py      # сверка с опубликованными итогами опроса
    python src/load_db.py     # SQLite: db/survey.db по sql/01_schema.sql
    python src/queries.py     # запросы sql/02_queries.sql -> reports/tables
    python src/describe.py    # описательная статистика, пропуски, частоты
    python src/visualize.py   # графики раздела 2 -> reports/figures
    python src/train.py       # PCA, модели, метрики, графики раздела 3
    python -m pytest          # тесты

Диаграмма классов: `dot -Tpng -Gdpi=200 reports/uml/classes.dot -o reports/figures/19_classes.png`

## Структура

    sql/        схема БД и запросы
    src/        программы (features.py и models.py — признаки и модели)
    tests/      тесты pytest
    reports/    таблицы (CSV), графики (PNG), UML
