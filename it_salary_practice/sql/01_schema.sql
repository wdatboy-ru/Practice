-- Схема БД для анкет Stack Overflow Developer Survey 2025 (SQLite).
-- Повторяющиеся текстовые ответы вынесены в справочники, языки программирования
-- (множественный выбор) хранятся в отдельной таблице связи «многие ко многим».

PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS v_respondent;
DROP TABLE IF EXISTS respondent_language;
DROP TABLE IF EXISTS respondent;
DROP TABLE IF EXISTS language;
DROP TABLE IF EXISTS dev_type;
DROP TABLE IF EXISTS remote_mode;
DROP TABLE IF EXISTS org_size;
DROP TABLE IF EXISTS ed_level;
DROP TABLE IF EXISTS age_group;
DROP TABLE IF EXISTS country;

CREATE TABLE country (
    country_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE
);

-- midpoint: середина возрастного интервала, лет
CREATE TABLE age_group (
    age_group_id INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    midpoint     REAL
);

-- ed_rank: порядковый код уровня образования (0 — начальное, 5 — PhD и аналоги)
CREATE TABLE ed_level (
    ed_level_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    ed_rank     INTEGER
);

-- employees: типичная численность сотрудников для интервала
CREATE TABLE org_size (
    org_size_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    employees   INTEGER
);

-- remote_rank: 0 — только офис, 3 — полностью удалённо
CREATE TABLE remote_mode (
    remote_mode_id INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    remote_rank    INTEGER
);

CREATE TABLE dev_type (
    dev_type_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE language (
    language_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE respondent (
    respondent_id  INTEGER PRIMARY KEY,               -- ResponseId из анкеты
    main_branch    TEXT NOT NULL,
    employment     TEXT,
    age_group_id   INTEGER REFERENCES age_group (age_group_id),
    ed_level_id    INTEGER REFERENCES ed_level (ed_level_id),
    country_id     INTEGER REFERENCES country (country_id),
    dev_type_id    INTEGER REFERENCES dev_type (dev_type_id),
    org_size_id    INTEGER REFERENCES org_size (org_size_id),
    remote_mode_id INTEGER REFERENCES remote_mode (remote_mode_id),
    is_manager     INTEGER CHECK (is_manager IN (0, 1)),
    work_exp       REAL CHECK (work_exp >= 0),        -- стаж профессиональной работы, лет
    years_code     REAL CHECK (years_code >= 0),      -- стаж программирования, лет
    comp_total     REAL,                              -- доход в местной валюте
    currency       TEXT,
    comp_usd       REAL CHECK (comp_usd >= 0)         -- годовой доход, USD
);

CREATE TABLE respondent_language (
    respondent_id INTEGER NOT NULL REFERENCES respondent (respondent_id) ON DELETE CASCADE,
    language_id   INTEGER NOT NULL REFERENCES language (language_id),
    PRIMARY KEY (respondent_id, language_id)
);

CREATE INDEX ix_respondent_country ON respondent (country_id);
CREATE INDEX ix_respondent_dev_type ON respondent (dev_type_id);
CREATE INDEX ix_respondent_language_lang ON respondent_language (language_id);

-- Плоское представление: одна строка на респондента, справочники раскрыты,
-- языки собраны в строку через «;» (как в исходном CSV).
CREATE VIEW v_respondent AS
SELECT r.respondent_id,
       r.main_branch,
       r.employment,
       a.name         AS age_group,
       a.midpoint     AS age_mid,
       e.name         AS ed_level,
       e.ed_rank,
       c.name         AS country,
       d.name         AS dev_type,
       o.name         AS org_size,
       o.employees    AS org_employees,
       m.name         AS remote_work,
       m.remote_rank,
       r.is_manager,
       r.work_exp,
       r.years_code,
       r.comp_usd,
       (SELECT GROUP_CONCAT(l.name, ';')
          FROM respondent_language rl
          JOIN language l ON l.language_id = rl.language_id
         WHERE rl.respondent_id = r.respondent_id) AS languages
  FROM respondent r
  LEFT JOIN age_group a   ON a.age_group_id = r.age_group_id
  LEFT JOIN ed_level e    ON e.ed_level_id = r.ed_level_id
  LEFT JOIN country c     ON c.country_id = r.country_id
  LEFT JOIN dev_type d    ON d.dev_type_id = r.dev_type_id
  LEFT JOIN org_size o    ON o.org_size_id = r.org_size_id
  LEFT JOIN remote_mode m ON m.remote_mode_id = r.remote_mode_id;
