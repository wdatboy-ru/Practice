-- Запросы для извлечения агрегированной информации из БД survey.db.
-- Каждый запрос начинается со строки «-- name: ...», по ней src/queries.py
-- делит файл и сохраняет результаты в reports/tables/.

-- name: q1_counts
-- Общее число анкет и число анкет с указанным доходом
SELECT COUNT(*)                                    AS respondents,
       SUM(comp_usd IS NOT NULL)                   AS with_salary,
       ROUND(100.0 * SUM(comp_usd IS NOT NULL) / COUNT(*), 1) AS salary_share_pct
  FROM respondent;

-- name: q2_main_branch
-- Распределение респондентов по отношению к разработке
SELECT main_branch,
       COUNT(*)                                             AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)   AS share_pct
  FROM respondent
 GROUP BY main_branch
 ORDER BY n DESC;

-- name: q3_median_by_dev_type
-- Медиана дохода по роли (все респонденты с доходом). SQLite не имеет
-- функции MEDIAN, поэтому медиана считается через оконную нумерацию.
WITH ranked AS (
    SELECT d.name AS dev_type,
           r.comp_usd,
           ROW_NUMBER() OVER (PARTITION BY r.dev_type_id ORDER BY r.comp_usd) AS rn,
           COUNT(*)     OVER (PARTITION BY r.dev_type_id)                     AS cnt
      FROM respondent r
      JOIN dev_type d ON d.dev_type_id = r.dev_type_id
     WHERE r.comp_usd IS NOT NULL
)
SELECT dev_type, cnt AS n, AVG(comp_usd) AS median_usd
  FROM ranked
 WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
 GROUP BY dev_type, cnt
 ORDER BY median_usd DESC;

-- name: q4_top_countries
-- Страны с наибольшим числом работающих профессиональных разработчиков,
-- указавших доход
SELECT c.name                 AS country,
       COUNT(*)               AS n,
       ROUND(AVG(r.comp_usd)) AS mean_usd,
       MIN(r.comp_usd)        AS min_usd,
       MAX(r.comp_usd)        AS max_usd
  FROM respondent r
  JOIN country c ON c.country_id = r.country_id
 WHERE r.main_branch = 'I am a developer by profession'
   AND r.employment = 'Employed'
   AND r.comp_usd IS NOT NULL
 GROUP BY c.name
 ORDER BY n DESC
 LIMIT 15;

-- name: q5_languages
-- Популярность языков среди профессиональных разработчиков
SELECT l.name AS language,
       COUNT(*) AS users,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(DISTINCT rl2.respondent_id)
                                   FROM respondent_language rl2
                                   JOIN respondent r2 ON r2.respondent_id = rl2.respondent_id
                                  WHERE r2.main_branch = 'I am a developer by profession'), 1) AS share_pct
  FROM respondent_language rl
  JOIN language l   ON l.language_id = rl.language_id
  JOIN respondent r ON r.respondent_id = rl.respondent_id
 WHERE r.main_branch = 'I am a developer by profession'
 GROUP BY l.name
 ORDER BY users DESC
 LIMIT 15;

-- name: q6_education
-- Средний стаж и доход по уровню образования (только указавшие доход)
SELECT e.ed_rank,
       e.name                        AS ed_level,
       COUNT(*)                      AS n,
       ROUND(AVG(r.work_exp), 1)     AS mean_work_exp,
       ROUND(AVG(r.comp_usd))        AS mean_usd
  FROM respondent r
  JOIN ed_level e ON e.ed_level_id = r.ed_level_id
 WHERE r.comp_usd IS NOT NULL
 GROUP BY e.ed_level_id
 ORDER BY e.ed_rank IS NULL, e.ed_rank;

-- name: q7_remote_by_country
-- Доля полностью удалённой работы в крупнейших странах выборки
SELECT c.name AS country,
       COUNT(*) AS n,
       ROUND(100.0 * SUM(m.remote_rank = 3) / COUNT(*), 1) AS remote_pct,
       ROUND(100.0 * SUM(m.remote_rank = 0) / COUNT(*), 1) AS office_pct
  FROM respondent r
  JOIN country c     ON c.country_id = r.country_id
  JOIN remote_mode m ON m.remote_mode_id = r.remote_mode_id
 WHERE r.employment = 'Employed'
 GROUP BY c.name
HAVING COUNT(*) >= 500
 ORDER BY remote_pct DESC;

-- name: q8_quality
-- Логические противоречия в ответах: опыт работы больше стажа программирования
-- и подозрительно малый доход
SELECT SUM(work_exp > years_code)                       AS exp_gt_code,
       SUM(work_exp >= 50)                              AS exp_ge_50,
       SUM(comp_usd < 1000)                             AS salary_lt_1000,
       SUM(comp_usd > 1000000)                          AS salary_gt_1m
  FROM respondent
 WHERE main_branch = 'I am a developer by profession'
   AND employment = 'Employed';
