"""Раздел 2: графики по очищенной выборке и таблицы с их числами."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import MIN_GROUP, FeatureBuilder, clean_experience, load_sample, remove_outliers
from plot_style import (COUNTRY_RU, DEV_RU, DIVERGING, EDU_RU, FEATURE_RU, GRID, INK, INK_2,
                        ORG_RU, REMOTE_RU, SEQUENTIAL, SERIES, fmt_dec, fmt_int, save, setup, usd_k)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
US, DE, IN = "United States of America", "Germany", "India"


def prepare():
    sample = load_sample()
    kept, _ = remove_outliers(sample)
    kept, _ = clean_experience(kept)
    kept = kept.assign(ln_salary=np.log(kept["comp_usd"]))
    return sample, kept, FeatureBuilder().profile(kept)


def missing_share():
    miss = pd.read_csv(TABLES / "missing.csv").set_index("column")
    names = {"age_group": "Возраст", "ed_level": "Образование", "country": "Страна",
             "dev_type": "Роль", "org_size": "Размер компании", "remote_work": "Формат работы",
             "is_manager": "Руководитель/исполнитель", "work_exp": "Стаж работы",
             "years_code": "Стаж программирования", "comp_usd": "Доход", "languages": "Языки"}
    miss = miss.rename(index=names).sort_values("all_pct")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(miss))
    ax.barh(y + 0.2, miss["all_pct"], height=0.38, color=SERIES[0], label="все анкеты (49 191)")
    ax.barh(y - 0.2, miss["sample_pct"], height=0.38, color=SERIES[1], label="выборка с доходом (16 891)")
    for yi, (a, s) in zip(y, miss[["all_pct", "sample_pct"]].to_numpy()):
        ax.text(a + 0.6, yi + 0.2, fmt_dec(a, 1), va="center", fontsize=7, color=INK_2)
        ax.text(s + 0.6, yi - 0.2, fmt_dec(s, 1), va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, miss.index)
    ax.set_xlabel("Доля пропусков, %")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "01_missing_share")


def salary_distribution(sample, df):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    raw = sample["comp_usd"]
    ax[0].hist(np.log10(raw), bins=60, color=SERIES[3], edgecolor="white", linewidth=0.4)
    ax[0].set_xlabel("lg(доход, USD)")
    ax[0].set_ylabel("Число анкет")
    ax[0].set_title(f"До очистки (n = {fmt_int(len(raw))})")
    ax[0].axvline(np.log10(1000), color=INK_2, ls="--", lw=1)
    ax[0].axvline(np.log10(1_000_000), color=INK_2, ls="--", lw=1)
    ax[1].hist(df["comp_usd"] / 1000, bins=60, color=SERIES[0], edgecolor="white", linewidth=0.4)
    ax[1].set_xlabel("Доход, тыс. USD")
    ax[1].set_title(f"После очистки, асимметрия {fmt_dec(df['comp_usd'].skew())}")
    ax[2].hist(df["ln_salary"], bins=60, color=SERIES[2], edgecolor="white", linewidth=0.4)
    ax[2].set_xlabel("ln(доход, USD)")
    ax[2].set_title(f"Логарифм дохода, асимметрия {fmt_dec(df['ln_salary'].skew())}")
    fig.tight_layout()
    save(fig, "02_salary_distribution")


def remote_pie(df):
    order = list(REMOTE_RU)
    counts = df["remote_work"].value_counts().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    wedges, _, autotexts = ax.pie(
        counts, colors=SERIES[:len(counts)], startangle=90, counterclock=False,
        autopct=lambda p: f"{fmt_dec(p, 1)} %", pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
    for t in autotexts:
        t.set_color(INK)
        t.set_fontsize(8)
    ax.legend(wedges, [f"{REMOTE_RU[k]} ({fmt_int(v)})" for k, v in counts.items()],
              loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Формат работы, n = {fmt_int(counts.sum())}")
    save(fig, "03_remote_pie")
    counts.rename(index=REMOTE_RU).to_csv(TABLES / "fig_remote.csv")


def country_bar(df):
    top = df["country"].value_counts().head(15).index
    stats = (df[df["country"].isin(top)].groupby("country")["comp_usd"]
             .agg(n="size", median="median").sort_values("median"))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = [f"{COUNTRY_RU.get(c, c)} ({fmt_int(n)})" for c, n in stats["n"].items()]
    ax.barh(labels, stats["median"] / 1000, color=SERIES[0], height=0.7)
    for i, v in enumerate(stats["median"]):
        ax.text(v / 1000 + 1.5, i, usd_k(v), va="center", fontsize=7.5, color=INK_2)
    overall = df["comp_usd"].median() / 1000
    ax.axvline(overall, color=INK_2, ls="--", lw=1)
    ax.text(overall + 1, len(stats) - 0.1, f"медиана выборки {fmt_int(round(overall))} тыс.",
            fontsize=7.5, color=INK_2)
    ax.set_ylim(-0.6, len(stats) + 0.4)
    ax.set_xlabel("Медианный годовой доход, тыс. USD")
    ax.grid(axis="y", visible=False)
    save(fig, "04_country_median")
    stats.assign(country_ru=[COUNTRY_RU.get(c, c) for c in stats.index]).to_csv(TABLES / "fig_country.csv")


def experience_line(df):
    edges = [1, 3, 5, 7, 10, 13, 16, 20, 25, 30, 41]
    mids = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
    groups = pd.cut(df["work_exp"], edges, right=False, labels=mids)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = []
    for name, part, color in [("Все страны", df, INK_2), ("США", df[df.country == US], SERIES[0]),
                              ("Германия", df[df.country == DE], SERIES[1]),
                              ("Индия", df[df.country == IN], SERIES[2])]:
        g = part.groupby(groups.loc[part.index], observed=True)["comp_usd"].agg(["size", "median"])
        g = g[g["size"] >= 20]
        style = {"ls": "--", "lw": 1.5} if name == "Все страны" else {}
        ax.plot(g.index.astype(float), g["median"] / 1000, marker="o", ms=4, color=color, label=name, **style)
        last = g.iloc[-1]
        ax.text(float(g.index[-1]) + 0.6, last["median"] / 1000, name, color=INK, fontsize=8, va="center")
        rows += [(name, float(k), int(v["size"]), v["median"]) for k, v in g.iterrows()]
    ax.set_xlabel("Стаж профессиональной работы, лет (середина интервала)")
    ax.set_ylabel("Медианный доход, тыс. USD")
    ax.set_xlim(0, 40)
    ax.legend(loc="upper left")
    save(fig, "05_experience_line")
    pd.DataFrame(rows, columns=["group", "exp_mid", "n", "median"]).to_csv(TABLES / "fig_experience.csv", index=False)


def correlation_heatmap(df, x):
    data = x.assign(ln_salary=df["ln_salary"])
    corr = data.corr()
    labels = [FEATURE_RU[c] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10, 8.4))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iat[i, j]
            ax.text(j, i, fmt_dec(v), ha="center", va="center",
                    fontsize=5, color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Коэффициент корреляции Пирсона")
    save(fig, "06_correlation")
    corr.to_csv(TABLES / "fig_correlation.csv")


def boxplots(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    specs = [("ed_level", "ed_rank", EDU_RU, "Уровень образования"),
             ("org_size", "org_employees", ORG_RU, "Численность сотрудников компании")]
    rows = []
    for axis, (col, key, names, title) in zip(ax, specs):
        part = df.dropna(subset=[key])
        order = part.groupby(col)[key].first().sort_values().index
        data = [part.loc[part[col] == c, "comp_usd"] / 1000 for c in order]
        axis.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                     boxprops={"facecolor": "#b7d3f6", "edgecolor": "#1c5cab"},
                     medianprops={"color": INK, "linewidth": 1.6},
                     whiskerprops={"color": "#1c5cab"}, capprops={"color": "#1c5cab"})
        axis.set_xticks(range(1, len(order) + 1),
                        [f"{names[c]}\n({fmt_int(len(d))})" for c, d in zip(order, data)],
                        rotation=35, ha="right", fontsize=7.5)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
        rows += [(col, names[c], len(d), d.median() * 1000) for c, d in zip(order, data)]
    ax[0].set_ylabel("Годовой доход, тыс. USD")
    fig.tight_layout()
    save(fig, "07_boxplots")
    pd.DataFrame(rows, columns=["factor", "level", "n", "median"]).to_csv(TABLES / "fig_boxplots.csv", index=False)


def language_premium(df, x):
    # поправка на страну: сравниваем отклонения ln(дохода) от средней по своей стране
    sizes = df["country"].map(df["country"].value_counts())
    group = df["country"].where(sizes >= MIN_GROUP, "Other")
    resid = df["ln_salary"] - df.groupby(group)["ln_salary"].transform("mean")
    rows = []
    for col in [c for c in x.columns if c.startswith("lang_")]:
        users = x[col] == 1
        known = x[col].notna()
        raw = df.loc[users, "ln_salary"].mean() - df.loc[known & ~users, "ln_salary"].mean()
        adj = resid[users].mean() - resid[known & ~users].mean()
        rows.append((FEATURE_RU[col], int(users.sum()), np.expm1(raw) * 100, np.expm1(adj) * 100))
    table = pd.DataFrame(rows, columns=["language", "users", "raw_pct", "adj_pct"]).sort_values("adj_pct")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = [SERIES[0] if v >= 0 else SERIES[7] for v in table["adj_pct"]]
    ax.barh(table["language"], table["adj_pct"], color=colors, height=0.68)
    for i, v in enumerate(table["adj_pct"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, ("+" if v >= 0 else "") + f"{fmt_dec(v, 1)} %", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Разница в доходе пользователей языка и остальных\nс поправкой на страну, %")
    ax.grid(axis="y", visible=False)
    lim = table["adj_pct"].abs().max() + 6
    ax.set_xlim(-lim, lim)
    save(fig, "08_language_premium")
    table.to_csv(TABLES / "fig_language_premium.csv", index=False)


def devtype_bar(df):
    stats = df.groupby("dev_type")["comp_usd"].agg(n="size", median="median")
    stats = stats[stats["n"] >= 100].sort_values("median")
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    labels = [f"{DEV_RU.get(d, d)} ({fmt_int(n)})" for d, n in stats["n"].items()]
    ax.barh(labels, stats["median"] / 1000, color=SERIES[0], height=0.7)
    for i, v in enumerate(stats["median"]):
        ax.text(v / 1000 + 1.5, i, usd_k(v), va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Медианный годовой доход, тыс. USD")
    ax.grid(axis="y", visible=False)
    save(fig, "09_devtype_median")
    stats.assign(dev_ru=[DEV_RU.get(d, d) for d in stats.index]).to_csv(TABLES / "fig_devtype.csv")


def experience_hexbin(df):
    part = df.dropna(subset=["work_exp"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    hb = ax.hexbin(part["work_exp"], part["ln_salary"], gridsize=(40, 25), cmap=SEQUENTIAL,
                   mincnt=1, bins="log", linewidths=0.2, edgecolors="white")
    med = part[part["work_exp"] <= 40].groupby("work_exp")["ln_salary"].median()
    ax.plot(med.index, med.values, color=SERIES[1], lw=2, label="медиана ln(дохода) по стажу")
    ax.set_xlabel("Стаж профессиональной работы, лет")
    ax.set_ylabel("ln(годовой доход, USD)")
    ax.set_xlim(0, 45)
    ax.grid(False)
    ax.legend(loc="lower right")
    fig.colorbar(hb, ax=ax, label="Число анкет в ячейке (лог. шкала)")
    save(fig, "10_experience_hexbin")


def main():
    setup()
    sample, df, x = prepare()
    missing_share()
    salary_distribution(sample, df)
    remote_pie(df)
    country_bar(df)
    experience_line(df)
    correlation_heatmap(df, x)
    boxplots(df)
    language_premium(df, x)
    devtype_bar(df)
    experience_hexbin(df)
    print("графики:", ", ".join(sorted(p.name for p in (ROOT / "reports" / "figures").glob("*.png"))))


if __name__ == "__main__":
    main()
