"""Раздел 3: обучение моделей, выбор числа компонент, метрики и графики."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from features import FeatureBuilder, clean_experience, load_sample, remove_outliers
from models import CountryModel, PCAAnalyzer, PCRModel, ProfileOLS
from plot_style import (COUNTRY_RU, DIVERGING, FEATURE_RU, INK, INK_2, SEQUENTIAL, SERIES,
                        fmt_dec, save, setup)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
SEED = 42
TEST_SIZE = 0.2


class Experiment:
    """Готовит выборку, делит её 80/20 и обучает набор моделей на одних и тех же данных."""

    def __init__(self, test_size=TEST_SIZE, seed=SEED):
        sample = load_sample()
        kept, self.outliers = remove_outliers(sample)
        kept, _ = clean_experience(kept)
        complete = FeatureBuilder().profile(kept).notna().all(axis=1)
        self.data = kept[complete]
        self.train, self.test = train_test_split(self.data, test_size=test_size, random_state=seed)
        self.builder = FeatureBuilder().fit(self.train)
        self.models = []

    def add(self, model):
        self.models.append(model.fit(self.train))
        return model

    def metrics_table(self):
        rows = []
        for m in self.models:
            for part, df in [("train", self.train), ("test", self.test)]:
                r = m.evaluate(df)
                rows.append({"model": m.name, "sample": part, "p": r.p, "r2": r.r2,
                             "adj_r2": r.adj_r2, "mae": r.mae, "rmse": r.rmse, "mae_usd": r.mae_usd})
        return pd.DataFrame(rows)

    def cv_rmse(self, max_k, folds=5):
        """Средняя RMSE регрессии на k компонент по K-блочной кросс-валидации на train."""
        kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        rows = []
        for k in range(1, max_k + 1):
            scores = [PCRModel(self.builder, k).fit(self.train.iloc[a]).evaluate(self.train.iloc[b]).rmse
                      for a, b in kf.split(self.train)]
            rows.append({"k": k, "rmse_mean": np.mean(scores), "rmse_std": np.std(scores)})
        return pd.DataFrame(rows)


def vif_table(X):
    values = X.assign(const=1.0).to_numpy()
    return pd.Series([variance_inflation_factor(values, i) for i in range(X.shape[1])], index=X.columns)


def plot_scree(pca, k_kaiser, k80):
    idx = np.arange(1, len(pca.eigenvalues_) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(idx, pca.eigenvalues_, color=[SERIES[0] if i <= k_kaiser else "#b7d3f6" for i in idx])
    ax[0].axhline(1, color=INK_2, ls="--", lw=1)
    ax[0].text(len(idx), 1.05, "λ = 1", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax[0].set_xlabel("Номер главной компоненты")
    ax[0].set_ylabel("Собственное число λ")
    ax[0].set_xticks(idx)
    cum = np.cumsum(pca.explained_ratio_) * 100
    ax[1].plot(idx, cum, marker="o", ms=4, color=SERIES[1])
    ax[1].axhline(80, color=INK_2, ls="--", lw=1)
    ax[1].annotate(f"k = {k80}: {fmt_dec(cum[k80 - 1], 1)} %", (k80, cum[k80 - 1]),
                   xytext=(k80 + 1.5, cum[k80 - 1] - 14), fontsize=8,
                   arrowprops={"arrowstyle": "-", "color": INK_2})
    ax[1].set_xlabel("Число компонент k")
    ax[1].set_ylabel("Накопленная доля дисперсии, %")
    ax[1].set_xticks(idx)
    ax[1].set_ylim(0, 102)
    fig.tight_layout()
    save(fig, "11_pca_scree")


def plot_loadings(loadings):
    fig, ax = plt.subplots(figsize=(7.5, 7.6))
    im = ax.imshow(loadings, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(loadings.shape[1]), loadings.columns)
    ax.set_yticks(range(loadings.shape[0]), [FEATURE_RU[f] for f in loadings.index])
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            v = loadings.iat[i, j]
            ax.text(j, i, fmt_dec(v), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK, fontweight="bold" if abs(v) >= 0.4 else None)
    ax.grid(False)
    ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, shrink=0.6, label="Нагрузка (корреляция признака с компонентой)")
    save(fig, "12_pca_loadings")


def plot_pc_plane(scores, ln_salary):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    hb = ax.hexbin(scores["PC1"], scores["PC2"], C=np.exp(ln_salary) / 1000, reduce_C_function=np.median,
                   gridsize=35, cmap=SEQUENTIAL, mincnt=5, linewidths=0.2, edgecolors="white")
    ax.set_xlim(np.percentile(scores["PC1"], [0.5, 99.5]))
    ax.set_ylim(np.percentile(scores["PC2"], [0.5, 99.5]))
    ax.set_xlabel("PC1 («профессиональная зрелость»)")
    ax.set_ylabel("PC2 («широта стека»)")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, label="Медианный доход в ячейке, тыс. USD")
    save(fig, "13_pc_plane")


def plot_cv(cv, k_best):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar(cv["k"], cv["rmse_mean"], yerr=cv["rmse_std"], marker="o", ms=4, color=SERIES[0],
                ecolor="#86b6ef", capsize=2)
    ax.axvline(k_best, color=INK_2, ls="--", lw=1)
    ax.text(k_best + 0.3, cv["rmse_mean"].max(), f"выбрано k = {k_best}", fontsize=8, color=INK_2, va="top")
    ax.set_xlabel("Число главных компонент k (плюс фиктивные переменные страны)")
    ax.set_ylabel("RMSE на проверочных блоках, ln(USD)")
    ax.set_xticks(cv["k"])
    save(fig, "14_cv_k")


def plot_actual_vs_predicted(model, test):
    y = model.builder.target(test)
    pred = model.predict(test)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].scatter(y, pred, s=5, alpha=0.25, color=SERIES[0], edgecolors="none")
    lims = [min(y.min(), pred.min()) - 0.2, max(y.max(), pred.max()) + 0.2]
    ax[0].plot(lims, lims, color=INK, lw=1, ls="--", label="идеальный прогноз")
    ax[0].set_xlim(lims)
    ax[0].set_ylim(lims)
    ax[0].set_xlabel("Фактический ln(дохода)")
    ax[0].set_ylabel("Прогноз ln(дохода)")
    ax[0].set_title(model.name)
    ax[0].legend(loc="upper left")
    resid = y - pred
    ax[1].hist(resid, bins=60, color=SERIES[2], edgecolor="white", linewidth=0.4)
    ax[1].axvline(0, color=INK, lw=1)
    ax[1].set_xlabel("Остаток: факт − прогноз, ln(USD)")
    ax[1].set_ylabel("Число анкет")
    ax[1].set_title(f"Остатки на тестовой выборке, медиана {fmt_dec(resid.median())}")
    fig.tight_layout()
    save(fig, "15_actual_vs_predicted")


def plot_effects(series, ci, name, xlabel):
    order = series.sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.2))
    colors = [SERIES[0] if v >= 0 else SERIES[7] for v in series[order]]
    err = None if ci is None else np.abs(ci.loc[order].to_numpy().T - series[order].to_numpy())
    ax.barh(range(len(order)), series[order], color=colors, height=0.65,
            xerr=err, error_kw={"ecolor": INK_2, "lw": 0.8, "capsize": 2})
    ax.set_yticks(range(len(order)), [FEATURE_RU.get(i, i) for i in order])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    save(fig, name)


def main():
    setup()
    exp = Experiment()
    X_train = exp.builder.profile(exp.train)

    # --- анализ главных компонент на обучающей выборке
    pca = PCAAnalyzer().fit(X_train)
    k_kaiser, k80 = pca.n_kaiser(), pca.n_for_share(0.8)
    plot_scree(pca, k_kaiser, k80)
    plot_pc_plane(pca.transform(X_train, 2), exp.builder.target(exp.train))
    pd.DataFrame({"eigenvalue": pca.eigenvalues_, "ratio": pca.explained_ratio_,
                  "cum_ratio": np.cumsum(pca.explained_ratio_)},
                 index=pd.RangeIndex(1, len(pca.eigenvalues_) + 1, name="k")).to_csv(TABLES / "pca_eigen.csv")

    # --- выбор k кросс-валидацией: наименьшее k, чья RMSE не хуже минимума + 1 стандартная ошибка
    cv = exp.cv_rmse(max_k=X_train.shape[1])
    best = cv.loc[cv["rmse_mean"].idxmin()]
    k_cv = int(cv.loc[cv["rmse_mean"] <= best["rmse_mean"] + best["rmse_std"] / np.sqrt(5), "k"].min())
    plot_cv(cv, k_cv)
    cv.to_csv(TABLES / "cv_k.csv", index=False)

    loadings = pca.loadings(max(k_kaiser, k_cv))
    plot_loadings(loadings)
    loadings.to_csv(TABLES / "pca_loadings.csv")

    # --- модели
    exp.add(CountryModel(exp.builder))
    exp.add(ProfileOLS(exp.builder, with_country=False))
    ols = exp.add(ProfileOLS(exp.builder))
    exp.add(PCRModel(exp.builder, k_kaiser, with_country=False))
    pcr_kaiser = exp.add(PCRModel(exp.builder, k_kaiser))
    pcr = exp.add(PCRModel(exp.builder, k_cv)) if k_cv != k_kaiser else pcr_kaiser
    metrics = exp.metrics_table()
    metrics.to_csv(TABLES / "metrics.csv", index=False)

    vif = vif_table(X_train)
    vif.rename("vif").to_csv(TABLES / "vif.csv")
    ols.coefficients().to_csv(TABLES / "coef_ols.csv")
    pcr.coefficients().to_csv(TABLES / "coef_pcr.csv")
    effects = pcr.feature_effects()
    effects_pct = (np.expm1(effects) * 100).rename("effect_pct_per_sd")
    effects_pct.to_csv(TABLES / "pcr_feature_effects.csv")

    pc_names = [f"PC{i + 1}" for i in range(pcr.k)]
    coef = pcr.coefficients()
    plot_effects(coef.loc[pc_names, "effect_pct"], np.expm1(coef.loc[pc_names, ["ci_low", "ci_high"]]) * 100,
                 "16_pcr_components", "Изменение дохода при росте компоненты на 1, % (95 % ДИ)")
    plot_effects(effects_pct, None, "17_feature_effects",
                 "Изменение дохода при росте признака на 1 стандартное отклонение, %")
    countries = coef.loc[[c for c in coef.index if c.startswith("country_")], "effect_pct"]
    countries.index = [COUNTRY_RU.get(c.removeprefix("country_"), c.removeprefix("country_")) for c in countries.index]
    plot_effects(countries, None, "18_country_effects", "Отличие дохода от США при равном профиле, %")
    plot_actual_vs_predicted(pcr, exp.test)

    summary = {"n_complete": len(exp.data), "n_train": len(exp.train), "n_test": len(exp.test),
               "n_countries_dummies": len(exp.builder.countries_), "k_kaiser": k_kaiser, "k80": k80,
               "k_cv": k_cv, "cum_ratio_kaiser": float(np.cumsum(pca.explained_ratio_)[k_kaiser - 1]),
               "ols_r2_train": float(ols.result_.rsquared), "ols_f_pvalue": float(ols.result_.f_pvalue),
               "pcr_name": pcr.name}
    (TABLES / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 160)
    print(json.dumps(summary, ensure_ascii=False))
    print(metrics.round(4).to_string(index=False))
    print(vif.round(2).sort_values(ascending=False).head(6).to_string())
    print(loadings.round(2).to_string())


if __name__ == "__main__":
    main()
