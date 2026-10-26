import numpy as np
import pandas as pd
import pytest

from models import CountryModel, Metrics, PCAAnalyzer, PCRModel, ProfileOLS


class StubBuilder:
    """Минимальная замена FeatureBuilder: признаки берутся из столбцов x*."""

    def profile(self, df):
        return df[[c for c in df.columns if c.startswith("x")]]

    def countries(self, df):
        return pd.get_dummies(df["country"], prefix="country", dtype=int).drop(columns="country_US")

    def target(self, df):
        return df["y"]


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=(2, n))
    noise = rng.normal(scale=0.3, size=(6, n))
    x = {"x1": f1 + noise[0], "x2": f1 + noise[1], "x3": f1 + noise[2],
         "x4": f2 + noise[3], "x5": f2 + noise[4], "x6": noise[5]}
    df = pd.DataFrame(x)
    df["country"] = rng.choice(["US", "DE"], size=n)
    df["y"] = 11 + 0.3 * df["x1"] - 0.2 * df["x4"] - 0.5 * (df["country"] == "DE") \
        + rng.normal(scale=0.1, size=n)
    return df


def test_metrics_formulas():
    y = pd.Series([10.0, 11.0, 12.0, 13.0])
    m = Metrics.compute(y, y + 0.5, n_params=1)
    assert m.mae == pytest.approx(0.5) and m.rmse == pytest.approx(0.5)
    assert m.r2 == pytest.approx(1 - 4 * 0.25 / 5)
    assert m.adj_r2 == pytest.approx(1 - (1 - m.r2) * 3 / 2)
    assert m.mae_usd == pytest.approx(np.mean(np.exp(y + 0.5) - np.exp(y)))


def test_pca_finds_two_factors_and_aligns_signs():
    pca = PCAAnalyzer().fit(synthetic().filter(like="x"))
    assert pca.n_kaiser() == 2
    assert pca.n_for_share(0.8) <= 4
    loadings = pca.loadings(2)
    assert list(loadings.columns) == ["PC1", "PC2"]
    for pc in loadings:
        assert loadings[pc].loc[loadings[pc].abs().idxmax()] > 0
    # первая компонента собирает x1–x3
    assert set(loadings["PC1"].abs().nlargest(3).index) == {"x1", "x2", "x3"}


def test_ols_recovers_coefficients():
    df = synthetic()
    model = ProfileOLS(StubBuilder()).fit(df)
    coef = model.coefficients()["coef"]
    assert coef["country_DE"] == pytest.approx(-0.5, abs=0.05)
    assert model.evaluate(df).r2 > 0.9


def test_pcr_with_all_components_matches_ols():
    df = synthetic()
    ols = ProfileOLS(StubBuilder()).fit(df)
    pcr = PCRModel(StubBuilder(), k=6).fit(df)
    assert np.allclose(ols.predict(df), pcr.predict(df))


def test_pcr_feature_effects_shape_and_test_data_unchanged():
    train, test = synthetic(seed=1), synthetic(seed=2)
    pcr = PCRModel(StubBuilder(), k=2).fit(train)
    before = pcr.pca.loadings(2).copy()
    pcr.predict(test)
    assert pcr.pca.loadings(2).equals(before)  # тестовая выборка только преобразуется
    effects = pcr.feature_effects()
    assert list(effects.index) == ["x1", "x2", "x3", "x4", "x5", "x6"]
    assert effects["x1"] > 0 > effects["x4"]


def test_country_model_uses_only_dummies():
    df = synthetic()
    model = CountryModel(StubBuilder()).fit(df)
    assert list(model.coefficients().index) == ["const", "country_DE"]
