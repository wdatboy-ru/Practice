"""Раздел 3: метод главных компонент и регрессионные модели дохода.

Все модели наследуются от SalaryModel: общий код обучения (МНК через
statsmodels), прогноза и оценки качества написан один раз, а подклассы
определяют только матрицу признаков (метод design).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Metrics:
    r2: float
    adj_r2: float
    mae: float      # в единицах ln(дохода)
    rmse: float
    mae_usd: float  # после обратного перехода к долларам
    n: int
    p: int

    @classmethod
    def compute(cls, y_true, y_pred, n_params):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
        return cls(r2=r2, adj_r2=adj_r2,
                   mae=mean_absolute_error(y_true, y_pred),
                   rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                   mae_usd=mean_absolute_error(np.exp(y_true), np.exp(y_pred)),
                   n=n, p=n_params)


class PCAAnalyzer:
    """Стандартизация признаков и PCA с доступом к собственным числам и нагрузкам."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.features_ = None

    def fit(self, X):
        self.features_ = list(X.columns)
        self.pca.fit(self.scaler.fit_transform(X))
        # знак компоненты произволен; выбираем его так, чтобы наибольшая
        # по модулю нагрузка была положительной — так легче читать компоненты
        comp = self.pca.components_
        signs = np.sign(comp[np.arange(len(comp)), np.abs(comp).argmax(axis=1)])
        self.pca.components_ = comp * signs[:, None]
        return self

    @property
    def eigenvalues_(self):
        return self.pca.explained_variance_

    @property
    def explained_ratio_(self):
        return self.pca.explained_variance_ratio_

    def n_kaiser(self):
        """Число компонент с собственным числом больше 1 (критерий Кайзера)."""
        return int((self.eigenvalues_ > 1).sum())

    def n_for_share(self, share):
        """Минимальное число компонент, объясняющих долю дисперсии share."""
        return int(np.searchsorted(np.cumsum(self.explained_ratio_), share) + 1)

    def loadings(self, k):
        """Нагрузки = корреляции признаков с компонентами: v_j * sqrt(lambda_j)."""
        values = self.pca.components_[:k].T * np.sqrt(self.eigenvalues_[:k])
        return pd.DataFrame(values, index=self.features_, columns=self._names(k))

    def transform(self, X, k):
        scores = self.pca.transform(self.scaler.transform(X[self.features_]))[:, :k]
        return pd.DataFrame(scores, index=X.index, columns=self._names(k))

    @staticmethod
    def _names(k):
        return [f"PC{i + 1}" for i in range(k)]


class SalaryModel(ABC):
    """Линейная регрессия ln(дохода) по МНК. Подкласс задаёт только design()."""

    name = "Модель"

    def __init__(self, builder):
        self.builder = builder
        self.columns_ = None
        self.result_ = None

    @abstractmethod
    def design(self, df, fit=False):
        """Матрица признаков X без свободного члена."""

    def fit(self, df):
        X = self.design(df, fit=True)
        self.columns_ = list(X.columns)
        self.result_ = sm.OLS(self.builder.target(df), sm.add_constant(X, has_constant="add")).fit()
        return self

    def predict(self, df):
        X = sm.add_constant(self.design(df), has_constant="add")
        return self.result_.predict(X[["const"] + self.columns_])

    def evaluate(self, df):
        return Metrics.compute(self.builder.target(df), self.predict(df), len(self.columns_))

    def coefficients(self):
        r = self.result_
        ci = r.conf_int()
        table = pd.DataFrame({"coef": r.params, "std_err": r.bse, "t": r.tvalues,
                              "p_value": r.pvalues, "ci_low": ci[0], "ci_high": ci[1],
                              # при цели ln(y) рост фактора на 1 меняет доход на (e^b - 1) * 100 %
                              "effect_pct": np.expm1(r.params) * 100})
        table.loc["const", "effect_pct"] = np.nan
        return table


class CountryModel(SalaryModel):
    """Базовая модель: доход объясняется только страной."""

    name = "Только страна"

    def design(self, df, fit=False):
        return self.builder.countries(df)


class ProfileOLS(SalaryModel):
    """Множественная регрессия на исходные признаки профиля (и страну)."""

    def __init__(self, builder, with_country=True):
        super().__init__(builder)
        self.with_country = with_country
        self.name = "МНК: признаки + страна" if with_country else "МНК: только признаки"

    def design(self, df, fit=False):
        X = self.builder.profile(df)
        return X.join(self.builder.countries(df)) if self.with_country else X


class PCRModel(SalaryModel):
    """Регрессия на главные компоненты (Principal Component Regression).

    PCA обучается только в fit() на обучающей выборке; при прогнозе
    тестовые данные лишь проецируются на найденные компоненты.
    """

    def __init__(self, builder, k, with_country=True):
        super().__init__(builder)
        self.k = k
        self.with_country = with_country
        self.pca = PCAAnalyzer()
        self.name = f"PCR, k = {k}" + (" + страна" if with_country else "")

    def design(self, df, fit=False):
        X = self.builder.profile(df)
        if fit:
            self.pca.fit(X)
        Z = self.pca.transform(X, self.k)
        return Z.join(self.builder.countries(df)) if self.with_country else Z

    def feature_effects(self):
        """Пересчёт коэффициентов при компонентах в эффекты исходных признаков:
        изменение ln(дохода) при росте признака на одно стандартное отклонение."""
        gamma = self.result_.params[self.pca._names(self.k)].to_numpy()
        effects = self.pca.pca.components_[:self.k].T @ gamma
        return pd.Series(effects, index=self.pca.features_)
