import numpy as np
import pandas as pd
import pytest

from features import FeatureBuilder, clean_experience, modified_z, remove_outliers

US = "United States of America"


def respondent(country, salary, **extra):
    row = {"country": country, "comp_usd": salary, "work_exp": 5.0, "years_code": 8.0,
           "age_mid": 29.5, "ed_rank": 3, "org_employees": 300, "remote_rank": 2,
           "is_manager": 0, "languages": "Python;SQL"}
    row.update(extra)
    return row


def test_modified_z_marks_far_value():
    z = modified_z(pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 60.0]))
    assert abs(z.iloc[2]) < 0.5
    assert z.iloc[-1] > 3.5


def test_modified_z_constant_group_gives_nan():
    assert modified_z(pd.Series([5.0, 5.0, 5.0])).isna().all()


def test_remove_outliers_applies_absolute_bounds():
    df = pd.DataFrame([respondent(US, s) for s in [500, 60_000, 70_000, 80_000, 2_000_000]])
    kept, report = remove_outliers(df, min_group=1)
    assert kept["comp_usd"].tolist() == [60_000, 70_000, 80_000]
    assert report["below_bound"] == 1 and report["above_bound"] == 1


def test_remove_outliers_is_relative_to_country():
    rich = [respondent("A", s) for s in [50_000, 52_000, 55_000, 58_000, 60_000, 5_000]]
    poor = [respondent("B", s) for s in [4_000, 4_500, 5_000, 5_500, 6_000]]
    kept, report = remove_outliers(pd.DataFrame(rich + poor), min_group=5)
    # 5 000 USD — выброс для страны A, но обычный доход для страны B
    assert len(kept[(kept.country == "A") & (kept.comp_usd == 5_000)]) == 0
    assert len(kept[(kept.country == "B") & (kept.comp_usd == 5_000)]) == 1
    assert report["country_outliers"] == 1


def test_clean_experience_blanks_values_impossible_for_age():
    df = pd.DataFrame([
        respondent(US, 1e5, age_group="18-24 years old", work_exp=50.0, years_code=30.0),
        respondent(US, 1e5, age_group="35-44 years old", work_exp=20.0, years_code=25.0),
        respondent(US, 1e5, age_group="65 years or older", work_exp=50.0, years_code=60.0),
    ])
    cleaned, n = clean_experience(df)
    assert np.isnan(cleaned.loc[0, "work_exp"]) and np.isnan(cleaned.loc[0, "years_code"])
    assert cleaned.loc[1, "work_exp"] == 20.0 and cleaned.loc[2, "years_code"] == 60.0
    assert n == 2


def test_profile_encodes_languages_and_scales():
    df = pd.DataFrame([respondent(US, 1e5), respondent(US, 1e5, languages=None)])
    x = FeatureBuilder().profile(df)
    assert x.loc[0, "lang_python"] == 1 and x.loc[0, "lang_sql"] == 1 and x.loc[0, "lang_go"] == 0
    assert x.loc[0, "n_languages"] == 2
    assert np.isnan(x.loc[1, "n_languages"]) and np.isnan(x.loc[1, "lang_python"])
    assert x.loc[0, "org_size_log"] == pytest.approx(np.log10(300))
    assert len(x.columns) == 20


def test_country_dummies_group_rare_and_unseen():
    train = pd.DataFrame([respondent(c, 1e5) for c in [US, US, "Germany", "Germany", "Chad"]])
    fb = FeatureBuilder(min_country=2).fit(train)
    assert fb.countries_ == ["Germany"]
    d = fb.countries(pd.DataFrame([respondent(c, 1e5) for c in [US, "Germany", "Chad", "Peru"]]))
    assert list(d.columns) == ["country_Germany", "country_Other"]
    assert d.to_numpy().tolist() == [[0, 0], [1, 0], [0, 1], [0, 1]]


def test_target_is_natural_log():
    df = pd.DataFrame([respondent(US, np.e ** 11)])
    assert FeatureBuilder().target(df).iloc[0] == pytest.approx(11)
