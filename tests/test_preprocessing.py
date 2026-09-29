import pandas as pd
import pytest

from churn_fastapi import dataset, preprocessing
from churn_fastapi.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def test_column_groups_cover_all_features():
    numeric, categorical = preprocessing.get_column_groups()
    assert numeric == NUMERIC_FEATURES
    assert categorical == CATEGORICAL_FEATURES
    assert set(numeric) | set(categorical) == set(ALL_FEATURES)
    assert not set(numeric) & set(categorical)


def test_split_xy_separates_target():
    df = dataset.load_dataframe()
    X, y = preprocessing.split_xy(df)
    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN
    assert set(X.columns) == set(ALL_FEATURES)
    assert len(X) == len(y) == 2000


def test_make_split_sizes():
    df = dataset.load_dataframe()
    split = preprocessing.make_split(df)
    assert split.train_size + split.test_size == 2000
    assert split.test_size == 400
    assert split.train_size == 1600
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)
    assert set(split.X_train.columns) == set(ALL_FEATURES)


def test_make_split_is_reproducible():
    df = dataset.load_dataframe()
    a = preprocessing.make_split(df, random_state=42)
    b = preprocessing.make_split(df, random_state=42)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_make_split_different_seed_gives_different_split():
    df = dataset.load_dataframe()
    a = preprocessing.make_split(df, random_state=42)
    b = preprocessing.make_split(df, random_state=7)
    assert a.train_size == b.train_size
    assert not a.X_train.index.equals(b.X_train.index)


def test_stratified_split_keeps_class_ratio():
    df = dataset.load_dataframe()
    split = preprocessing.make_split(df, stratify=True)
    full_ratio = df[TARGET_COLUMN].mean()
    assert abs(split.y_train.mean() - full_ratio) < 0.01
    assert abs(split.y_test.mean() - full_ratio) < 0.01


def test_unstratified_split_allowed():
    df = dataset.load_dataframe()
    split = preprocessing.make_split(df, stratify=False)
    assert split.train_size + split.test_size == 2000


def test_invalid_test_size_raises():
    df = dataset.load_dataframe()
    with pytest.raises(ValueError, match="test_size"):
        preprocessing.make_split(df, test_size=1.5)
    with pytest.raises(ValueError, match="test_size"):
        preprocessing.make_split(df, test_size=0.0)


def test_describe_missing_on_clean_dataset():
    df = dataset.load_dataframe()
    assert preprocessing.describe_missing(df) == {}
    assert sum(preprocessing.describe_missing(df).values()) == 0


def test_preprocessor_handles_missing_values():
    import numpy as np

    df = dataset.load_dataframe().head(200).copy()
    df.loc[df.index[:10], "monthly_fee"] = np.nan
    df.loc[df.index[10:20], "region"] = None

    split = preprocessing.make_split(df)
    pipeline = preprocessing.build_pipeline(_DummyClassifier())
    pipeline.fit(split.X_train, split.y_train)

    transformed = pipeline.named_steps["preprocessor"].transform(split.X_test)
    assert not pd.isna(transformed).any()


class _DummyClassifier:
    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        import numpy as np

        return np.tile([0.8, 0.2], (len(X), 1))


class TestSplitInfoEndpoint:
    def test_defaults(self, client):
        resp = client.get("/dataset/split-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["test_size"] == 0.2
        assert data["random_state"] == 42
        assert data["stratify"] is True
        assert data["total_rows"] == 2000
        assert data["train_size"] == 1600
        assert data["test_rows"] == 400
        assert data["numeric_features"] == NUMERIC_FEATURES
        assert data["categorical_features"] == CATEGORICAL_FEATURES
        assert data["features"] == ALL_FEATURES
        assert data["target_column"] == TARGET_COLUMN
        assert data["missing_values_total"] == 0
        assert data["stratified_consistent"] is True

    def test_churn_distribution_for_all_three_sets(self, client):
        resp = client.get("/dataset/split-info")
        dist = resp.json()["churn_distribution"]
        assert set(dist) == {"full", "train", "test"}
        assert dist["full"] == {"0": 1597, "1": 403}
        assert sum(dist["train"].values()) == 1600
        assert sum(dist["test"].values()) == 400
        assert set(dist["train"]) == {"0", "1"}
        assert set(dist["test"]) == {"0", "1"}

    def test_churn_ratios_are_close(self, client):
        data = client.get("/dataset/split-info").json()
        ratios = data["churn_ratio"]
        assert abs(ratios["full"] - ratios["train"]) < 0.01
        assert abs(ratios["full"] - ratios["test"]) < 0.01

    def test_custom_test_size(self, client):
        resp = client.get("/dataset/split-info", params={"test_size": 0.3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["test_rows"] == 600
        assert data["train_size"] == 1400
        assert data["train_size"] + data["test_rows"] == data["total_rows"]

    def test_custom_random_state(self, client):
        data = client.get("/dataset/split-info", params={"random_state": 7}).json()
        assert data["random_state"] == 7
        assert data["train_size"] == 1600

    def test_without_stratify(self, client):
        data = client.get("/dataset/split-info", params={"stratify": False}).json()
        assert data["stratify"] is False
        assert data["train_size"] + data["test_rows"] == 2000

    def test_invalid_test_size_returns_422(self, client):
        assert client.get("/dataset/split-info", params={"test_size": 0}).status_code == 422
        assert client.get("/dataset/split-info", params={"test_size": 1}).status_code == 422
        assert client.get("/dataset/split-info", params={"test_size": 2}).status_code == 422

    def test_returns_404_when_dataset_missing(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(dataset, "DATASET_PATH", tmp_path / "missing.csv")
        dataset.clear_cache()
        try:
            assert client.get("/dataset/split-info").status_code == 404
        finally:
            dataset.clear_cache()
