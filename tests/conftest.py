"""Test configuration and fixtures for Yohou-MLflow."""

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import pytest
import yaml
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase
from sklearn.linear_model import Ridge
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import DecompositionPipeline
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.stationarity import PolynomialTrendForecaster

# Hypothesis remembers failing examples so a rerun replays them first. That example
# database defaults to `.hypothesis/` at the repo root; this puts it under
# `.artifacts/` with every other piece of throwaway output. It has no config-file
# key, so registering and loading a profile is the only way to set it -- which is why
# this lives here rather than in pyproject.toml.
#
# This moves the example database ONLY. Hypothesis also writes a `.hypothesis/`
# storage directory for its own constants and unicode caches, which no setting
# relocates. Newer versions drop a self-ignoring `.gitignore` inside it and older
# ones do not, so `.gitignore` lists it explicitly rather than depending on which
# version resolved.
settings.register_profile("default", database=DirectoryBasedExampleDatabase(".artifacts/hypothesis"))
settings.load_profile("default")

N_ROWS = 220
FIT_END = 180  # rows [0, FIT_END) are fitted
OBSERVED_END = 190  # rows [FIT_END, OBSERVED_END) are observed after fitting
NEXT_END = 195  # rows [OBSERVED_END, NEXT_END) are "new" rows a test may observe
HORIZON = 5

FAMILIES = ("point", "interval", "class_proba", "panel", "composite")


def _time(n: int, time_zone: str | None = None) -> pl.Series:
    start = dt.datetime(2020, 1, 1)
    return pl.datetime_range(start, start + dt.timedelta(days=n - 1), "1d", eager=True, time_zone=time_zone)


def make_series(n: int = N_ROWS, time_zone: str | None = None) -> pl.DataFrame:
    """Return a univariate daily series with a weekly pattern and noise."""
    rng = np.random.default_rng(0)
    steps = np.arange(n)
    return pl.DataFrame({"time": _time(n, time_zone), "v": np.sin(steps / 7) + rng.normal(0, 0.1, n)})


def make_panel(n: int = N_ROWS) -> pl.DataFrame:
    """Return a three-group panel (``a``, ``b``, ``c``) in yohou's ``group__column`` layout."""
    steps = np.arange(n)
    return pl.DataFrame({
        "time": _time(n),
        "a__v": np.sin(steps / 7),
        "b__v": np.cos(steps / 7),
        "c__v": steps / n,
    })


def make_classes(n: int = N_ROWS) -> pl.DataFrame:
    """Return a daily categorical series with three classes."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({"time": _time(n), "weather": rng.choice(["sun", "rain", "cloud"], n).tolist()})


def _build(family: str) -> tuple[Any, pl.DataFrame, int]:
    if family == "point":
        return PointReductionForecaster(estimator=Ridge()), make_series(), HORIZON
    if family == "interval":
        forecaster = SplitConformalForecaster(point_forecaster=PointReductionForecaster(estimator=Ridge()))
        return forecaster, make_series(), HORIZON
    if family == "class_proba":
        # The default classifier predicts one step ahead only.
        return ClassProbaReductionForecaster(), make_classes(), 1
    if family == "panel":
        return PointReductionForecaster(estimator=Ridge()), make_panel(), HORIZON
    if family == "composite":
        forecaster = DecompositionPipeline([
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", SeasonalNaive(seasonality=7)),
        ])
        return forecaster, make_series(), HORIZON
    raise ValueError(family)


@dataclass
class Case:
    """A fitted forecaster that has observed rows after fitting, plus data around it."""

    family: str
    forecaster: Any
    data: pl.DataFrame
    horizon: int

    @property
    def predict_method(self) -> str:
        """Name of the method returning this family's main prediction."""
        return "predict_class_proba" if self.family == "class_proba" else "predict"

    @property
    def next_rows(self) -> pl.DataFrame:
        """Rows right after the last observed one."""
        return self.data[OBSERVED_END:NEXT_END]

    @property
    def rewind_window(self) -> pl.DataFrame:
        """A window ending before the last observed row, long enough to rewind to."""
        return self.data[FIT_END - 100 : FIT_END + 5]


def _fit_and_observe(family: str, forecaster: Any, data: pl.DataFrame, horizon: int) -> Case:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(data[:FIT_END], forecasting_horizon=horizon)
        forecaster.observe(data[FIT_END:OBSERVED_END])
    return Case(family, forecaster, data, horizon)


def make_case(family: str) -> Case:
    """Fit a fresh forecaster of ``family`` and observe rows after fitting."""
    return _fit_and_observe(family, *_build(family))


@pytest.fixture(params=FAMILIES)
def case(request: pytest.FixtureRequest) -> Case:
    """One fitted, observed forecaster per family."""
    return make_case(request.param)


@pytest.fixture
def point_case() -> Case:
    """A fitted, observed point forecaster."""
    return make_case("point")


@pytest.fixture
def interval_case() -> Case:
    """A fitted, observed conformal interval forecaster."""
    return make_case("interval")


@pytest.fixture
def class_case() -> Case:
    """A fitted, observed class-probability forecaster."""
    return make_case("class_proba")


@pytest.fixture
def panel_case() -> Case:
    """A fitted, observed forecaster on a three-group panel."""
    return make_case("panel")


@pytest.fixture
def local_ridge_case() -> Case:
    """A point forecaster whose regressor is a class outside the trust policy."""
    from local_estimators import LocalRidge

    return _fit_and_observe("point", PointReductionForecaster(estimator=LocalRidge()), make_series(), HORIZON)


@pytest.fixture
def local_ridge_type() -> str:
    """Fully qualified name skops reports for ``LocalRidge``."""
    from local_estimators import LocalRidge

    return f"{LocalRidge.__module__}.{LocalRidge.__qualname__}"


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from its own temporary directory.

    MLflow's defaults are relative to the working directory (a ``mlflow.db`` tracking
    store, ``./mlruns`` artifacts), so any code path that reaches them from the repo
    root would write there.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def tracking(tmp_path: Path):
    """Point MLflow at a SQLite tracking store and registry under ``tmp_path``.

    Artifacts go under ``tmp_path`` too: with a database backend MLflow otherwise
    writes them to ``./mlruns`` in the working directory.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment_id = mlflow.create_experiment("tests", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(previous_tracking)
    mlflow.set_registry_uri(previous_registry)


def read_flavor(path: Path) -> dict[str, Any]:
    """Return the ``yohou`` section of a saved model's ``MLmodel`` file."""
    return yaml.safe_load((path / "MLmodel").read_text())["flavors"]["yohou"]


def edit_flavor(path: Path, **changes: Any) -> None:
    """Overwrite keys of the ``yohou`` section of a saved model's ``MLmodel`` file."""
    mlmodel = yaml.safe_load((path / "MLmodel").read_text())
    mlmodel["flavors"]["yohou"].update(changes)
    (path / "MLmodel").write_text(yaml.safe_dump(mlmodel))
