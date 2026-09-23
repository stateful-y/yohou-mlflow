"""Tests for the ``python_function`` flavour, always loaded through ``mlflow.pyfunc``."""

import copy
import datetime as dt
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import polars as pl
import pytest
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema
from sklearn.base import BaseEstimator
from yohou.point import PointReductionForecaster

import yohou_mlflow
from conftest import HORIZON, Case
from yohou_mlflow import UntrustedTypesError, VersionMismatchError, VersionMismatchWarning, YohouMlflowError
from yohou_mlflow._pyfunc import default_prediction_type, supported_prediction_types


def _pyfunc(case: Case, tmp_path: Path, **kwargs):
    path = tmp_path / "model"
    yohou_mlflow.save_model(case.forecaster, path, **kwargs)
    return mlflow.pyfunc.load_model(str(path))


def _observed_copy(case: Case):
    forecaster = copy.deepcopy(case.forecaster)
    forecaster.observe(case.next_rows)
    return forecaster


# -- Observe, then predict, on a copy -----------------------------------------------


def test_observe_then_predict(point_case: Case, tmp_path: Path) -> None:
    """With ``y``, the result is one forecast from the updated state."""
    model = _pyfunc(point_case, tmp_path)
    result = model.predict({"y": point_case.next_rows})
    assert result.equals(_observed_copy(point_case).predict())
    assert result["vintage_time"].unique().to_list() == [point_case.next_rows["time"].max()]


def test_calls_do_not_share_state(point_case: Case, tmp_path: Path) -> None:
    """Two identical calls return equal results; the loaded forecaster is unchanged."""
    model = _pyfunc(point_case, tmp_path)
    first = model.predict({"y": point_case.next_rows})
    second = model.predict({"y": point_case.next_rows})
    assert first.equals(second)
    assert model.get_raw_model().predict().equals(point_case.forecaster.predict())


@pytest.mark.parametrize(
    ("fixture", "prediction_type", "method"),
    [
        ("point_case", "point", "predict"),
        ("interval_case", "interval", "predict_interval"),
        ("class_case", "class_proba", "predict_class_proba"),
    ],
)
def test_predict_from_saved_state(
    fixture: str, prediction_type: str, method: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Without ``y``, each prediction type calls its predict method on the saved state."""
    case: Case = request.getfixturevalue(fixture)
    model = _pyfunc(case, tmp_path)
    result = model.predict({}, params={"prediction_type": prediction_type})
    assert result.equals(getattr(case.forecaster, method)())


# -- prediction_type and params -----------------------------------------------------


def test_interval_with_coverage_rates(interval_case: Case, tmp_path: Path) -> None:
    """``coverage_rates`` reaches ``predict_interval`` after observing."""
    model = _pyfunc(interval_case, tmp_path)
    params = {"prediction_type": "interval", "coverage_rates": [0.8, 0.95]}
    result = model.predict({"y": interval_case.next_rows}, params=params)
    assert result.equals(_observed_copy(interval_case).predict_interval(coverage_rates=[0.8, 0.95]))
    assert {"v_lower_0.8", "v_upper_0.95"} <= set(result.columns)


def test_class_probabilities_not_labels(class_case: Case, tmp_path: Path) -> None:
    """``class_proba`` returns one probability column per class, not the argmax label."""
    model = _pyfunc(class_case, tmp_path)
    result = model.predict({"y": class_case.next_rows}, params={"prediction_type": "class_proba"})
    observed = _observed_copy(class_case)
    assert result.equals(observed.predict_class_proba())
    assert {"weather_proba_cloud", "weather_proba_rain", "weather_proba_sun"} <= set(result.columns)
    assert not result.equals(observed.predict())


def test_default_follows_the_forecaster(class_case: Case, tmp_path: Path) -> None:
    """A class-probability forecaster defaults to ``class_proba``."""
    model = _pyfunc(class_case, tmp_path)
    assert model.predict({"y": class_case.next_rows}).equals(_observed_copy(class_case).predict_class_proba())


def test_unsupported_prediction_type(point_case: Case, tmp_path: Path) -> None:
    """A point-only forecaster refuses ``interval``, naming the supported types."""
    model = _pyfunc(point_case, tmp_path)
    with pytest.raises(ValueError, match=r"prediction_type='interval'.*\['point'\]"):
        model.predict({}, params={"prediction_type": "interval"})


def test_coverage_rates_need_interval(interval_case: Case, tmp_path: Path) -> None:
    """``coverage_rates`` with a non-interval prediction type is an error."""
    model = _pyfunc(interval_case, tmp_path)
    with pytest.raises(ValueError, match="coverage_rates only applies"):
        model.predict({}, params={"prediction_type": "point", "coverage_rates": [0.9]})


def test_params_reach_the_forecaster(point_case: Case, tmp_path: Path) -> None:
    """``forecasting_horizon`` changes the number of forecast steps."""
    assert HORIZON != 3
    model = _pyfunc(point_case, tmp_path)
    assert model.predict({}, params={"forecasting_horizon": 3}).height == 3


def test_omitted_list_params_use_yohou_defaults(interval_case: Case, tmp_path: Path) -> None:
    """Omitted ``coverage_rates`` and ``groups`` behave as yohou's ``None`` defaults."""
    model = _pyfunc(interval_case, tmp_path)
    result = model.predict({}, params={"prediction_type": "interval"})
    assert result.equals(interval_case.forecaster.predict_interval(coverage_rates=None, groups=None))


def test_groups_restrict_the_panel(panel_case: Case, tmp_path: Path) -> None:
    """``groups`` keeps only the named panel groups."""
    model = _pyfunc(panel_case, tmp_path)
    result = model.predict({"y": panel_case.next_rows}, params={"groups": ["a"]})
    assert [c for c in result.columns if "__" in c] == ["a__v"]
    assert result.equals(_observed_copy(panel_case).predict(groups=["a"]))


# -- Signature ----------------------------------------------------------------------


def test_params_only_signature(point_case: Case, tmp_path: Path) -> None:
    """The saved signature declares the four params and no inputs."""
    model = _pyfunc(point_case, tmp_path)
    signature = model.metadata.signature
    assert signature.inputs is None
    assert [p.name for p in signature.params.params] == list(yohou_mlflow.PARAM_NAMES)


def test_user_signature_without_params_gets_them(point_case: Case, tmp_path: Path) -> None:
    """A caller's signature with no params gains exactly the flavour params and keeps its outputs."""
    signature = ModelSignature(inputs=None, outputs=Schema([ColSpec("double", "v")]))
    model = _pyfunc(point_case, tmp_path, signature=signature)
    assert [p.name for p in model.metadata.signature.params.params] == list(yohou_mlflow.PARAM_NAMES)
    assert model.metadata.signature.outputs is not None


def test_user_signature_params_are_merged(point_case: Case, tmp_path: Path) -> None:
    """A caller's signature keeps its outputs and params and gains the flavour params."""
    signature = ModelSignature(
        inputs=None,
        outputs=Schema([ColSpec("double", "v")]),
        params=ParamSchema([ParamSpec("note", "string", "")]),
    )
    model = _pyfunc(point_case, tmp_path, signature=signature)
    names = [p.name for p in model.metadata.signature.params.params]
    assert names == ["note", *yohou_mlflow.PARAM_NAMES]
    assert model.metadata.signature.outputs is not None


def test_params_name_clash_refused(point_case: Case, tmp_path: Path) -> None:
    """A caller's param named like a flavour param is refused, and nothing is written."""
    signature = ModelSignature(inputs=None, params=ParamSchema([ParamSpec("prediction_type", "string", "x")]))
    path = tmp_path / "model"
    with pytest.raises(ValueError, match="prediction_type"):
        yohou_mlflow.save_model(point_case.forecaster, path, signature=signature)
    assert not path.exists()


def test_input_schema_refused(point_case: Case, tmp_path: Path) -> None:
    """A caller's signature with inputs is refused, and nothing is written."""
    signature = ModelSignature(inputs=Schema([ColSpec("double", "v")]))
    path = tmp_path / "model"
    with pytest.raises(ValueError, match="cannot declare an input schema"):
        yohou_mlflow.save_model(point_case.forecaster, path, signature=signature)
    assert not path.exists()


# -- Input contract -----------------------------------------------------------------


def test_frames_reach_the_forecaster_unchanged(
    point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four frames reach yohou as the very objects sent, dtypes included."""
    model = _pyfunc(point_case, tmp_path)
    t = point_case.next_rows["time"]
    sent = {
        "y": point_case.next_rows.with_columns(count=pl.Series(range(5), dtype=pl.Int64)),
        "X_actual": pl.DataFrame({"time": t, "flag": pl.Series([True] * 5)}),
        "X_future": pl.DataFrame({"time": t, "day": pl.Series([dt.date(2020, 1, 1)] * 5)}),
        "X_forecast": pl.DataFrame({
            "vintage_time": t,
            "time": t,
            "kind": pl.Series(["a"] * 5, dtype=pl.Categorical),
        }),
    }
    received: dict = {}

    def observe(self, y, *, X_actual=None, groups=None, X_future=None, X_forecast=None):
        received["observe"] = {"y": y, "X_actual": X_actual, "X_future": X_future, "X_forecast": X_forecast}
        return self

    def predict(self, *, X_future=None, X_forecast=None, forecasting_horizon=None, groups=None, **kwargs):
        received["predict"] = {"X_future": X_future, "X_forecast": X_forecast}
        return pl.DataFrame({"ok": [1]})

    monkeypatch.setattr(PointReductionForecaster, "observe", observe)
    monkeypatch.setattr(PointReductionForecaster, "predict", predict)
    result = model.predict(sent)
    for key, frame in sent.items():
        assert received["observe"][key] is frame
    assert received["predict"]["X_future"] is sent["X_future"]
    assert received["predict"]["X_forecast"] is sent["X_forecast"]
    assert isinstance(result, pl.DataFrame)


def test_output_is_polars(point_case: Case, tmp_path: Path) -> None:
    """The result is the polars frame yohou returns."""
    assert isinstance(_pyfunc(point_case, tmp_path).predict({}), pl.DataFrame)


@pytest.mark.parametrize(
    ("model_input", "error", "match"),
    [
        ({"Y": pl.DataFrame()}, ValueError, r"\['Y'\].*\['y', 'X_actual', 'X_future', 'X_forecast'\]"),
        ({"y": pd.DataFrame({"v": [1.0]})}, TypeError, "'y' must be a polars DataFrame"),
        ({"X_actual": pl.DataFrame()}, ValueError, "only accepted together with 'y'"),
        (pd.DataFrame({"v": [1.0]}), TypeError, "pandas and REST input is not supported yet"),
    ],
)
def test_invalid_input(point_case: Case, tmp_path: Path, model_input, error, match: str) -> None:
    """Each invalid input raises the specified error."""
    with pytest.raises(error, match=match):
        _pyfunc(point_case, tmp_path).predict(model_input)


# -- Load options through model_config ----------------------------------------------


def test_model_config_extra_trusted_types(local_ridge_case: Case, local_ridge_type: str, tmp_path: Path) -> None:
    """``extra_trusted_types`` in ``model_config`` loads a type outside the policy."""
    path = tmp_path / "model"
    yohou_mlflow.save_model(local_ridge_case.forecaster, path, extra_trusted_types=[local_ridge_type])
    with pytest.raises(UntrustedTypesError):
        mlflow.pyfunc.load_model(str(path))
    model = mlflow.pyfunc.load_model(str(path), model_config={"extra_trusted_types": [local_ridge_type]})
    assert model.predict({}).equals(local_ridge_case.forecaster.predict())


def test_model_config_unknown_key(point_case: Case, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """MLflow drops an unknown ``model_config`` key with a warning naming it; the load proceeds."""
    path = tmp_path / "model"
    yohou_mlflow.save_model(point_case.forecaster, path)
    model = mlflow.pyfunc.load_model(str(path), model_config={"trusted": []})
    assert "trusted" in caplog.text
    assert model.predict({}).equals(point_case.forecaster.predict())


def test_model_config_strict(point_case: Case, tmp_path: Path) -> None:
    """``strict`` in ``model_config`` turns a version mismatch from an error into a warning."""
    path = tmp_path / "model"
    yohou_mlflow.save_model(point_case.forecaster, path)
    mlmodel = yaml.safe_load((path / "MLmodel").read_text())
    mlmodel["flavors"]["yohou"]["versions"]["yohou"] = "0.0.0"
    (path / "MLmodel").write_text(yaml.safe_dump(mlmodel))
    with pytest.raises(VersionMismatchError):
        mlflow.pyfunc.load_model(str(path))
    with pytest.warns(VersionMismatchWarning):
        model = mlflow.pyfunc.load_model(str(path), model_config={"strict": False})
    assert model.predict({}).equals(point_case.forecaster.predict())


def test_saved_load_options_cannot_relax_strictness(point_case: Case, tmp_path: Path) -> None:
    """A model file that pre-sets ``strict: false`` is refused."""
    path = tmp_path / "model"
    yohou_mlflow.save_model(point_case.forecaster, path)
    mlmodel = yaml.safe_load((path / "MLmodel").read_text())
    mlmodel["flavors"]["python_function"][mlflow.pyfunc.MODEL_CONFIG]["strict"] = False
    (path / "MLmodel").write_text(yaml.safe_dump(mlmodel))
    with pytest.raises(YohouMlflowError, match="cannot choose its own trust or strictness"):
        mlflow.pyfunc.load_model(str(path))


def test_saved_load_options_cannot_extend_trust(local_ridge_case: Case, local_ridge_type: str, tmp_path: Path) -> None:
    """A model file that pre-sets ``extra_trusted_types`` is refused."""
    path = tmp_path / "model"
    yohou_mlflow.save_model(local_ridge_case.forecaster, path, extra_trusted_types=[local_ridge_type])
    mlmodel = yaml.safe_load((path / "MLmodel").read_text())
    mlmodel["flavors"]["python_function"][mlflow.pyfunc.MODEL_CONFIG]["extra_trusted_types"] = [local_ridge_type]
    (path / "MLmodel").write_text(yaml.safe_dump(mlmodel))
    with pytest.raises(YohouMlflowError, match="cannot choose its own trust"):
        mlflow.pyfunc.load_model(str(path))


# -- Prediction type discovery ------------------------------------------------------


class _UntaggedIntervalForecaster(BaseEstimator):
    """An estimator with no ``forecaster_type`` tag that defines ``predict_interval``."""

    def predict_interval(self):
        """Return nothing; only the method's presence matters."""


def test_prediction_types_fall_back_to_methods() -> None:
    """Without a ``forecaster_type`` tag, supported types come from the methods defined."""
    assert supported_prediction_types(_UntaggedIntervalForecaster()) == frozenset({"interval"})


def test_no_supported_prediction_type() -> None:
    """A forecaster supporting no prediction type has no default."""
    with pytest.raises(ValueError, match="supports none of the prediction types"):
        default_prediction_type(frozenset())
