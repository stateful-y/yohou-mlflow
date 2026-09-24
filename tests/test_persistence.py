"""Tests for saving, logging, loading and checking yohou forecasters."""

import copy
import re
import sys
import warnings
from pathlib import Path

import mlflow
import pytest
import skops.io
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from yohou.point import PointReductionForecaster

import yohou_mlflow
from conftest import (
    FAMILIES,
    Case,
    edit_flavor,
    make_case,
    make_series,
    read_flavor,
)
from yohou_mlflow import (
    FormatVersionError,
    SaveVerificationError,
    UntrustedTypesError,
    VersionMismatchError,
    VersionMismatchWarning,
    YohouMlflowError,
)
from yohou_mlflow._trust import is_trusted, untrusted_outside_policy
from yohou_mlflow._versions import compare_versions, installed_versions


def _predict(case: Case, forecaster):
    return getattr(forecaster, case.predict_method)()


def _saved(case: Case, tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "model"
    yohou_mlflow.save_model(case.forecaster, path, **kwargs)
    return path


# -- Save, log and load -------------------------------------------------------------


def test_round_trip(case: Case, tmp_path: Path) -> None:
    """A loaded forecaster predicts what the saved one predicted."""
    loaded = yohou_mlflow.load_model(str(_saved(case, tmp_path)))
    assert _predict(case, loaded).equals(_predict(case, case.forecaster))


def test_continues_after_observe(case: Case, tmp_path: Path) -> None:
    """Loaded and original forecasters agree after observing the same new rows."""
    loaded = yohou_mlflow.load_model(str(_saved(case, tmp_path)))
    original = copy.deepcopy(case.forecaster)
    for forecaster in (original, loaded):
        forecaster.observe(case.next_rows)
    assert _predict(case, loaded).equals(_predict(case, original))


def test_rewind_after_loading(case: Case, tmp_path: Path) -> None:
    """Loaded and original forecasters agree after rewinding to the same window."""
    loaded = yohou_mlflow.load_model(str(_saved(case, tmp_path)))
    original = copy.deepcopy(case.forecaster)
    for forecaster in (original, loaded):
        forecaster.rewind(case.rewind_window)
    assert _predict(case, loaded).equals(_predict(case, original))


def test_registry_round_trip(case: Case, tracking: str) -> None:
    """A registered model loads back from ``models:/<name>/1``."""
    with mlflow.start_run():
        yohou_mlflow.log_model(case.forecaster, name="forecaster", registered_model_name="m")
    loaded = yohou_mlflow.load_model("models:/m/1")
    assert _predict(case, loaded).equals(_predict(case, case.forecaster))


def test_log_model_returns_model_info(point_case: Case, tracking: str) -> None:
    """``log_model`` returns MLflow's ModelInfo, whose URI loads the forecaster."""
    with mlflow.start_run():
        info = yohou_mlflow.log_model(point_case.forecaster, name="forecaster")
    loaded = yohou_mlflow.load_model(info.model_uri)
    assert loaded.predict().equals(point_case.forecaster.predict())


def test_unfitted_forecaster_rejected(tmp_path: Path) -> None:
    """An unfitted forecaster is refused before anything is written."""
    path = tmp_path / "model"
    with pytest.raises(NotFittedError):
        yohou_mlflow.save_model(PointReductionForecaster(estimator=Ridge()), path)
    assert not path.exists()


def test_non_yohou_object_rejected(tmp_path: Path) -> None:
    """Only yohou forecasters are accepted."""
    with pytest.raises(TypeError, match="expects a yohou forecaster"):
        yohou_mlflow.save_model(Ridge(), tmp_path / "model")


# -- Trust policy -------------------------------------------------------------------


def test_policy_trusts_yohou_polars_and_datetime() -> None:
    """The built-in policy covers what a yohou forecaster holds, and nothing else."""
    assert is_trusted("yohou.point.reduction.PointReductionForecaster")
    assert is_trusted("polars.datatypes.classes.Float64")
    assert is_trusted("polars.dataframe.frame.DataFrame")
    assert is_trusted("zoneinfo.ZoneInfo")
    assert is_trusted("datetime.timezone")
    assert is_trusted("sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor")
    assert not is_trusted("yohou_mlflow.anything")
    assert not is_trusted("builtins.eval")
    assert untrusted_outside_policy(["os.system", "datetime.date"], ["os.system"]) == []


def test_gradient_boosting_needs_no_extra_types(tmp_path: Path) -> None:
    """scikit-learn internals skops leaves untrusted, such as TreePredictor, are trusted."""
    data = make_series()
    forecaster = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=10), reduction_strategy="direct"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(data[:180], forecasting_horizon=5)
    yohou_mlflow.save_model(forecaster, tmp_path / "model")
    assert yohou_mlflow.load_model(str(tmp_path / "model")).predict().equals(forecaster.predict())


def test_user_functions_stay_untrusted() -> None:
    """A function from the user's code is its own entry, which no policy prefix matches."""
    from local_estimators import identity

    types = skops.io.get_untrusted_types(data=skops.io.dumps(FunctionTransformer(identity)))
    assert untrusted_outside_policy(types) == ["local_estimators.identity"]


def test_untrusted_type_refused_at_save(local_ridge_case: Case, local_ridge_type: str, tmp_path: Path) -> None:
    """A type outside the policy is reported when saving, and nothing is kept."""
    path = tmp_path / "model"
    with pytest.raises(UntrustedTypesError, match=re.escape(local_ridge_type)) as info:
        yohou_mlflow.save_model(local_ridge_case.forecaster, path)
    assert info.value.types == (local_ridge_type,)
    assert not path.exists()


def test_untrusted_type_refused_at_load_without_constructing(
    local_ridge_case: Case, local_ridge_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading refuses a type the caller does not trust before building any object."""
    path = _saved(local_ridge_case, tmp_path, extra_trusted_types=[local_ridge_type])

    def forbidden(*args, **kwargs):
        raise AssertionError("skops.io.load must not be called for an untrusted model")

    monkeypatch.setattr(skops.io, "load", forbidden)
    with pytest.raises(UntrustedTypesError, match="extra_trusted_types"):
        yohou_mlflow.load_model(str(path))


def test_extra_trusted_types_at_load(local_ridge_case: Case, local_ridge_type: str, tmp_path: Path) -> None:
    """The caller can trust an extra type at load time."""
    path = _saved(local_ridge_case, tmp_path, extra_trusted_types=[local_ridge_type])
    loaded = yohou_mlflow.load_model(str(path), extra_trusted_types=[local_ridge_type])
    assert loaded.predict().equals(local_ridge_case.forecaster.predict())


def test_tampered_type_list_does_not_extend_trust(
    local_ridge_case: Case, local_ridge_type: str, tmp_path: Path
) -> None:
    """Editing the recorded type list in ``MLmodel`` grants no trust."""
    path = _saved(local_ridge_case, tmp_path, extra_trusted_types=[local_ridge_type])
    edit_flavor(path, recorded_types=[local_ridge_type], trusted_types=[local_ridge_type])
    with pytest.raises(UntrustedTypesError, match=re.escape(local_ridge_type)):
        yohou_mlflow.load_model(str(path))


def test_component_path_cannot_leave_the_model(point_case: Case, tmp_path: Path) -> None:
    """A component file name from the model file cannot point outside the model."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, components={"forecaster": "../outside.skops"})
    with pytest.raises(YohouMlflowError, match="invalid file"):
        yohou_mlflow.load_model(str(path))


# -- Versions -----------------------------------------------------------------------


def test_compare_versions_rules() -> None:
    """yohou must match exactly; scikit-learn and polars must match major and minor."""
    base = {"yohou": "0.1.0a13", "scikit-learn": "1.9.1", "polars": "1.44.2", "skops": "0.15.0"}
    assert compare_versions(base, dict(base, skops="0.1.0", **{"yohou-mlflow": "9"})) == []
    assert compare_versions(base, dict(base, **{"scikit-learn": "1.9.7", "yohou-mlflow": "9"})) == []
    mismatches = compare_versions(
        base, dict(base, yohou="0.1.0a14", polars="1.45.0", **{"scikit-learn": "2.0.0", "yohou-mlflow": "9"})
    )
    assert [(m.package, m.rule) for m in mismatches] == [
        ("yohou", "exact"),
        ("scikit-learn", "major.minor"),
        ("polars", "major.minor"),
    ]


def test_different_yohou_version_refused(point_case: Case, tmp_path: Path) -> None:
    """A model saved under another yohou version is refused, naming both versions."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, versions=dict(read_flavor(path)["versions"], yohou="0.0.0"))
    installed = installed_versions()["yohou"]
    with pytest.raises(VersionMismatchError, match=rf"yohou: saved with 0\.0\.0, installed {re.escape(installed)}"):
        yohou_mlflow.load_model(str(path))


def test_patch_level_scikit_learn_difference_accepted(point_case: Case, tmp_path: Path) -> None:
    """Only the patch version of scikit-learn differs: the model loads."""
    path = _saved(point_case, tmp_path)
    major, minor = installed_versions()["scikit-learn"].split(".")[:2]
    edit_flavor(path, versions=dict(read_flavor(path)["versions"], **{"scikit-learn": f"{major}.{minor}.999"}))
    loaded = yohou_mlflow.load_model(str(path))
    assert loaded.predict().equals(point_case.forecaster.predict())


def test_non_strict_load_warns(point_case: Case, tmp_path: Path) -> None:
    """``strict=False`` turns the mismatch error into a warning with the same text."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, versions=dict(read_flavor(path)["versions"], yohou="0.0.0"))
    with pytest.raises(VersionMismatchError) as error:
        yohou_mlflow.load_model(str(path))
    with pytest.warns(VersionMismatchWarning) as record:
        loaded = yohou_mlflow.load_model(str(path), strict=False)
    assert str(record[0].message) == str(error.value)
    assert loaded.predict().equals(point_case.forecaster.predict())


# -- Compatibility check ------------------------------------------------------------


def test_check_compatible_model(point_case: Case, tmp_path: Path) -> None:
    """A model saved in this environment is reported loadable."""
    report = yohou_mlflow.check_compatibility(str(_saved(point_case, tmp_path)))
    assert report.loadable
    assert report.problems == ()
    assert report.format_version == "1.0"


def test_check_reports_version_mismatch(point_case: Case, tmp_path: Path) -> None:
    """A yohou mismatch is reported with both versions."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, versions=dict(read_flavor(path)["versions"], yohou="0.0.0"))
    report = yohou_mlflow.check_compatibility(str(path))
    assert not report.loadable
    assert [m.package for m in report.version_mismatches] == ["yohou"]
    assert "strict load would refuse" in report.problems[0]
    assert "saved with 0.0.0" in report.problems[0]


def test_check_reports_untrusted_type(local_ridge_case: Case, local_ridge_type: str, tmp_path: Path) -> None:
    """A type outside the policy is reported, and trusting it clears the report."""
    path = _saved(local_ridge_case, tmp_path, extra_trusted_types=[local_ridge_type])
    report = yohou_mlflow.check_compatibility(str(path))
    assert not report.loadable
    assert report.untrusted_types == (local_ridge_type,)
    assert yohou_mlflow.check_compatibility(str(path), [local_ridge_type]).loadable


def test_check_does_not_deserialize(point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check never calls a skops load function."""
    path = _saved(point_case, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("check_compatibility must not deserialize the forecaster")

    monkeypatch.setattr(skops.io, "load", forbidden)
    monkeypatch.setattr(skops.io, "loads", forbidden)
    assert yohou_mlflow.check_compatibility(str(path)).loadable


def test_check_reports_unsupported_format(point_case: Case, tmp_path: Path) -> None:
    """An unreadable format version is reported rather than raised."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, format_version="2.0")
    report = yohou_mlflow.check_compatibility(str(path))
    assert not report.loadable
    assert "2.0" in report.problems[0]


# -- Layout -------------------------------------------------------------------------


def test_layout(point_case: Case, tmp_path: Path) -> None:
    """``MLmodel`` records the format version and one ``forecaster`` component."""
    path = _saved(point_case, tmp_path)
    flavor = read_flavor(path)
    assert flavor["format_version"] == "1.0"
    assert flavor["components"] == {"forecaster": "forecaster.skops"}
    assert (path / "forecaster.skops").is_file()
    assert flavor["forecaster_class"] == "yohou.point.reduction.PointReductionForecaster"
    assert flavor["forecaster_type"] == ["point"]
    assert set(flavor["versions"]) == {"yohou", "scikit-learn", "polars", "skops", "yohou-mlflow"}


def test_unsupported_format_version_refused(point_case: Case, tmp_path: Path) -> None:
    """A newer major format version is refused, naming both versions."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, format_version="2.0")
    with pytest.raises(FormatVersionError, match=r"2\.0.*1\.0.*Upgrade yohou-mlflow"):
        yohou_mlflow.load_model(str(path))


def test_default_requirements_pin_saving_environment(point_case: Case, tmp_path: Path) -> None:
    """``requirements.txt`` pins the five packages the model depends on."""
    lines = (_saved(point_case, tmp_path) / "requirements.txt").read_text().splitlines()
    names = {re.split(r"==", line)[0].replace("_", "-").lower() for line in lines if "==" in line}
    assert {"yohou-mlflow", "yohou", "scikit-learn", "polars", "skops"} <= names


def test_pip_requirements_override(point_case: Case, tmp_path: Path) -> None:
    """``pip_requirements`` replaces the defaults, ``extra_pip_requirements`` adds to them."""
    replaced = _saved(point_case, tmp_path / "a", pip_requirements=["yohou"]) / "requirements.txt"
    assert "skops" not in replaced.read_text()
    added = _saved(point_case, tmp_path / "b", extra_pip_requirements=["lightgbm"]) / "requirements.txt"
    assert "lightgbm" in added.read_text()
    assert "skops" in added.read_text()


# -- Save-time verification ---------------------------------------------------------


def test_verification_passes(point_case: Case, tmp_path: Path) -> None:
    """A normal save completes and keeps the model directory."""
    assert (_saved(point_case, tmp_path) / "MLmodel").is_file()


def test_verification_load_failure(point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that cannot be loaded back is refused, and nothing is kept."""

    def failing(*args, **kwargs):
        raise RuntimeError("cannot rebuild")

    monkeypatch.setattr(skops.io, "loads", failing)
    path = tmp_path / "model"
    with pytest.raises(SaveVerificationError, match="could not be loaded back"):
        yohou_mlflow.save_model(point_case.forecaster, path)
    assert not path.exists()


def test_verification_mismatch(point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that loads back but predicts differently is refused, and nothing is kept."""
    other = PointReductionForecaster(estimator=Ridge()).fit(make_series()[:150], forecasting_horizon=5)
    monkeypatch.setattr(skops.io, "loads", lambda *args, **kwargs: other)
    path = tmp_path / "model"
    with pytest.raises(SaveVerificationError, match="predicts differently"):
        yohou_mlflow.save_model(point_case.forecaster, path)
    assert not path.exists()


def test_verification_reloaded_predict_failure(
    point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reloaded model that fails to predict is refused with a named cause, and nothing is kept."""

    class _Broken:
        def predict(self):
            raise KeyError("missing state")

    monkeypatch.setattr(skops.io, "loads", lambda *args, **kwargs: _Broken())
    path = tmp_path / "model"
    with pytest.raises(SaveVerificationError, match="reloaded model raised KeyError"):
        yohou_mlflow.save_model(point_case.forecaster, path)
    assert not path.exists()


def test_load_never_imports_code_from_the_model(point_case: Case, tmp_path: Path) -> None:
    """A ``code`` directory named in the model file is never put on the import path."""
    path = _saved(point_case, tmp_path)
    package = path / "code" / "planted_by_model"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    edit_flavor(path, code="code")
    before = list(sys.path)
    yohou_mlflow.load_model(str(path))
    assert sys.path == before
    assert str(path / "code") not in sys.path
    assert "planted_by_model" not in sys.modules


def test_verification_without_predictable_original(
    point_case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the original cannot predict without inputs, the save warns and completes."""

    def needs_inputs(self, *args, **kwargs):
        raise ValueError("X_future is required")

    monkeypatch.setattr(PointReductionForecaster, "predict", needs_inputs)
    with pytest.warns(UserWarning, match="compared no predictions"):
        path = _saved(point_case, tmp_path)
    assert (path / "MLmodel").is_file()


@pytest.mark.parametrize("time_zone", ["UTC", "Etc/GMT-2"])
def test_time_zone_aware_forecaster(time_zone: str, tmp_path: Path) -> None:
    """A forecaster fitted on time-zone-aware data round-trips, time zone included."""
    data = make_series(time_zone=time_zone)
    forecaster = PointReductionForecaster(estimator=Ridge())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.fit(data[:180], forecasting_horizon=5)
        forecaster.observe(data[180:190])
    yohou_mlflow.save_model(forecaster, tmp_path / "model")
    loaded = yohou_mlflow.load_model(str(tmp_path / "model"))
    assert loaded.observed_time_ == forecaster.observed_time_
    assert str(loaded.observed_time_.tzinfo) == time_zone
    assert loaded.predict().equals(forecaster.predict())


def test_every_family_saves_within_policy() -> None:
    """Each test family's forecaster holds only types the policy trusts."""
    for family in FAMILIES:
        types = skops.io.get_untrusted_types(data=skops.io.dumps(make_case(family).forecaster))
        assert untrusted_outside_policy(types) == [], family


# -- Options and edge cases ---------------------------------------------------------


def test_metadata_is_stored(point_case: Case, tmp_path: Path) -> None:
    """``metadata`` lands in the ``MLmodel`` file."""
    path = _saved(point_case, tmp_path, metadata={"owner": "forecasting"})
    assert yaml.safe_load((path / "MLmodel").read_text())["metadata"] == {"owner": "forecasting"}


def test_conda_env_replaces_the_default_environment(point_case: Case, tmp_path: Path) -> None:
    """A ``conda_env`` is written as given, with its pip section as the requirements."""
    conda_env = {"name": "forecasting", "dependencies": ["python=3.12", {"pip": ["yohou_mlflow", "yohou"]}]}
    path = _saved(point_case, tmp_path, conda_env=conda_env)
    assert yaml.safe_load((path / "conda.yaml").read_text())["name"] == "forecasting"
    requirements = (path / "requirements.txt").read_text().splitlines()
    # MLflow adds its own pin; the rest is the conda_env's pip section, not the defaults.
    assert [r for r in requirements if not r.startswith("mlflow==")] == ["yohou_mlflow", "yohou"]


def test_constraints_file_is_written(point_case: Case, tmp_path: Path) -> None:
    """A ``-c`` entry in the requirements produces ``constraints.txt``."""
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("polars<2\n")
    path = _saved(point_case, tmp_path, extra_pip_requirements=[f"-c {constraints}"])
    assert (path / "constraints.txt").read_text() == "polars<2"
    assert "-c constraints.txt" in (path / "requirements.txt").read_text()


def test_non_numeric_format_version_refused(point_case: Case, tmp_path: Path) -> None:
    """A format version that does not start with a number is refused."""
    path = _saved(point_case, tmp_path)
    edit_flavor(path, format_version="next")
    with pytest.raises(FormatVersionError, match="next"):
        yohou_mlflow.load_model(str(path))


def test_unparseable_versions_compare_as_text() -> None:
    """A version that is not PEP 440 is compared as a whole string."""
    base = {"yohou": "0.1.0a13", "scikit-learn": "1.9.1", "polars": "main", "skops": "0.15.0", "yohou-mlflow": "9"}
    assert compare_versions({"polars": "main"}, base) == []
    assert [m.package for m in compare_versions({"polars": "dev"}, base)] == ["polars"]


def test_compatibility_report_text(point_case: Case, tmp_path: Path) -> None:
    """The report states loadability, or lists each problem."""
    path = _saved(point_case, tmp_path)
    assert str(yohou_mlflow.check_compatibility(str(path))) == "Loadable: a strict load_model would succeed."
    edit_flavor(path, versions=dict(read_flavor(path)["versions"], yohou="0.0.0"))
    text = str(yohou_mlflow.check_compatibility(str(path)))
    assert text.startswith("Not loadable:\n  - A strict load would refuse the model")
