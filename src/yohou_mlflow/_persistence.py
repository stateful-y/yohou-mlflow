"""Save, log, load and check yohou forecasters as MLflow models.

The saved model directory holds an ``MLmodel`` file and one skops file per component.
Format version 1 has a single component, ``forecaster``, holding the whole fitted
forecaster including the state it accumulated through ``observe``.

The ``mlflow.utils`` helpers imported below are private to MLflow, but they are the
ones MLflow's own flavours use to write environments and resolve paths. They were
checked to exist with the same signatures on mlflow-skinny 3.0.0 and 3.16.1.
"""

from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import skops.io
import yaml
from mlflow import pyfunc
from mlflow.artifacts import download_artifacts
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME, ModelInfo
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from sklearn.utils.validation import check_is_fitted

from yohou_mlflow._exceptions import (
    FormatVersionError,
    SaveVerificationError,
    UntrustedTypesError,
    VersionMismatchError,
    VersionMismatchWarning,
    YohouMlflowError,
)
from yohou_mlflow._pyfunc import (
    _PREDICT_METHODS,
    YohouPyfuncModel,
    default_prediction_type,
    merge_signature,
    supported_prediction_types,
)
from yohou_mlflow._trust import untrusted_outside_policy
from yohou_mlflow._versions import VersionMismatch, compare_versions, describe_mismatches, installed_versions

FLAVOR_NAME = "yohou"
"""Name of the flavour in a model's ``MLmodel`` file."""

FORMAT_VERSION = "1.0"
"""On-disk format this release writes. A reader refuses any other major number."""

_SUPPORTED_FORMAT_MAJOR = 1
# pyfunc load options and the only values a saved model may carry for them.
_SAVED_LOAD_OPTIONS: dict[str, Any] = {"extra_trusted_types": [], "strict": True}
_FORECASTER_COMPONENT = "forecaster"
_FORECASTER_FILE = "forecaster.skops"

# (distribution, importable module) for each requirement pinned by default.
_PINNED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("yohou_mlflow", "yohou_mlflow"),
    ("yohou", "yohou"),
    ("scikit-learn", "sklearn"),
    ("polars", "polars"),
    ("skops", "skops"),
)


def get_default_pip_requirements() -> list[str]:
    """Return the pip requirements a saved model gets unless the caller overrides them.

    Returns
    -------
    list of str
        yohou-mlflow, yohou, scikit-learn, polars and skops, each pinned to its
        installed version.
    """
    return [_get_pinned_requirement(name, module=module) for name, module in _PINNED_PACKAGES]


def save_model(
    forecaster: Any,
    path: str | os.PathLike[str],
    *,
    extra_trusted_types: Iterable[str] | None = None,
    signature: ModelSignature | None = None,
    conda_env: Any = None,
    mlflow_model: Model | None = None,
    pip_requirements: Any = None,
    extra_pip_requirements: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a fitted yohou forecaster as an MLflow model directory.

    The forecaster is written with skops, never pickle. Before anything is kept, the
    written file is checked against the trust policy and loaded back, and its
    predictions are compared with the original's. Any failure removes the directory.

    Parameters
    ----------
    forecaster : yohou forecaster
        A fitted yohou forecaster, including the state it accumulated through
        ``observe``.
    path : str or path-like
        Directory to create. It must not exist, or must be empty.
    extra_trusted_types : iterable of str or None, default=None
        Exact type names to trust in addition to the built-in policy, for example a
        third-party regressor inside the forecaster. They are not stored: pass the
        same names every time the model is loaded.
    signature : mlflow.models.ModelSignature or None, default=None
        Optional signature. It may carry outputs and extra params, but no inputs.
        The flavour's four params are always added.
    conda_env : dict, str or None, default=None
        Conda environment, as for MLflow's built-in flavours.
    mlflow_model : mlflow.models.Model or None, default=None
        Model configuration to add the flavour to. Used by ``log_model``.
    pip_requirements : iterable of str, str or None, default=None
        Replaces the default pip requirements.
    extra_pip_requirements : iterable of str, str or None, default=None
        Added to the default pip requirements.
    metadata : dict or None, default=None
        Custom metadata stored in the ``MLmodel`` file.

    Raises
    ------
    TypeError
        If ``forecaster`` is not a yohou forecaster.
    sklearn.exceptions.NotFittedError
        If ``forecaster`` is not fitted. Nothing is written.
    ValueError
        If ``signature`` has an input schema or redefines a flavour param.
    UntrustedTypesError
        If the forecaster holds types outside the trust policy and
        ``extra_trusted_types``.
    SaveVerificationError
        If the written model cannot be loaded back, fails to predict once reloaded, or
        loads back with different predictions, including forecasters fitted on time-zone-aware data while
        skops cannot rebuild ``zoneinfo.ZoneInfo``.

    See Also
    --------
    log_model : Save the forecaster into an MLflow run and optionally register it.
    load_model : Load a saved forecaster.
    """
    from yohou.base import BaseForecaster

    if not isinstance(forecaster, BaseForecaster):
        msg = (
            f"save_model expects a yohou forecaster, got {type(forecaster).__module__}.{type(forecaster).__qualname__}."
        )
        raise TypeError(msg)
    check_is_fitted(forecaster)
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    supported = supported_prediction_types(forecaster)
    default_type = default_prediction_type(supported)
    signature = merge_signature(signature, default_type)

    target = os.path.abspath(os.fspath(path))
    _validate_and_prepare_target_save_path(target)
    try:
        _write_model(
            forecaster,
            target,
            extra_trusted_types=extra_trusted_types,
            signature=signature,
            supported=supported,
            default_type=default_type,
            conda_env=conda_env,
            mlflow_model=mlflow_model,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _write_model(
    forecaster: Any,
    path: str,
    *,
    extra_trusted_types: Iterable[str] | None,
    signature: ModelSignature,
    supported: frozenset[str],
    default_type: str,
    conda_env: Any,
    mlflow_model: Model | None,
    pip_requirements: Any,
    extra_pip_requirements: Any,
    metadata: dict[str, Any] | None,
) -> None:
    file = os.path.join(path, _FORECASTER_FILE)
    skops.io.dump(forecaster, file, compression=zipfile.ZIP_DEFLATED)
    data = Path(file).read_bytes()
    types = skops.io.get_untrusted_types(data=data)
    outside = untrusted_outside_policy(types, extra_trusted_types)
    if outside:
        raise UntrustedTypesError(outside, when="save")
    _verify_round_trip(forecaster, data, types, default_type)

    if mlflow_model is None:
        mlflow_model = Model()
    mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        format_version=FORMAT_VERSION,
        components={_FORECASTER_COMPONENT: _FORECASTER_FILE},
        versions=installed_versions(),
        # Informational only: loading never extends trust from this list.
        recorded_types=sorted(types),
        forecaster_class=f"{type(forecaster).__module__}.{type(forecaster).__qualname__}",
        forecaster_type=sorted(supported),
        default_prediction_type=default_type,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="yohou_mlflow",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        # MLflow only lets `mlflow.pyfunc.load_model(..., model_config=...)` override
        # keys the model was saved with, so the load options are saved here with their
        # safe values. `_load_pyfunc` refuses a file that changed them.
        model_config=dict(_SAVED_LOAD_OPTIONS),
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
    _write_environment(path, conda_env, pip_requirements, extra_pip_requirements)


def _verify_round_trip(original: Any, data: bytes, types: list[str], default_type: str) -> None:
    """Load the written bytes back and compare predictions with the original."""
    try:
        loaded = skops.io.loads(data, trusted=list(types))
    except Exception as exc:
        if "zoneinfo.ZoneInfo" in types:
            msg = (
                "The forecaster was fitted on time-zone-aware data, and the installed skops "
                "cannot rebuild the zoneinfo.ZoneInfo time zones it holds, so the saved model "
                f"could not be loaded back ({type(exc).__name__}: {exc}). Nothing was saved. "
                "Fit on time-zone-naive data, or upgrade skops once a release supports ZoneInfo."
            )
        else:
            msg = f"The saved forecaster could not be loaded back ({type(exc).__name__}: {exc}). Nothing was saved."
        raise SaveVerificationError(msg) from exc

    method = _PREDICT_METHODS[default_type]
    try:
        expected = getattr(original, method)()
    except Exception as exc:  # noqa: BLE001 - any failure here means there is nothing to compare
        warnings.warn(
            f"Save-time verification loaded the model back but compared no predictions: "
            f"{method}() without inputs raised {type(exc).__name__} on the original forecaster.",
            UserWarning,
            stacklevel=3,
        )
        return
    try:
        actual = getattr(loaded, method)()
    except Exception as exc:
        msg = (
            f"The saved forecaster loaded back, but the reloaded model raised {type(exc).__name__} "
            f"in {method}() ({exc}). Nothing was saved."
        )
        raise SaveVerificationError(msg) from exc
    if not actual.equals(expected):
        msg = (
            f"The saved forecaster loaded back, but the reloaded model predicts differently "
            f"from the original ({method}() results differ). Nothing was saved."
        )
        raise SaveVerificationError(msg)


def _write_environment(path: str, conda_env: Any, pip_requirements: Any, extra_pip_requirements: Any) -> None:
    if conda_env is None:
        default_requirements = get_default_pip_requirements() if pip_requirements is None else None
        conda_env, pip_reqs, constraints = _process_pip_requirements(
            default_requirements, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_reqs, constraints = _process_conda_env(conda_env)
    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w", encoding="utf-8") as stream:
        yaml.safe_dump(conda_env, stream=stream, default_flow_style=False)
    if constraints:
        Path(path, _CONSTRAINTS_FILE_NAME).write_text("\n".join(constraints), encoding="utf-8")
    Path(path, _REQUIREMENTS_FILE_NAME).write_text("\n".join(pip_reqs), encoding="utf-8")
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def log_model(
    forecaster: Any,
    name: str | None = None,
    *,
    registered_model_name: str | None = None,
    await_registration_for: int = 300,
    extra_trusted_types: Iterable[str] | None = None,
    signature: ModelSignature | None = None,
    conda_env: Any = None,
    pip_requirements: Any = None,
    extra_pip_requirements: Any = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelInfo:
    """Log a fitted yohou forecaster as an MLflow model, optionally registering it.

    Parameters
    ----------
    forecaster : yohou forecaster
        A fitted yohou forecaster.
    name : str or None, default=None
        Name of the logged model, as for MLflow's built-in flavours.
    registered_model_name : str or None, default=None
        When given, also create a version of this registered model.
    await_registration_for : int, default=300
        Seconds to wait for the model version to become ready.
    extra_trusted_types : iterable of str or None, default=None
        See `save_model`.
    signature : mlflow.models.ModelSignature or None, default=None
        See `save_model`.
    conda_env : dict, str or None, default=None
        See `save_model`.
    pip_requirements : iterable of str, str or None, default=None
        See `save_model`.
    extra_pip_requirements : iterable of str, str or None, default=None
        See `save_model`.
    metadata : dict or None, default=None
        Custom metadata stored in the ``MLmodel`` file.
    **kwargs : Any
        Passed to ``mlflow.models.Model.log``.

    Returns
    -------
    mlflow.models.model.ModelInfo
        Metadata of the logged model, including its ``model_uri``.

    See Also
    --------
    save_model : The checks every logged model goes through.
    load_model : Load a logged or registered forecaster.
    """
    import yohou_mlflow

    return Model.log(
        artifact_path=None,
        name=name,
        flavor=yohou_mlflow,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        metadata=metadata,
        forecaster=forecaster,
        extra_trusted_types=extra_trusted_types,
        signature=signature,
        conda_env=conda_env,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def load_model(
    model_uri: str,
    dst_path: str | None = None,
    *,
    extra_trusted_types: Iterable[str] | None = None,
    strict: bool = True,
) -> Any:
    """Load a yohou forecaster saved by this flavour.

    Checks run in this order, and each one stops the load before anything from the
    file is constructed: the format version, the package versions, then the types in
    the skops file against the trust policy.

    Parameters
    ----------
    model_uri : str
        Any MLflow model URI, for example ``"models:/name/1"`` or a local path.
    dst_path : str or None, default=None
        Local directory to download the model into.
    extra_trusted_types : iterable of str or None, default=None
        Exact type names to trust in addition to the built-in policy. The saved model
        cannot add to this list.
    strict : bool, default=True
        When ``True``, a package version mismatch raises `VersionMismatchError`. When
        ``False``, it emits a `VersionMismatchWarning` and the load proceeds.

    Returns
    -------
    yohou forecaster
        The forecaster, in the state it was saved in.

    Raises
    ------
    FormatVersionError
        If the model was saved in a format version this release cannot read.
    VersionMismatchError
        If ``strict`` and the installed versions break the comparison rules.
    UntrustedTypesError
        If the skops file holds types outside the policy and ``extra_trusted_types``.

    See Also
    --------
    check_compatibility : Run the same checks without loading anything.
    """
    local = download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
    forecaster, _ = _load_local(local, extra_trusted_types=extra_trusted_types, strict=strict)
    return forecaster


def _load_local(local: str, *, extra_trusted_types: Iterable[str] | None, strict: bool) -> tuple[Any, dict[str, Any]]:
    conf = _get_flavor_configuration(local, FLAVOR_NAME)
    _check_format_version(conf)
    # No `code` directory from the model is ever put on the import path: a saved model
    # could otherwise ship a package named like a trusted one (for example `yohou`) and
    # have it imported while skops resolves the types below.
    mismatches = compare_versions(conf.get("versions", {}))
    if mismatches:
        if strict:
            raise VersionMismatchError(mismatches)
        warnings.warn(describe_mismatches(mismatches), VersionMismatchWarning, stacklevel=3)
    file = _component_path(local, conf, _FORECASTER_COMPONENT)
    types = skops.io.get_untrusted_types(file=file)
    outside = untrusted_outside_policy(types, extra_trusted_types)
    if outside:
        raise UntrustedTypesError(outside, when="load")
    return skops.io.load(file, trusted=list(types)), conf


def _check_format_version(conf: dict[str, Any]) -> str:
    text = str(conf.get("format_version", ""))
    try:
        major = int(text.partition(".")[0])
    except ValueError:
        major = None
    if major != _SUPPORTED_FORMAT_MAJOR:
        msg = (
            f"This model was saved in yohou flavour format version {text or '(missing)'}, and this "
            f"yohou-mlflow reads format version {FORMAT_VERSION} (major {_SUPPORTED_FORMAT_MAJOR}). "
            "Upgrade yohou-mlflow to load it."
        )
        raise FormatVersionError(msg)
    return text


def _component_path(local: str, conf: dict[str, Any], component: str) -> str:
    name = str(conf.get("components", {}).get(component, ""))
    # The name comes from the model file, so it must not reach outside the model.
    if not name or os.path.basename(name) != name or name in {".", ".."}:
        msg = f"The model's {component!r} component names an invalid file: {name!r}."
        raise YohouMlflowError(msg)
    return os.path.join(local, name)


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of `check_compatibility`.

    Parameters
    ----------
    loadable : bool
        ``True`` when a default, strict ``load_model`` would load the model.
    problems : tuple of str
        One sentence per reason the load would be refused. Empty when loadable.
    format_version : str
        Format version recorded in the model.
    version_mismatches : tuple of VersionMismatch, default=()
        Packages whose installed version breaks the comparison rules.
    untrusted_types : tuple of str, default=()
        Types in the skops file outside the policy and ``extra_trusted_types``.
    """

    loadable: bool
    problems: tuple[str, ...]
    format_version: str
    version_mismatches: tuple[VersionMismatch, ...] = field(default=())
    untrusted_types: tuple[str, ...] = field(default=())

    def __str__(self) -> str:
        """Summarize the report in a few lines.

        Returns
        -------
        str
            "Loadable." or the list of problems.
        """
        if self.loadable:
            return "Loadable: a strict load_model would succeed."
        return "Not loadable:\n" + "\n".join(f"  - {problem}" for problem in self.problems)


def check_compatibility(
    model_uri: str,
    extra_trusted_types: Iterable[str] | None = None,
    *,
    dst_path: str | None = None,
) -> CompatibilityReport:
    """Report whether a saved forecaster would load here, without loading it.

    Reads only the ``MLmodel`` file and the schema stored in the skops file. No object
    from the saved forecaster is constructed, so this is safe to run on an untrusted
    model, for example in a pre-deploy check.

    Parameters
    ----------
    model_uri : str
        Any MLflow model URI.
    extra_trusted_types : iterable of str or None, default=None
        Exact type names the eventual load will trust in addition to the policy.
    dst_path : str or None, default=None
        Local directory to download the model into.

    Returns
    -------
    CompatibilityReport
        ``loadable`` and one problem per reason a strict load would be refused.

    See Also
    --------
    load_model : Load the forecaster once the report is clean.
    """
    local = download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
    conf = _get_flavor_configuration(local, FLAVOR_NAME)
    try:
        format_version = _check_format_version(conf)
    except FormatVersionError as exc:
        # Nothing else in a format this release cannot read can be trusted to mean
        # what this release thinks it means.
        return CompatibilityReport(False, (str(exc),), str(conf.get("format_version", "")))

    problems = []
    mismatches = tuple(compare_versions(conf.get("versions", {})))
    problems.extend(f"A strict load would refuse the model, version mismatch: {m}" for m in mismatches)
    types = skops.io.get_untrusted_types(file=_component_path(local, conf, _FORECASTER_COMPONENT))
    outside = tuple(untrusted_outside_policy(types, extra_trusted_types))
    if outside:
        problems.append(
            f"A load would refuse the model, types outside the trust policy: {list(outside)}. "
            "Pass them in extra_trusted_types if you trust them."
        )
    return CompatibilityReport(not problems, tuple(problems), format_version, mismatches, outside)


def _load_pyfunc(path: str, model_config: dict[str, Any] | None = None) -> YohouPyfuncModel:
    """Load the pyfunc model; called by ``mlflow.pyfunc.load_model``.

    Parameters
    ----------
    path : str
        Local model directory.
    model_config : dict or None, default=None
        Load options passed as ``mlflow.pyfunc.load_model(..., model_config=...)``:
        ``extra_trusted_types`` (list of str) and ``strict`` (bool, default ``True``).

    Returns
    -------
    YohouPyfuncModel
        The model ``PyFuncModel`` wraps.

    Raises
    ------
    YohouMlflowError
        If the saved model changed the load options it was saved with. Only the
        caller may relax them.

    Notes
    -----
    MLflow merges the caller's ``model_config`` into the options saved in the model,
    and drops, with a warning, any key the model was not saved with. The saved values
    are the safe ones (no extra trusted types, strict), and they are read back from
    the ``MLmodel`` file here so that a file cannot pre-set a weaker value.
    """
    saved = Model.load(os.path.join(path, MLMODEL_FILE_NAME)).flavors.get(pyfunc.FLAVOR_NAME, {})
    saved_options = saved.get(pyfunc.MODEL_CONFIG) or {}
    tampered = sorted(
        key for key, value in saved_options.items() if key in _SAVED_LOAD_OPTIONS and value != _SAVED_LOAD_OPTIONS[key]
    )
    if tampered:
        msg = (
            f"The model file sets load option(s) {tampered} to non-default values. A saved model "
            "cannot choose its own trust or strictness; pass them to mlflow.pyfunc.load_model "
            "through model_config instead."
        )
        raise YohouMlflowError(msg)
    config = dict(model_config or {})
    forecaster, conf = _load_local(
        path, extra_trusted_types=config.get("extra_trusted_types"), strict=bool(config.get("strict", True))
    )
    return YohouPyfuncModel(forecaster, conf.get("default_prediction_type") or "point")
