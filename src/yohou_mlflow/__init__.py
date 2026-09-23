"""Yohou-MLflow: an MLflow flavour for saving, loading and serving yohou forecasters."""

from importlib.metadata import version

from yohou_mlflow._exceptions import (
    FormatVersionError,
    SaveVerificationError,
    UntrustedTypesError,
    VersionMismatchError,
    VersionMismatchWarning,
    YohouMlflowError,
)
from yohou_mlflow._persistence import (
    FLAVOR_NAME,
    FORMAT_VERSION,
    CompatibilityReport,
    check_compatibility,
    get_default_pip_requirements,
    load_model,
    log_model,
    save_model,
)
from yohou_mlflow._persistence import (
    _load_pyfunc as _load_pyfunc,
)
from yohou_mlflow._pyfunc import INPUT_KEYS, PARAM_NAMES, PREDICTION_TYPES, YohouPyfuncModel
from yohou_mlflow._trust import TRUSTED_TYPE_PREFIXES, TRUSTED_TYPES
from yohou_mlflow._versions import VERSION_RULES, VersionMismatch

__version__ = version(__name__)

# `_load_pyfunc` is imported above because `mlflow.pyfunc.load_model` looks it up on
# this module (the model's `loader_module`); it is not public API.
__all__ = [
    "FLAVOR_NAME",
    "FORMAT_VERSION",
    "INPUT_KEYS",
    "PARAM_NAMES",
    "PREDICTION_TYPES",
    "TRUSTED_TYPES",
    "TRUSTED_TYPE_PREFIXES",
    "VERSION_RULES",
    "CompatibilityReport",
    "FormatVersionError",
    "SaveVerificationError",
    "UntrustedTypesError",
    "VersionMismatch",
    "VersionMismatchError",
    "VersionMismatchWarning",
    "YohouMlflowError",
    "YohouPyfuncModel",
    "__version__",
    "check_compatibility",
    "get_default_pip_requirements",
    "load_model",
    "log_model",
    "save_model",
]
