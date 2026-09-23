"""The ``python_function`` side of the flavour: signature, input contract and model.

MLflow hands a pyfunc model its input untouched only when the signature declares no
input schema, and it drops every param unless the signature declares a params schema.
So the signature always has params and never has inputs. The input is a dict of polars
frames keyed by yohou's own argument names, which keeps every dtype as it was sent.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import polars as pl
from mlflow.models import ModelSignature
from mlflow.types.schema import ParamSchema, ParamSpec
from sklearn.utils import get_tags

INPUT_KEYS: tuple[str, ...] = ("y", "X_actual", "X_future", "X_forecast")
"""Keys a pyfunc input dict may hold, in yohou's argument order."""

PREDICTION_TYPES: tuple[str, ...] = ("point", "class_proba", "interval")
"""Values of the ``prediction_type`` param, in the order the default is chosen."""

PARAM_NAMES: tuple[str, ...] = ("prediction_type", "forecasting_horizon", "coverage_rates", "groups")
"""Params every saved model's signature declares."""

_PREDICT_METHODS: dict[str, str] = {
    "point": "predict",
    "interval": "predict_interval",
    "class_proba": "predict_class_proba",
}


def supported_prediction_types(forecaster: Any) -> frozenset[str]:
    """Return the prediction types a fitted yohou forecaster supports.

    Parameters
    ----------
    forecaster : yohou forecaster
        A yohou forecaster.

    Returns
    -------
    frozenset of str
        Subset of ``PREDICTION_TYPES``, read from the forecaster's ``forecaster_type``
        tag, or from the prediction methods it defines when the tag is unset.
    """
    tags = getattr(get_tags(forecaster), "forecaster_tags", None)
    declared = getattr(tags, "forecaster_type", None)
    if declared:
        return frozenset(declared) & frozenset(PREDICTION_TYPES)
    return frozenset(kind for kind, method in _PREDICT_METHODS.items() if callable(getattr(forecaster, method, None)))


def default_prediction_type(supported: frozenset[str]) -> str:
    """Return the prediction type used when a call names none.

    Parameters
    ----------
    supported : frozenset of str
        The forecaster's supported prediction types.

    Returns
    -------
    str
        ``"point"`` when supported, otherwise ``"class_proba"``, otherwise ``"interval"``.

    Raises
    ------
    ValueError
        If ``supported`` holds none of the three types.

    Examples
    --------
    >>> default_prediction_type(frozenset({"interval", "point"}))
    'point'
    >>> default_prediction_type(frozenset({"class_proba"}))
    'class_proba'
    """
    for kind in PREDICTION_TYPES:
        if kind in supported:
            return kind
    msg = f"The forecaster supports none of the prediction types {list(PREDICTION_TYPES)}."
    raise ValueError(msg)


def params_schema(default_type: str) -> ParamSchema:
    """Build the params schema every saved model declares.

    Parameters
    ----------
    default_type : str
        Default for ``prediction_type``.

    Returns
    -------
    mlflow.types.schema.ParamSchema
        ``prediction_type`` (string), ``forecasting_horizon`` (long, default ``None``),
        ``coverage_rates`` (list of double, default ``[]``) and ``groups`` (list of
        string, default ``[]``). MLflow rejects ``None`` as the default of a list
        param, so an empty list stands for "the forecaster's default".
    """
    return ParamSchema([
        ParamSpec("prediction_type", "string", default_type),
        ParamSpec("forecasting_horizon", "long", None),
        ParamSpec("coverage_rates", "double", [], (-1,)),
        ParamSpec("groups", "string", [], (-1,)),
    ])


def merge_signature(signature: ModelSignature | None, default_type: str) -> ModelSignature:
    """Return the signature to save: the caller's, with the flavour's params added.

    Parameters
    ----------
    signature : mlflow.models.ModelSignature or None
        Signature the caller passed, if any.
    default_type : str
        Default for ``prediction_type``.

    Returns
    -------
    mlflow.models.ModelSignature
        A signature with no input schema, the caller's output schema and params, and
        the four flavour params.

    Raises
    ------
    ValueError
        If ``signature`` has an input schema, or declares a param named like one of
        the flavour's params.
    """
    ours = params_schema(default_type)
    if signature is None:
        return ModelSignature(inputs=None, params=ours)
    if signature.inputs is not None:
        msg = (
            "The yohou flavour takes a dict of polars frames and cannot declare an input "
            "schema: MLflow would convert the frames before the model receives them. "
            "Pass a signature without inputs."
        )
        raise ValueError(msg)
    theirs = list(signature.params.params) if signature.params is not None else []
    clashes = sorted({spec.name for spec in theirs} & set(PARAM_NAMES))
    if clashes:
        msg = f"The signature declares params the yohou flavour defines itself: {clashes}. Rename or remove them."
        raise ValueError(msg)
    return ModelSignature(inputs=None, outputs=signature.outputs, params=ParamSchema(theirs + list(ours.params)))


def validate_input(model_input: Any) -> dict[str, pl.DataFrame]:
    """Check a pyfunc input and return it as a dict of polars frames.

    Parameters
    ----------
    model_input : Any
        What the caller passed to ``predict``.

    Returns
    -------
    dict of str to polars.DataFrame
        The same frames, keyed by a subset of ``INPUT_KEYS``.

    Raises
    ------
    TypeError
        If the input is not a dict, or a value is not a polars DataFrame.
    ValueError
        If a key is not one of ``INPUT_KEYS``, or ``X_actual`` is given without ``y``.

    Examples
    --------
    >>> import polars as pl
    >>> sorted(validate_input({"y": pl.DataFrame({"v": [1.0]})}))
    ['y']
    >>> validate_input({"Y": pl.DataFrame()})
    Traceback (most recent call last):
    ...
    ValueError: Unknown pyfunc input key(s) ['Y']. Valid keys: ['y', 'X_actual', 'X_future', 'X_forecast'].
    """
    if not isinstance(model_input, Mapping):
        msg = (
            "The yohou pyfunc input must be a dict of polars DataFrames keyed by "
            f"{list(INPUT_KEYS)}, got {type(model_input).__module__}.{type(model_input).__qualname__}. "
            "pandas and REST input is not supported yet."
        )
        raise TypeError(msg)
    unknown = sorted(str(key) for key in model_input if key not in INPUT_KEYS)
    if unknown:
        msg = f"Unknown pyfunc input key(s) {unknown}. Valid keys: {list(INPUT_KEYS)}."
        raise ValueError(msg)
    for key, value in model_input.items():
        if not isinstance(value, pl.DataFrame):
            msg = (
                f"The pyfunc input {key!r} must be a polars DataFrame, got "
                f"{type(value).__module__}.{type(value).__qualname__}. Valid keys: {list(INPUT_KEYS)}."
            )
            raise TypeError(msg)
    if "X_actual" in model_input and "y" not in model_input:
        msg = "The pyfunc input 'X_actual' is only accepted together with 'y': actual features are observed with the target."
        raise ValueError(msg)
    return dict(model_input)


class YohouPyfuncModel:
    """The object ``mlflow.pyfunc.load_model`` wraps for a yohou model.

    Each call works on a copy of the loaded forecaster: it observes ``y`` (and the
    other frames) when given, then predicts once from the updated state. Nothing a
    call changes is visible to the next call.

    Parameters
    ----------
    forecaster : yohou forecaster
        The loaded, fitted forecaster.
    default_type : str
        ``prediction_type`` used when a call does not name one.
    """

    def __init__(self, forecaster: Any, default_type: str) -> None:
        self._forecaster = forecaster
        self._default_type = default_type
        self._supported = supported_prediction_types(forecaster)

    def get_raw_model(self) -> Any:
        """Return the loaded forecaster itself, as ``PyFuncModel.get_raw_model`` expects.

        Returns
        -------
        yohou forecaster
            The forecaster this model wraps. Changing it changes later predictions.
        """
        return self._forecaster

    def predict(self, model_input: Any, params: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """Observe the given frames on a copy of the forecaster, then predict once.

        Parameters
        ----------
        model_input : dict of str to polars.DataFrame
            A subset of ``y``, ``X_actual``, ``X_future`` and ``X_forecast``. With
            ``y``, the copy observes the frames before predicting. Without ``y``, it
            predicts from the saved state.
        params : mapping or None, default=None
            ``prediction_type``, ``forecasting_horizon``, ``coverage_rates`` and
            ``groups``. An empty ``coverage_rates`` or ``groups`` means the
            forecaster's default.

        Returns
        -------
        polars.DataFrame
            The frame the selected yohou predict method returns.

        Raises
        ------
        ValueError
            If ``prediction_type`` is unknown or unsupported by the forecaster, or
            ``coverage_rates`` is given for a prediction type other than ``interval``.
        """
        frames = validate_input(model_input)
        params = dict(params or {})
        kind = params.get("prediction_type") or self._default_type
        if kind not in self._supported:
            msg = f"prediction_type={kind!r} is not supported by this forecaster. Supported: {sorted(self._supported)}."
            raise ValueError(msg)
        horizon = params.get("forecasting_horizon")
        # yohou applies its defaults only for None, and MLflow sends [] for an omitted
        # list param: [] would raise on a non-panel forecaster for `groups`, select no
        # series on a panel one, and produce no intervals for `coverage_rates`.
        coverage_rates = list(params.get("coverage_rates") or []) or None
        groups = list(params.get("groups") or []) or None
        if coverage_rates is not None and kind != "interval":
            msg = f"coverage_rates only applies to prediction_type='interval', not {kind!r}."
            raise ValueError(msg)

        forecaster = copy.deepcopy(self._forecaster)
        if "y" in frames:
            forecaster.observe(
                frames["y"],
                X_actual=frames.get("X_actual"),
                groups=groups,
                X_future=frames.get("X_future"),
                X_forecast=frames.get("X_forecast"),
            )
        kwargs: dict[str, Any] = {
            "X_future": frames.get("X_future"),
            "X_forecast": frames.get("X_forecast"),
            "forecasting_horizon": None if horizon is None else int(horizon),
            "groups": groups,
        }
        if kind == "interval":
            kwargs["coverage_rates"] = coverage_rates
        return getattr(forecaster, _PREDICT_METHODS[kind])(**kwargs)
