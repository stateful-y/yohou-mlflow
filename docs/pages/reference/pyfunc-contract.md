# pyfunc Contract

Description of the `python_function` flavour of a yohou model: how it loads, what
`predict` accepts, and what it returns.

## Loading

```python
model = mlflow.pyfunc.load_model(model_uri, model_config=None)
```

| `model_config` key | Type | Default | Meaning |
|---|---|---|---|
| `extra_trusted_types` | list of string | `[]` | Type names to trust in addition to the trust policy |
| `strict` | bool | `true` | Raise on a package version mismatch; `false` warns and loads |

MLflow drops any other key with a warning. The loading checks are those of
`load_model`; see [Saved model format](saved-model-format.md#load-order).

Before those checks run, MLflow imports the `loader_module` and any `code` directory
named in the model's `flavors.python_function`. Only models you trust should be loaded
this way; check others with `check_compatibility` first.

## `predict(model_input, params=None)`

### Input

`model_input` is a dict. Keys are a subset of `yohou_mlflow.INPUT_KEYS`; values are
polars DataFrames, passed to yohou unchanged.

| Key | yohou argument | Required |
|---|---|---|
| `y` | `y` | No |
| `X_actual` | `X_actual` | Only with `y` |
| `X_future` | `X_future` | No |
| `X_forecast` | `X_forecast` | No |

An empty dict predicts from the saved state.

### Params

The signature declares these params (`yohou_mlflow.PARAM_NAMES`):

| Param | Type | Default | Meaning |
|---|---|---|---|
| `prediction_type` | string | `default_prediction_type` from `MLmodel` | One of `point`, `interval`, `class_proba` |
| `forecasting_horizon` | integer | `None` | Steps to forecast; `None` uses the fitted horizon |
| `coverage_rates` | list of float | `[]` | Interval coverage rates; `[]` uses the forecaster's default. `interval` only |
| `groups` | list of string | `[]` | Panel groups to predict; `[]` predicts all groups |

The default `prediction_type` is `point` when the forecaster supports it, otherwise
`class_proba`, otherwise `interval`.

### Behaviour

Each call works on a deep copy of the loaded forecaster. Changes made during a call are
not visible to later calls.

1. When `y` is present, call `observe(y, X_actual=..., groups=..., X_future=..., X_forecast=...)`.
2. Call the method for `prediction_type` with `X_future`, `X_forecast`,
   `forecasting_horizon` and `groups`:

| `prediction_type` | Method | Extra argument |
|---|---|---|
| `point` | `predict` | |
| `interval` | `predict_interval` | `coverage_rates` |
| `class_proba` | `predict_class_proba` | |

Empty `coverage_rates` and `groups` are passed as `None`.

### Output

The polars DataFrame returned by the method, unchanged. With `y`, it holds one vintage,
at the last observed time.

## Errors

| Condition | Exception |
|---|---|
| `model_input` is not a dict (for example a pandas DataFrame) | `TypeError` |
| A key is not in `INPUT_KEYS` | `ValueError` |
| A value is not a polars DataFrame | `TypeError` |
| `X_actual` without `y` | `ValueError` |
| `prediction_type` not supported by the forecaster | `ValueError` |
| Non-empty `coverage_rates` with a `prediction_type` other than `interval` | `ValueError` |

## Signature rules at save time

| Signature passed to `save_model` | Result |
|---|---|
| None | Params-only signature with the four params |
| Without params or inputs | The four params are added; outputs are kept |
| With other params | Merged with the four params |
| With a param named like one of the four | `ValueError`; nothing is written |
| With an input schema | `ValueError`; nothing is written |

## Not supported

pandas input, and therefore `mlflow models serve` and Spark UDFs.
