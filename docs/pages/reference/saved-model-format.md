# Saved Model Format

Description of a model directory written by `save_model` or `log_model`, and of the
checks `load_model` and `check_compatibility` apply to it. Format version `1.0`
(`yohou_mlflow.FORMAT_VERSION`).

## Directory layout

| File | Content |
|---|---|
| `MLmodel` | MLflow model configuration, with the `yohou` and `python_function` flavours |
| `forecaster.skops` | The fitted forecaster, written with `skops.io.dump` and `ZIP_DEFLATED` compression |
| `requirements.txt` | pip requirements |
| `conda.yaml` | Conda environment |
| `python_env.yaml` | Python version and build dependencies |
| `constraints.txt` | pip constraints, present only when the requirements carry any |

## `MLmodel`: `flavors.yohou`

| Key | Type | Value |
|---|---|---|
| `format_version` | string | `"1.0"` |
| `components` | mapping | Component name to file name. Format 1.0: `{forecaster: forecaster.skops}` |
| `versions` | mapping | Installed version at save time of `yohou`, `scikit-learn`, `polars`, `skops`, `yohou-mlflow` |
| `recorded_types` | list of string | Types in `forecaster.skops` that skops does not trust by default. Informational: never used to grant trust |
| `forecaster_class` | string | Fully qualified class of the saved forecaster |
| `forecaster_type` | list of string | Supported prediction types, from the forecaster's `forecaster_type` tag |
| `default_prediction_type` | string | `prediction_type` used by the pyfunc flavour when a call names none |

Example:

```yaml
flavors:
  yohou:
    components:
      forecaster: forecaster.skops
    default_prediction_type: point
    forecaster_class: yohou.interval.split_conformal.SplitConformalForecaster
    forecaster_type:
    - interval
    - point
    format_version: '1.0'
    recorded_types:
    - datetime.datetime
    - polars.dataframe.frame.DataFrame
    - polars.datatypes.classes.Float64
    - yohou.interval.split_conformal.SplitConformalForecaster
    - yohou.metrics.conformity.Residual
    - yohou.point.reduction.PointReductionForecaster
    versions:
      polars: 1.44.2
      scikit-learn: 1.9.1
      skops: 0.15.0
      yohou: 0.1.0a13
      yohou-mlflow: 0.1.0
```

## `MLmodel`: `flavors.python_function`

| Key | Value |
|---|---|
| `loader_module` | `yohou_mlflow` |
| `config` | `{extra_trusted_types: [], strict: true}`. Loading refuses a model whose saved values differ |
| `env` | `conda.yaml` and `python_env.yaml` |
| `python_version` | Python version at save time, recorded by MLflow |

The model signature declares params and no inputs. See [pyfunc contract](pyfunc-contract.md).

## Format version

| Recorded major version | `load_model` | `check_compatibility` |
|---|---|---|
| `1` | Loads | Reports no format problem |
| Any other, missing or unparseable | Raises `FormatVersionError` | Reports the format as the only problem |

A component file name that is empty, contains a path separator, or is `.` or `..`,
raises `YohouMlflowError`.

## Trust policy

A type in `forecaster.skops` is trusted when any of the following holds:

| Rule | Types |
|---|---|
| skops default | Types skops trusts on its own, including most scikit-learn, numpy and scipy types |
| Prefix | `yohou.`, `sklearn.`, `polars.datatypes.` |
| Exact name | `polars.dataframe.frame.DataFrame`, `polars.series.series.Series`, `datetime.date`, `datetime.datetime`, `datetime.timedelta`, `zoneinfo.ZoneInfo` |
| Caller | Names passed as `extra_trusted_types`, to `save_model`, `log_model`, `load_model`, `check_compatibility`, or to `mlflow.pyfunc.load_model` through `model_config` |

The prefixes and exact names are exported as `yohou_mlflow.TRUSTED_TYPE_PREFIXES` and
`yohou_mlflow.TRUSTED_TYPES`.

polars frames are stored in polars' binary format and rebuilt by
`polars.DataFrame.deserialize`, which skops does not inspect.

## Version rules

| Package | Rule | Mismatch when |
|---|---|---|
| `yohou` | exact | The installed version differs from the recorded one |
| `scikit-learn` | major.minor | The major or minor version differs |
| `polars` | major.minor | The major or minor version differs |
| `skops` | recorded only | Never |
| `yohou-mlflow` | recorded only | Never |

The rules are exported as `yohou_mlflow.VERSION_RULES`. With `strict=True` (default),
a mismatch raises `VersionMismatchError`. With `strict=False`, it emits
`VersionMismatchWarning` with the same text and the load proceeds.

## Load order

`load_model` applies these steps in order. Each failing step stops the load before any
object is constructed from `forecaster.skops`:

1. Read `flavors.yohou`; check `format_version`.
2. Compare `versions` under the version rules.
3. Read the types in `forecaster.skops` with `skops.io.get_untrusted_types` and check
   them against the trust policy; raise `UntrustedTypesError` for any outside it.
4. Load with `skops.io.load`, trusting exactly those types.

`check_compatibility` applies steps 1 to 3 and returns a `CompatibilityReport` instead of
raising.

Neither function puts any directory from the model on Python's import path, and a `code`
key in the `MLmodel` file is ignored.

`mlflow.pyfunc.load_model` is outside this guarantee: MLflow imports the `loader_module` and
any `code` directory named in `flavors.python_function` before calling this package.

## Save checks

`save_model` applies the first check before anything is written, and the others after
writing `forecaster.skops`, removing the directory if one fails:

| Check | Failure |
|---|---|
| Forecaster is a fitted yohou forecaster | `TypeError`, `NotFittedError` (raised before anything is written) |
| Types are within the trust policy | `UntrustedTypesError` |
| The file loads back | `SaveVerificationError` |
| The loaded copy's default prediction equals the original's | `SaveVerificationError` |

When the original's default predict method raises without inputs, the comparison is
skipped and a `UserWarning` is emitted.

## Default pip requirements

`yohou_mlflow`, `yohou`, `scikit-learn`, `polars` and `skops`, each pinned to the
installed version (`get_default_pip_requirements`). `pip_requirements` replaces them;
`extra_pip_requirements` adds to them.

## Known limitation: time zones

skops 0.15.0 and earlier cannot rebuild `zoneinfo.ZoneInfo`. A forecaster fitted on
time-zone-aware data (any time zone, including UTC) holds one in `observed_time_`, so
`save_model` raises `SaveVerificationError` naming time-zone-aware data. Such
forecasters save and load once a skops release supports `ZoneInfo`.

## Errors

| Exception | Raised by | When |
|---|---|---|
| `UntrustedTypesError` | `save_model`, `load_model`, pyfunc load | Types outside the trust policy; `.types` lists them |
| `VersionMismatchError` | `load_model`, pyfunc load | A version rule is broken and `strict=True`; `.mismatches` lists them |
| `SaveVerificationError` | `save_model` | The written model does not load back, or predicts differently |
| `FormatVersionError` | `load_model`, pyfunc load | Unsupported `format_version` |
| `YohouMlflowError` | all | Base class of the above; also raised for an invalid component file name and a pyfunc `config` changed in the file |
