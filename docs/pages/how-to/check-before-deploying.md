# How to Check That a Registered Model Will Load Before Deploying

This guide shows you how to find out, before a deployment, whether an environment can
load a registered forecaster. Use it when you upgrade yohou, scikit-learn or polars in
the environment that runs scheduled forecasts.

## Prerequisites

- A forecaster registered with `yohou_mlflow.log_model`
- The new environment installed, with access to the tracking server and registry

## Steps

### 1. Run the check in the new environment

Run this with the new environment's packages, for example in a CI job that installs
the upgraded lockfile:

```python
import yohou_mlflow

report = yohou_mlflow.check_compatibility("models:/daily-demand/latest")
print(report)
```

When the model would load, the output is:

```text
Loadable: a strict load_model would succeed.
```

When it would not, each reason is listed:

```text
Not loadable:
  - A strict load would refuse the model, version mismatch: yohou: saved with 0.1.0a12, installed 0.1.0a13 (must match exactly)
```

The check reads only the model's metadata and the type list inside the saved file. It
constructs nothing from the file, so it is safe on a model you have not reviewed.

If the forecaster needs extra trusted types, pass them as the second argument:

```python
report = yohou_mlflow.check_compatibility("models:/daily-demand/latest", ["my_project.models.DampedRidge"])
```

### 2. Fail the deployment when the model will not load

```python
if not report.loadable:
    raise SystemExit(str(report))
```

To check several versions, for example every version a rollback could use, run the
check for each URI (`models:/daily-demand/3`, `models:/daily-demand/4`, ...).

### 3. Act on a version mismatch

If the check reports a mismatch, choose one:

- Refit the forecaster in the new environment and register it, then deploy.
- Keep the old versions of the packages it names in the deployed environment.
- Load it anyway with `load_model(..., strict=False)`, which only warns. Compare its
  predictions with the previous environment's before relying on them.

## Troubleshooting

**Problem: the report lists `format version` as the only problem**
: The model was saved by a newer yohou-mlflow. Upgrade yohou-mlflow in the new
  environment.

**Problem: the report lists types outside the trust policy**
: Pass the types you trust, as in step 1, or see
  [How to save a forecaster that contains your own estimator classes](trust-third-party-estimators.md).

## See Also

- [Saved model format](../reference/saved-model-format.md): which packages are
  compared, and how strictly
- [Concepts](../explanation/concepts.md): why loading is strict about package versions
