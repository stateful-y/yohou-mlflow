# How to Save a Forecaster That Contains Your Own Estimator Classes

This guide shows you how to save and load a forecaster whose estimators come from your
own code or from a third-party library. Use it when `save_model` or `log_model` raises
`UntrustedTypesError`.

## Prerequisites

- A fitted yohou forecaster that uses an estimator class defined outside yohou and
  scikit-learn, for example `my_project.models.DampedRidge`
- The package defining that class is importable wherever the model is loaded

## Steps

### 1. Find the type names to trust

Save the forecaster once without extra types and read the names from the error:

```python
import yohou_mlflow

try:
    yohou_mlflow.save_model(forecaster, "model")
except yohou_mlflow.UntrustedTypesError as error:
    trusted = list(error.types)

print(trusted)
```

```text
['my_project.models.DampedRidge']
```

Nothing is written when this error is raised. Trust only types from code you control
or have reviewed: loading a trusted type runs that type's code.

### 2. Save with those types

Pass the names to `save_model`, or to `log_model` when you save into a run:

```python
yohou_mlflow.save_model(forecaster, "model", extra_trusted_types=trusted)
```

```python
with mlflow.start_run():
    yohou_mlflow.log_model(
        forecaster,
        name="forecaster",
        registered_model_name="daily-demand",
        extra_trusted_types=trusted,
    )
```

### 3. Pass the same types every time you load

The saved model does not remember them. Keep the list in your code or configuration
and pass it on every load.

If you load the forecaster itself:

```python
forecaster = yohou_mlflow.load_model("models:/daily-demand/latest", extra_trusted_types=trusted)
```

If you load it through MLflow's generic interface, pass the list in `model_config`:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/daily-demand/latest",
    model_config={"extra_trusted_types": trusted},
)
```

## Troubleshooting

**Problem: `UntrustedTypesError` when loading a model that saved fine**
: The load did not receive the extra types. Pass the same `extra_trusted_types` you
  used when saving.

**Problem: the load raises `ModuleNotFoundError` for your package**
: The class is trusted but not installed. Install the package that defines it in the
  loading environment, or bundle it with `code_paths` when saving.

**Problem: a model file lists your type in its `MLmodel` file, yet loading still refuses it**
: This is intended. The type list stored in the model is informational, and a saved
  model can never extend what is trusted.

## See Also

- [Saved model format](../reference/saved-model-format.md): the built-in trusted types
- [Concepts](../explanation/concepts.md): why the model file cannot say what to trust
