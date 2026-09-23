# How to Move Pickled Forecasters to the Yohou Flavour

This guide shows you how to re-save forecasters you stored with pickle, for example
through MLflow's generic Python model, as yohou-flavour models.

## Prerequisites

- The environment that saved the pickles, with the same yohou, scikit-learn and polars
  versions. A pickle may not load, or may load and misbehave, anywhere else.
- yohou-mlflow installed in that environment
- Pickles you trust: loading a pickle runs code from the file

## Steps

### 1. Load the pickled forecaster in its original environment

If you logged it with MLflow's scikit-learn flavour:

```python
import mlflow.sklearn

legacy = mlflow.sklearn.load_model("models:/daily-demand-pickle/7")
```

If you logged it inside your own `mlflow.pyfunc.PythonModel`, unwrap it and read the
attribute that holds the forecaster:

```python
import mlflow.pyfunc

wrapper = mlflow.pyfunc.load_model("models:/daily-demand-pickle/7").unwrap_python_model()
legacy = wrapper.forecaster  # the attribute your PythonModel stored it in
```

If it is a plain pickle file:

```python
import pickle

with open("forecaster.pkl", "rb") as file:
    legacy = pickle.load(file)
```

### 2. Log it with the yohou flavour

```python
import mlflow
import yohou_mlflow

with mlflow.start_run():
    info = yohou_mlflow.log_model(legacy, name="forecaster", registered_model_name="daily-demand")
```

The model is loaded back and compared with `legacy` before it is kept. If this raises
`UntrustedTypesError`, see
[How to save a forecaster that contains your own estimator classes](trust-third-party-estimators.md).

### 3. Confirm the new model

```python
print(yohou_mlflow.check_compatibility(info.model_uri).loadable)
print(yohou_mlflow.load_model(info.model_uri).predict().equals(legacy.predict()))
```

```text
True
True
```

### 4. Switch your jobs to the new registered model

Replace the loading code in your scheduled jobs with
`yohou_mlflow.load_model("models:/daily-demand/latest")`, and the saving code with
`yohou_mlflow.log_model(...)`.

## Troubleshooting

**Problem: `SaveVerificationError` mentioning time-zone-aware data**
: The installed skops cannot rebuild the time zones the forecaster holds. See the
  time-zone limitation in [Saved model format](../reference/saved-model-format.md).

## See Also

- [How to check that a registered model will load](check-before-deploying.md)
