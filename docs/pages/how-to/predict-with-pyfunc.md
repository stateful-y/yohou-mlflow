# How to Forecast Through MLflow's Generic Predict Interface

This guide shows you how to forecast with a yohou model loaded through
`mlflow.pyfunc.load_model`. Use it when your code loads MLflow models generically
rather than through `yohou_mlflow.load_model`.

## Prerequisites

- A forecaster saved or logged with yohou-mlflow
- New observations as polars DataFrames, if you want to forecast from them

## Steps

### 1. Load the model

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/daily-demand/latest")
```

If the forecaster contains your own estimator classes, pass them at load time:
`mlflow.pyfunc.load_model(uri, model_config={"extra_trusted_types": [...]})`.

### 2. Forecast

To forecast from the saved state, pass an empty dict:

```python
forecast = model.predict({})
```

To forecast from new observations, pass them under `y`. The model observes them and
forecasts once, from the last observed time:

```python
forecast = model.predict({"y": new_rows})
```

Pass exogenous features under `X_actual`, `X_future` and `X_forecast`, the names yohou
uses. `X_actual` is only accepted together with `y`:

```python
forecast = model.predict({"y": new_rows, "X_actual": new_features, "X_future": known_future})
```

### 3. Choose what to predict

Pass `params` to select the prediction and its options:

- Prediction intervals:

    ```python
    forecast = model.predict({"y": new_rows}, params={"prediction_type": "interval", "coverage_rates": [0.8, 0.95]})
    ```

- Class probabilities, for a class-probability forecaster:

    ```python
    forecast = model.predict({"y": new_rows}, params={"prediction_type": "class_proba"})
    ```

- A different number of steps, or some panel groups only:

    ```python
    forecast = model.predict({}, params={"forecasting_horizon": 3, "groups": ["store_1"]})
    ```

The result is a polars DataFrame, the same frame the yohou method returns.

### 4. Keep the update, if you need it

Each `predict` call works on a copy, so the observations are not kept. To keep them,
load the forecaster itself, observe, and register it again:

```python
import yohou_mlflow

forecaster = yohou_mlflow.load_model("models:/daily-demand/latest")
forecaster.observe(new_rows)
with mlflow.start_run():
    yohou_mlflow.log_model(forecaster, name="forecaster", registered_model_name="daily-demand")
```

## Troubleshooting

**Problem: `TypeError: The yohou pyfunc input must be a dict of polars DataFrames`**
: Pass a dict of polars frames. pandas input, and therefore `mlflow models serve` and
  Spark UDFs, is not supported yet.

**Problem: `prediction_type='interval' is not supported by this forecaster`**
: The forecaster cannot produce that prediction. The error lists the types it
  supports.

## See Also

- [pyfunc contract](../reference/pyfunc-contract.md): every input key, param and error
- [Concepts](../explanation/concepts.md): why each call works on a copy
