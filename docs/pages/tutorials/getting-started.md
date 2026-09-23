# Getting Started

In this tutorial, we will run the loop a scheduled forecasting job runs. We will fit
a forecaster, register it in an MLflow model registry, load it back as a later run
would, feed it a week of new observations, forecast the next week, and register the
updated forecaster so the following run can pick it up.

<!-- COMPANION_NOTEBOOKS -->

## Prerequisites

- Python 3.11 or later
- A terminal, opened in an empty directory: MLflow will create `mlflow.db` and an
  `mlruns/` folder there

## Installation

We install Yohou-MLflow together with the full `mlflow` package, which provides the
local SQLite model registry we use in this tutorial:

=== "pip"

    ```bash
    pip install yohou_mlflow mlflow
    ```

=== "uv"

    ```bash
    uv add yohou_mlflow mlflow
    ```

Start Python and check the installation:

```python
import yohou_mlflow

print(yohou_mlflow.__version__)
```

The output is the installed version, something like:

```text
0.1.0
```

## Create some data

First, let's create six months of daily demand with a weekly pattern. We keep the
last week aside: it plays the part of the observations that arrive after the model
was registered.

```python
import datetime as dt

import numpy as np
import polars as pl

days = pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 30), "1d", eager=True)
steps = np.arange(len(days))
demand = pl.DataFrame({"time": days, "demand": 100 + 10 * np.sin(2 * np.pi * steps / 7) + steps * 0.1})
history, next_week = demand[:-7], demand[-7:]
print(history.tail(3))
```

You should see the last three days of the history, ending on 23 June:

```text
shape: (3, 2)
┌─────────────────────┬────────────┐
│ time                ┆ demand     │
│ ---                 ┆ ---        │
│ datetime[μs]        ┆ f64        │
╞═════════════════════╪════════════╡
│ 2024-06-21 00:00:00 ┆ 112.861163 │
│ 2024-06-22 00:00:00 ┆ 107.550721 │
│ 2024-06-23 00:00:00 ┆ 109.581685 │
└─────────────────────┴────────────┘
```

## Fit a forecaster

Now we fit a forecaster that predicts the next seven days:

```python
from sklearn.linear_model import Ridge
from yohou.point import PointReductionForecaster

forecaster = PointReductionForecaster(estimator=Ridge())
forecaster.fit(history, forecasting_horizon=7)
print(forecaster.predict())
```

The output is a forecast for 24 to 30 June:

```text
shape: (7, 3)
┌─────────────────────┬─────────────────────┬────────────┐
│ vintage_time        ┆ time                ┆ demand     │
│ ---                 ┆ ---                 ┆ ---        │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64        │
╞═════════════════════╪═════════════════════╪════════════╡
│ 2024-06-23 00:00:00 ┆ 2024-06-24 00:00:00 ┆ 109.357054 │
│ 2024-06-23 00:00:00 ┆ 2024-06-25 00:00:00 ┆ 108.741323 │
│ 2024-06-23 00:00:00 ┆ 2024-06-26 00:00:00 ┆ 108.27335  │
│ 2024-06-23 00:00:00 ┆ 2024-06-27 00:00:00 ┆ 108.380829 │
│ 2024-06-23 00:00:00 ┆ 2024-06-28 00:00:00 ┆ 109.05813  │
│ 2024-06-23 00:00:00 ┆ 2024-06-29 00:00:00 ┆ 109.870532 │
│ 2024-06-23 00:00:00 ┆ 2024-06-30 00:00:00 ┆ 110.281583 │
└─────────────────────┴─────────────────────┴────────────┘
```

Notice the `vintage_time` column: every row was forecast from 23 June, the last day
the forecaster has seen.

## Register it

Now we point MLflow at a local database and log the forecaster as version 1 of a
registered model called `daily-demand`:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    info = yohou_mlflow.log_model(forecaster, name="forecaster", registered_model_name="daily-demand")
print(info.registered_model_version)
```

After a moment, MLflow reports the registration, and the last line is the version:

```text
Successfully registered model 'daily-demand'.
Created version '1' of model 'daily-demand'.
1
```

Before the model was kept, Yohou-MLflow loaded it back once and checked that it
predicts exactly what `forecaster` predicts. A model that saves but cannot be
loaded is refused here, not months later.

## Load it in a later run

A scheduled job starts from the registry, not from the `forecaster` variable. Let's
do the same:

```python
loaded = yohou_mlflow.load_model("models:/daily-demand/1")
print(loaded.predict().equals(forecaster.predict()))
```

```text
True
```

The loaded forecaster is the one we registered, and it predicts the same frame.

## Observe the new week and forecast

The week we kept aside has now "happened". We give it to the loaded forecaster and
forecast again:

```python
loaded.observe(next_week)
forecast = loaded.predict()
print(forecast)
```

```text
shape: (7, 3)
┌─────────────────────┬─────────────────────┬────────────┐
│ vintage_time        ┆ time                ┆ demand     │
│ ---                 ┆ ---                 ┆ ---        │
│ datetime[μs]        ┆ datetime[μs]        ┆ f64        │
╞═════════════════════╪═════════════════════╪════════════╡
│ 2024-06-30 00:00:00 ┆ 2024-07-01 00:00:00 ┆ 109.872558 │
│ 2024-06-30 00:00:00 ┆ 2024-07-02 00:00:00 ┆ 108.850057 │
│ 2024-06-30 00:00:00 ┆ 2024-07-03 00:00:00 ┆ 108.059289 │
│ 2024-06-30 00:00:00 ┆ 2024-07-04 00:00:00 ┆ 108.17102  │
│ 2024-06-30 00:00:00 ┆ 2024-07-05 00:00:00 ┆ 109.176415 │
│ 2024-06-30 00:00:00 ┆ 2024-07-06 00:00:00 ┆ 110.393695 │
│ 2024-06-30 00:00:00 ┆ 2024-07-07 00:00:00 ┆ 110.981524 │
└─────────────────────┴─────────────────────┴────────────┘
```

Notice that the vintage moved to 30 June and the forecast now covers the first week
of July. `observe` updated what the forecaster has seen without refitting it.

## Register the updated forecaster

The next run needs to start where this one stopped, so we register the updated
forecaster as version 2:

```python
with mlflow.start_run():
    info = yohou_mlflow.log_model(loaded, name="forecaster", registered_model_name="daily-demand")
print(info.registered_model_version)
```

```text
Registered model 'daily-demand' already exists. Creating a new version of this model...
Created version '2' of model 'daily-demand'.
2
```

Let's check that version 2 remembers the week it observed:

```python
latest = yohou_mlflow.load_model("models:/daily-demand/2")
print(latest.observed_time_)
print(latest.predict().equals(forecast))
```

```text
2024-06-30 00:00:00
True
```

Version 2 has seen everything up to 30 June and continues exactly where the previous
run stopped. You can run the last three sections again with new data: each run loads
the latest version, observes, forecasts, and registers the next version.

## What we built

You have run a scheduled forecasting loop against a model registry. Along the way,
you:

- logged and registered a fitted forecaster with `yohou_mlflow.log_model`
- loaded a registered version with `yohou_mlflow.load_model`
- updated it with new observations and registered the result as a new version

## Next steps

- [Check a model before deploying](../how-to/check-before-deploying.md) a new
  environment that will load it
- [Predict through MLflow's generic interface](../how-to/predict-with-pyfunc.md),
  observing new rows and forecasting in one call
- [Keep the registry small](../how-to/keep-the-registry-small.md) when every run
  registers a version
- [Concepts](../explanation/concepts.md): why loading is strict about package
  versions, and what a saved model contains
- [API Reference](../reference/api.md)
