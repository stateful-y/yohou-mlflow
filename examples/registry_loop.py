# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlflow",
#     "numpy",
#     "polars",
#     "scikit-learn",
#     "yohou",
#     "yohou-mlflow",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "Scheduled Forecasting with a Model Registry",
    "description": "Register a forecaster, load it in a later run, observe new data, forecast, and register the update.",
    "category": "tutorial",
    "companion": "pages/tutorials/getting-started.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Scheduled Forecasting with a Model Registry

        In this notebook, we will run the loop a scheduled forecasting job runs.
        We will fit a forecaster, register it in an MLflow model registry, load it
        back as a later run would, feed it a week of new observations, forecast the
        next week, and register the updated forecaster for the following run.

        **Prerequisites**: basic familiarity with yohou's `fit` and `predict`.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import datetime as dt
    import tempfile
    from pathlib import Path

    import mlflow
    import numpy as np
    import polars as pl
    from sklearn.linear_model import Ridge
    from yohou.point import PointReductionForecaster

    import yohou_mlflow

    return Path, PointReductionForecaster, Ridge, dt, mlflow, np, pl, tempfile, yohou_mlflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Create some data

        We create six months of daily demand with a weekly pattern, and keep the
        last week aside: it plays the part of the observations that arrive after
        the model was registered.
        """
    )
    return


@app.cell
def _(dt, np, pl):
    days = pl.datetime_range(dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 30), "1d", eager=True)
    steps = np.arange(len(days))
    demand = pl.DataFrame({"time": days, "demand": 100 + 10 * np.sin(2 * np.pi * steps / 7) + steps * 0.1})
    history, next_week = demand[:-7], demand[-7:]
    history.tail(3)
    return history, next_week


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Fit a forecaster

        We fit a forecaster that predicts the next seven days.
        """
    )
    return


@app.cell
def _(PointReductionForecaster, Ridge, history):
    forecaster = PointReductionForecaster(estimator=Ridge())
    forecaster.fit(history, forecasting_horizon=7)
    forecaster.predict()
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice the `vintage_time` column: every row was forecast from 23 June, the
        last day the forecaster has seen.

        ## 3. Register it

        We keep the registry in a temporary directory, so this notebook leaves
        nothing behind, and log the forecaster as version 1 of `daily-demand`.
        """
    )
    return


@app.cell
def _(Path, forecaster, mlflow, tempfile, yohou_mlflow):
    workdir = Path(tempfile.mkdtemp())
    mlflow.set_tracking_uri(f"sqlite:///{workdir / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment("registry-loop", artifact_location=(workdir / "artifacts").as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        first = yohou_mlflow.log_model(forecaster, name="forecaster", registered_model_name="daily-demand")
    first.registered_model_version
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before the model was kept, Yohou-MLflow loaded it back once and checked that
        it predicts exactly what `forecaster` predicts.

        ## 4. Load it in a later run

        A scheduled job starts from the registry, so we load version 1.
        """
    )
    return


@app.cell
def _(forecaster, yohou_mlflow):
    loaded = yohou_mlflow.load_model("models:/daily-demand/1")
    loaded.predict().equals(forecaster.predict())
    return (loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Observe the new week and forecast

        The week we kept aside has now happened. We give it to the loaded
        forecaster and forecast again.
        """
    )
    return


@app.cell
def _(loaded, next_week):
    loaded.observe(next_week)
    forecast = loaded.predict()
    forecast
    return (forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that the vintage moved to 30 June and the forecast now covers the
        first week of July: `observe` updated what the forecaster has seen without
        refitting it.

        ## 6. Register the updated forecaster

        The next run must start where this one stopped, so we register the updated
        forecaster as version 2 and check that it remembers the week it observed.
        """
    )
    return


@app.cell
def _(forecast, loaded, mlflow, yohou_mlflow):
    with mlflow.start_run():
        yohou_mlflow.log_model(loaded, name="forecaster", registered_model_name="daily-demand")
    latest = yohou_mlflow.load_model("models:/daily-demand/2")
    {"observed up to": latest.observed_time_, "same forecast": latest.predict().equals(forecast)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What We Built

        We ran a scheduled forecasting loop against a model registry: we logged and
        registered a fitted forecaster with `yohou_mlflow.log_model`, loaded it with
        `yohou_mlflow.load_model`, updated it with a week of new observations, and
        registered the result as version 2, which continues exactly where the
        previous run stopped.

        ## Next Steps

        - [How to check a model before deploying](/pages/how-to/check-before-deploying/)
        - [How to predict through MLflow's generic interface](/pages/how-to/predict-with-pyfunc/)
        - [How to keep the registry small](/pages/how-to/keep-the-registry-small/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
