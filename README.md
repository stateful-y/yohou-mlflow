<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-mlflow/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-mlflow/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/yohou-mlflow/main/docs/assets/logo_light.png" alt="Yohou-MLflow">
  </picture>
</p>


[![Python Version](https://img.shields.io/pypi/pyversions/yohou_mlflow)](https://pypi.org/project/yohou_mlflow/)
[![License](https://img.shields.io/github/license/stateful-y/yohou-mlflow)](https://github.com/stateful-y/yohou-mlflow/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/yohou_mlflow)](https://pypi.org/project/yohou_mlflow/)
[![codecov](https://codecov.io/gh/stateful-y/yohou-mlflow/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/yohou-mlflow)

## What is Yohou-MLflow?

Yohou-MLflow is an MLflow flavour for [yohou](https://github.com/stateful-y/yohou)
forecasters. It saves a fitted forecaster, including everything it has observed since
fitting, as an MLflow model you can log, register and load back, so a scheduled job can
load the latest version, observe new data, forecast, and register the update for the
next run.

Forecasters are stored with [skops](https://skops.readthedocs.io/) instead of pickle, so
`yohou_mlflow.load_model` never runs code chosen by the file. MLflow's generic
`mlflow.pyfunc.load_model` imports the modules a model file names, so load a model you have
not reviewed with `yohou_mlflow.load_model`, or check it first with `check_compatibility`.
Every save is loaded back and compared with the original before it is kept.

**Loading is strict about versions.** A model loads only under the yohou version it was
saved with, and under the same major and minor versions of scikit-learn and polars.
Upgrading yohou therefore means refitting saved forecasters, or loading them with
`strict=False`.

Yohou-MLflow supports Python 3.11 to 3.14, MLflow 3, and yohou 0.1.

## What are the features of Yohou-MLflow?

- **MLflow flavour**: `save_model`, `log_model` and `load_model` for point, interval,
  class-probability, panel and composite forecasters, with model registry support.
- **No code execution on load**: `yohou_mlflow.load_model` never imports code the model
  file names, and a fixed trust policy decides which types a saved model may contain; the
  model file cannot extend it.
- **Save-time verification**: a model that would not load back, or would predict
  differently, is refused when saving rather than months later.
- **Pre-deploy check**: `check_compatibility` reports whether an environment can load a
  registered model, without constructing anything from it.
- **Generic predict interface**: `mlflow.pyfunc.load_model(...).predict({"y": new_rows})`
  observes new rows on a copy and forecasts, with `prediction_type` selecting point,
  interval or class-probability forecasts.

## How to install Yohou-MLflow?

Install the Yohou-MLflow package using `pip`:

```bash
pip install yohou_mlflow
```

or using `uv`:

```bash
uv add yohou_mlflow
```

The package depends on `mlflow-skinny`. To run a local SQLite tracking server and model
registry, as in the tutorial, also install `mlflow`.

## How to get started with Yohou-MLflow?

Register a fitted forecaster:

```python
import mlflow
import yohou_mlflow

with mlflow.start_run():
    yohou_mlflow.log_model(forecaster, name="forecaster", registered_model_name="daily-demand")
```

In a later run, load it, observe new data, forecast, and register the update:

```python
forecaster = yohou_mlflow.load_model("models:/daily-demand/latest")
forecaster.observe(new_rows)
forecast = forecaster.predict()

with mlflow.start_run():
    yohou_mlflow.log_model(forecaster, name="forecaster", registered_model_name="daily-demand")
```

Before upgrading the environment that loads it, check that it still can:

```python
print(yohou_mlflow.check_compatibility("models:/daily-demand/latest"))
```

## How do I use Yohou-MLflow?

Full documentation is available at [https://yohou-mlflow.readthedocs.io/](https://yohou-mlflow.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://yohou-mlflow.readthedocs.io/en/latest/pages/examples/](https://yohou-mlflow.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `marimo edit examples/registry_loop.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/yohou-mlflow/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/yohou-mlflow/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/yohou-mlflow/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [GitHub Discussions](https://github.com/stateful-y/yohou-mlflow/discussions).

## Where can I learn more?

- Full documentation: [https://yohou-mlflow.readthedocs.io/](https://yohou-mlflow.readthedocs.io/)
- GitHub Discussions: [https://github.com/stateful-y/yohou-mlflow/discussions](https://github.com/stateful-y/yohou-mlflow/discussions)
- Interactive Examples: [https://yohou-mlflow.readthedocs.io/en/latest/pages/examples/](https://yohou-mlflow.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/yohou-mlflow/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou-mlflow/blob/main/LICENSE).

## How do I cite Yohou-MLflow?

If you use Yohou-MLflow in work you publish, please cite it:

Guillaume Tauzin. Yohou-MLflow: An MLflow integration for saving and serving Yohou forecasters. https://github.com/stateful-y/yohou-mlflow

Or in BibTeX:

```bibtex
@software{yohou_mlflow,
  author  = "Guillaume Tauzin",
  title   = "{Yohou-MLflow: An MLflow integration for saving and serving Yohou forecasters}",
  url     = "https://github.com/stateful-y/yohou-mlflow",
  license = "Apache-2.0"
}
```

Reference managers can read [CITATION.cff](https://github.com/stateful-y/yohou-mlflow/blob/main/CITATION.cff) directly. To cite a specific version, see the [citation page](https://yohou-mlflow.readthedocs.io/en/latest/pages/reference/citation/).

## Acknowledgements

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in MLOps and data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
