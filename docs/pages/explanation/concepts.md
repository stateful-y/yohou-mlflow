# About Saving Forecasters

A yohou forecaster that runs in production is rarely fitted and thrown away. A
scheduled job loads the latest saved forecaster, gives it the newest observations,
forecasts, and saves it again for the next run, so a single saved forecaster lives for
weeks or months. This page discusses what that asks of a saved model, and why
Yohou-MLflow answers the way it does. It grew out of
[yohou#214](https://github.com/stateful-y/yohou/issues/214), which describes that loop.

## The problem with pickle

Before this package, the only way to store a forecaster in MLflow was to pickle it
through MLflow's generic Python model. A pickle records each object's class location and
internal attributes as they were when it was saved, and loading it runs whatever code
the file names. Two things go wrong as a result:

- **A newer yohou may not read an old pickle.** When a release renames, moves or
  restructures an attribute, loading either fails with an error that does not name the
  cause, or succeeds and the forecaster misbehaves later. Either way, the scheduled job
  finds out, not the deployment.
- **Loading is running code.** A saved model moves between environments and people, and
  a pickle from an untrusted source can do anything the loading process can.

Yohou-MLflow addresses both, with one deliberate limit on the first.

## What a saved model contains

A saved model is a directory with MLflow's `MLmodel` file and one file per
*component*. Today there is one component, `forecaster`, holding the whole fitted
forecaster: its parameters, what it learned in `fit`, and everything it accumulated
through `observe` since. That is why a loaded forecaster continues exactly where the
saved one stopped.

The component layout is there for a future need. `observe` changes what a forecaster
has seen but never its fitted estimators, which are most of its size. A later format
version could store the estimators once and only the observed state per run. Doing that
today would mean reading attributes yohou keeps private, which would couple this
package to yohou's internals, so the layout makes room for the split without
performing it.

## Why loading is strict about versions

Loading a model under a newer yohou needs code that maps old fitted state onto new
classes. Only yohou can maintain that mapping reliably, because only yohou's own tests
see every change to its internals. A package outside yohou that tried would break
whenever yohou renamed an attribute, and nothing in yohou's CI would notice.

So this package does not attempt it. It records the versions of yohou, scikit-learn,
polars, skops and itself when saving, and by default it refuses to load when yohou
differs at all or scikit-learn or polars differ in their major or minor version. The
error names each package with both versions. yohou is compared exactly because it is
pre-1.0 and any release may change fitted state; patch releases of scikit-learn and
polars are let through so that routine dependency updates do not force a refit.

This is a trade-off. Upgrading yohou means refitting saved forecasters, or loading them
with `strict=False` and checking the result. In exchange, the "loads, then misbehaves"
failure becomes a clear refusal, and `check_compatibility` lets you find it before a
deployment rather than during a scheduled run. Version tolerance, with migrations and
tests that load models saved by older releases, is planned as a yohou change, where the
knowledge it needs lives.

## Why the file cannot say what to trust

The forecaster is written with [skops](https://skops.readthedocs.io/), not pickle. skops
stores objects as data and rebuilds them only from types the loader trusts, so loading
never runs code the file chooses. What matters is where that list of trusted types comes
from.

MLflow's own scikit-learn flavour, when it uses skops, stores the trusted types in the
saved model. That trusts whatever the file says, and the file is exactly the thing
being checked. Yohou-MLflow keeps its trust policy in code instead: any class from
yohou or scikit-learn, polars frames and dtypes, and date and time types. The user
already runs yohou's and scikit-learn's code, so trusting their classes adds no new
source of code. Anything else, such as an estimator class from your own project, must
be named by the caller, when saving and again on every load (see
[How to save a forecaster that contains your own estimator classes](../how-to/trust-third-party-estimators.md)). The type list recorded in
`MLmodel` is only there to be read.

The same reasoning rules out bundled code. MLflow flavours usually let a model carry a
`code/` directory that loading puts on Python's import path. For this package that would
be code from the file by definition, and worse: a model could ship a package named like a
trusted one, such as `yohou`, and have it imported while the forecaster's types are
resolved. So a yohou model carries no bundled code, and `load_model` never puts anything
from the model directory on the import path.

The guarantee belongs to `yohou_mlflow.load_model` and `check_compatibility`. MLflow's
generic `mlflow.pyfunc.load_model` runs first and on MLflow's terms: it imports the loader
module and any bundled code that the `MLmodel` file names, before this package is called.
A model you have not reviewed should therefore go through `check_compatibility` or
`yohou_mlflow.load_model`, never straight to the generic interface.

One boundary is worth knowing. A polars frame is stored as polars' own binary format and
rebuilt by polars' native reader. That format cannot hold Python objects, so no Python
code runs, but a crafted file does reach a native parser.

## Why every save is loaded back

A model that saves correctly but cannot be loaded is the worst case for a scheduled job,
because the failure appears months after the save. This is not hypothetical: skops 0.15.0
saved any forecaster fitted on time-zone-aware data, even UTC, without error, and then
failed to load it, because it could not rebuild the `zoneinfo.ZoneInfo` that yohou keeps
in `observed_time_`. The check below caught it before any such model was kept, and the fix
landed in skops 0.16.0, which this package now requires.

So `save_model` loads what it just wrote and compares its predictions with the
original's before keeping anything. The check is general: it caught the time-zone case,
and it catches whatever gap a future skops or polars release introduces. When a forecaster
cannot predict without inputs, for example because it needs future features, the load is
still verified and the comparison is skipped with a warning.

## What the generic predict interface does with state

MLflow's generic `pyfunc` interface is stateless: a call takes an input and returns a
result, and a loaded model may serve many calls, possibly at once. A forecaster is the
opposite, since `observe` changes it.

Yohou-MLflow resolves this by working on a copy for every call. With new observations,
the copy observes them and forecasts once from the last observed time; without them, it
forecasts from the saved state. The update is then discarded, so one call cannot change
what the next call sees. Keeping an update is the job of the native API, as in the
[tutorial](../tutorials/getting-started.md): load, observe, and register a new version.

yohou also offers `observe_predict`, which returns a rolling backtest: a forecast before
observing, then one after each block of new rows. That is useful for evaluation, but a
scheduled job wants the single forecast made after all the new data, so the generic
interface does not use it.

## Why the generic interface takes polars, not pandas

MLflow's generic interface is usually fed pandas, which is what its REST server and Spark
integration produce. yohou works in polars, and a forecaster can take up to four frames
(`y`, `X_actual`, `X_future`, `X_forecast`). Packing four frames into one pandas frame
proved lossy: integer columns came back as floats and dates as datetimes. Since MLflow
passes a dict of polars frames to the model untouched, that is the input this package
accepts. The cost is that `mlflow models serve` and Spark UDFs cannot use these models
yet; a pandas adapter can be added later without changing the dict input.

## What it costs

Each registered version is a full copy of the forecaster. For a direct reduction
forecasting 28 days ahead from three years of daily data, measured on one laptop:

| Regressor | Saved file | `save_model` (load-back check included / skipped) | Generic predict (saved state / 7 new rows) |
|---|---|---|---|
| `Ridge` | 17 KiB | 0.03 s / 0.01 s | 3 ms / 4 ms |
| `HistGradientBoostingRegressor` (100 iterations) | 6.6 MiB | 2.22 s / 1.57 s | 70 ms / 78 ms |

The time costs are small next to a scheduled job's run time. For the larger model, the
load-back check adds about 40 percent to a save, and the per-call copy about a quarter of
a generic prediction. For `Ridge` they add about 20 ms and under 1 ms, which is
large in proportion but small in absolute terms. The size is what accumulates: a daily job registering the
gradient-boosting forecaster adds about 2.4 GiB a year, which is why
[keeping the registry small](../how-to/keep-the-registry-small.md) matters, and why
storing the fitted estimators only once is the natural next step for the format.

## Connections

- [Saved model format](../reference/saved-model-format.md): the exact trust policy,
  version rules and `MLmodel` layout
- [pyfunc contract](../reference/pyfunc-contract.md): every input, param and error of the
  generic interface
- [How to check that a registered model will load](../how-to/check-before-deploying.md)
- [How to move pickled forecasters to the yohou flavour](../how-to/migrate-from-pickle.md)
- [Security](security.md): how this package's releases are published and verified
