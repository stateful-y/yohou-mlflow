# How to Keep the Registry Small When Every Run Registers a Version

This guide shows you how to delete old versions of a registered forecaster, and the
files behind them, after each scheduled run. Use it when a job registers a new version
on every run, since each version stores a full copy of the forecaster.

## Prerequisites

- A registered model that receives a new version per run
- MLflow 3, where each registered version points to a logged model
  (its `source` is `models:/m-...`)

## Steps

### 1. Decide what to keep

Pick how many recent versions to keep for rollbacks, and set an
[alias](https://mlflow.org/docs/latest/ml/model-registry/) such as `champion` on any
version that must never be deleted:

```python
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("daily-demand", "champion", "12")
```

### 2. Delete older versions and their logged models

Run this after registering the new version:

```python
name, keep = "daily-demand", 5

versions = client.search_model_versions(f"name='{name}'")
for version in sorted(versions, key=lambda v: int(v.version), reverse=True)[keep:]:
    # search results do not include aliases; fetch each version to read them
    details = client.get_model_version(name, version.version)
    if details.aliases:
        continue
    client.delete_model_version(name, version.version)
    # A version's source is "models:/<logged model id>" for models logged with MLflow 3.
    if details.source.startswith("models:/m-"):
        client.delete_logged_model(details.source.removeprefix("models:/"))
```

Deleting the version alone keeps the logged model and its files. Deleting the logged
model marks it for removal.

### 3. Remove the files

Deleted logged models keep their files until garbage collection. Run it on a schedule,
against the same tracking server:

```bash
MLFLOW_TRACKING_URI=<tracking-uri> mlflow gc --backend-store-uri <tracking-uri>
```

It reports each removed model:

```text
Logged model with ID m-43db1ef6ecea4f8c9f9e8a2b65c857a0 has been permanently deleted.
```

## Troubleshooting

**Problem: a version with an alias was deleted**
: Aliases are only present on `get_model_version` results, not on
  `search_model_versions` results. Check `details.aliases`, as in step 2.

**Problem: the files are still there after deleting versions**
: Delete the logged models as well, then run `mlflow gc`.

## See Also

- [Concepts](../explanation/concepts.md): why each version is a full copy, and the
  measured sizes
