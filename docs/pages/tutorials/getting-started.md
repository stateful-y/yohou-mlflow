# Getting Started

In this tutorial, we will install Yohou-MLflow and run a first example.

<!-- COMPANION_NOTEBOOKS -->

## Installation

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install yohou_mlflow
    ```

=== "uv"

    ```bash
    uv add yohou_mlflow
    ```

=== "conda"

    ```bash
    conda install -c conda-forge yohou_mlflow
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge yohou_mlflow
    ```

> **Note**: For conda/mamba, ensure the package is published to conda-forge first.

Verify the installation:

```python
import yohou_mlflow
print(yohou_mlflow.__version__)
```

## Your First Example

```python
from yohou_mlflow.example import hello

result = hello("World")
print(result)  # Output: Hello, World!
```

## Next Steps

- **Learn the concepts**: Read [Concepts](../explanation/concepts.md) to understand the design
- **Explore examples**: Check out the [Examples](../examples/index.md) for interactive notebooks
- **Dive into the API**: Browse the [API Reference](../reference/api.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/yohou-mlflow/discussions) or [open an issue](https://github.com/stateful-y/yohou-mlflow/issues)
