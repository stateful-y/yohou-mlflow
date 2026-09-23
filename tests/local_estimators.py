"""Estimators defined in the test suite, so their types fall outside the trust policy."""

from sklearn.linear_model import Ridge


class LocalRidge(Ridge):
    """A ``Ridge`` subclass whose fully qualified name no policy entry matches."""


def identity(values):
    """Return ``values`` unchanged; a function defined outside yohou and scikit-learn."""
    return values
