"""The fixed trust policy applied to every skops file this package loads.

The policy lives in code, never in the saved model: the file is the untrusted input,
so it cannot be allowed to say what to trust. Types skops already trusts on its own
(most scikit-learn estimators, numpy, scipy) never reach this policy, because
``skops.io.get_untrusted_types`` does not report them.
"""

from __future__ import annotations

from collections.abc import Iterable

# Any class yohou defines. The user already runs yohou's code, so trusting its
# classes adds no new source of code.
TRUSTED_TYPE_PREFIXES: tuple[str, ...] = (
    "yohou.",
    # Any class scikit-learn defines, for the same reason. skops's own defaults leave
    # out some internal types, such as the TreePredictor inside
    # HistGradientBoostingRegressor. Functions are separate entries in a skops file
    # (a user's own function is reported under its module), so this prefix does not
    # trust callables from the user's code.
    "sklearn.",
    # polars dtype classes (Float64, Datetime, Categorical, ...) held by frame schemas.
    "polars.datatypes.",
)

TRUSTED_TYPES: frozenset[str] = frozenset({
    # Rebuilt by polars' own binary deserializer, not by skops. No Python code runs,
    # but a crafted file reaches polars' native parser.
    "polars.dataframe.frame.DataFrame",
    "polars.series.series.Series",
    "datetime.date",
    "datetime.datetime",
    "datetime.timedelta",
    "datetime.timezone",
    "zoneinfo.ZoneInfo",
})


def is_trusted(type_name: str, extra_trusted_types: Iterable[str] = ()) -> bool:
    """Return whether one fully qualified type name passes the trust policy.

    Parameters
    ----------
    type_name : str
        Fully qualified type name, as reported by ``skops.io.get_untrusted_types``.
    extra_trusted_types : iterable of str, default=()
        Exact type names the caller trusts in addition to the built-in policy.

    Returns
    -------
    bool
        ``True`` when the name matches the built-in policy or ``extra_trusted_types``.

    Examples
    --------
    >>> is_trusted("yohou.point.reduction.PointReductionForecaster")
    True
    >>> is_trusted("lightgbm.sklearn.LGBMRegressor")
    False
    >>> is_trusted("lightgbm.sklearn.LGBMRegressor", ["lightgbm.sklearn.LGBMRegressor"])
    True
    """
    return (
        type_name in TRUSTED_TYPES
        or type_name.startswith(TRUSTED_TYPE_PREFIXES)
        or type_name in set(extra_trusted_types)
    )


def untrusted_outside_policy(type_names: Iterable[str], extra_trusted_types: Iterable[str] | None = None) -> list[str]:
    """Return the type names that neither the policy nor the caller trusts.

    Parameters
    ----------
    type_names : iterable of str
        Fully qualified type names, as reported by ``skops.io.get_untrusted_types``.
    extra_trusted_types : iterable of str or None, default=None
        Exact type names the caller trusts in addition to the built-in policy.

    Returns
    -------
    list of str
        The names outside the policy, sorted. Empty when everything is trusted.

    Examples
    --------
    >>> untrusted_outside_policy(["datetime.datetime", "lightgbm.sklearn.LGBMRegressor"])
    ['lightgbm.sklearn.LGBMRegressor']
    """
    extra = tuple(extra_trusted_types or ())
    return sorted({name for name in type_names if not is_trusted(name, extra)})
