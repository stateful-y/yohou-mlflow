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
    "zoneinfo.ZoneInfo",
})


def validate_prefixes(prefixes: Iterable[str] | None) -> tuple[str, ...]:
    """Return caller-given trusted prefixes, refusing any that does not end a module name.

    Parameters
    ----------
    prefixes : iterable of str or None
        Module prefixes, each ending in ``"."``.

    Returns
    -------
    tuple of str
        The prefixes, in order.

    Raises
    ------
    ValueError
        If a prefix is not a module path ending in ``"."``, since ``"mypkg"`` would also
        match a module named ``mypkg_other``.
    """
    checked = tuple(prefixes or ())
    bad = [prefix for prefix in checked if not prefix.endswith(".") or prefix == "."]
    if bad:
        msg = f"Each trusted prefix must be a module path ending in '.', got {bad}."
        raise ValueError(msg)
    return checked


def is_trusted(
    type_name: str, extra_trusted_types: Iterable[str] = (), extra_trusted_prefixes: Iterable[str] = ()
) -> bool:
    """Return whether one fully qualified type name passes the trust policy.

    Parameters
    ----------
    type_name : str
        Fully qualified type name, as reported by ``skops.io.get_untrusted_types``.
    extra_trusted_types : iterable of str, default=()
        Exact type names the caller trusts in addition to the built-in policy.
    extra_trusted_prefixes : iterable of str, default=()
        Module prefixes the caller trusts, each ending in ``"."``.

    Returns
    -------
    bool
        ``True`` when the name matches the built-in policy or the caller's additions.

    Raises
    ------
    ValueError
        If a prefix is not a module path ending in ``"."``.

    Examples
    --------
    >>> is_trusted("yohou.point.reduction.PointReductionForecaster")
    True
    >>> is_trusted("lightgbm.sklearn.LGBMRegressor")
    False
    >>> is_trusted("lightgbm.sklearn.LGBMRegressor", ["lightgbm.sklearn.LGBMRegressor"])
    True
    >>> is_trusted("mypkg.features.Lags", extra_trusted_prefixes=["mypkg."])
    True
    """
    return _is_trusted(type_name, frozenset(extra_trusted_types), validate_prefixes(extra_trusted_prefixes))


def _is_trusted(type_name: str, extra_trusted_types: frozenset[str], extra_trusted_prefixes: tuple[str, ...]) -> bool:
    """Match one name against the policy and already-validated caller additions."""
    return (
        type_name in TRUSTED_TYPES
        or type_name.startswith(TRUSTED_TYPE_PREFIXES + extra_trusted_prefixes)
        or type_name in extra_trusted_types
    )


def untrusted_outside_policy(
    type_names: Iterable[str],
    extra_trusted_types: Iterable[str] | None = None,
    extra_trusted_prefixes: Iterable[str] | None = None,
) -> list[str]:
    """Return the type names that neither the policy nor the caller trusts.

    Parameters
    ----------
    type_names : iterable of str
        Fully qualified type names, as reported by ``skops.io.get_untrusted_types``.
    extra_trusted_types : iterable of str or None, default=None
        Exact type names the caller trusts in addition to the built-in policy.
    extra_trusted_prefixes : iterable of str or None, default=None
        Module prefixes the caller trusts, each ending in ``"."``.

    Returns
    -------
    list of str
        The names outside the policy, sorted. Empty when everything is trusted.

    Examples
    --------
    >>> untrusted_outside_policy(["datetime.datetime", "lightgbm.sklearn.LGBMRegressor"])
    ['lightgbm.sklearn.LGBMRegressor']
    """
    extra = frozenset(extra_trusted_types or ())
    prefixes = validate_prefixes(extra_trusted_prefixes)
    return sorted({name for name in type_names if not _is_trusted(name, extra, prefixes)})
