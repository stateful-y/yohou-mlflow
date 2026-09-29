"""Record package versions at save time and compare them at load time."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, packages_distributions, version

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

# Distribution name -> comparison rule. Recorded but uncompared packages map to None:
# skops checks its own protocol version, and yohou-mlflow checks `format_version`.
VERSION_RULES: dict[str, str | None] = {
    # yohou is pre-1.0 alpha: any release can change fitted state.
    "yohou": "exact",
    # Fitted state can change between minor releases; patch releases are let through
    # so that routine patch bumps do not force a refit.
    "scikit-learn": "major.minor",
    "polars": "major.minor",
    "skops": None,
    "yohou-mlflow": None,
}


@dataclass(frozen=True)
class VersionMismatch:
    """One package whose installed version breaks its comparison rule.

    Parameters
    ----------
    package : str
        Distribution name, for example ``"yohou"``.
    recorded : str
        Version recorded when the model was saved.
    installed : str
        Version installed now.
    rule : {"exact", "major.minor"}
        The comparison the two versions failed.
    """

    package: str
    recorded: str
    installed: str
    rule: str

    def __str__(self) -> str:
        """Describe the mismatch in one line.

        Returns
        -------
        str
            For example ``"yohou: saved with 0.1.0a13, installed 0.1.0a14 (must match exactly)"``.
        """
        need = "must match exactly" if self.rule == "exact" else "major and minor versions must match"
        return f"{self.package}: saved with {self.recorded}, installed {self.installed} ({need})"


def installed_versions() -> dict[str, str]:
    """Return the installed version of every package the flavour records.

    Returns
    -------
    dict of str to str
        Distribution name to installed version, for each key of ``VERSION_RULES``.
    """
    return {package: version(package) for package in VERSION_RULES}


_RULE_NAMES = ("exact", "major.minor")
_NOT_INSTALLED = "(not installed)"


def validate_version_rules(rules: Mapping[str, str] | None) -> dict[str, str]:
    """Return caller-declared version rules keyed by canonical distribution name.

    Parameters
    ----------
    rules : mapping of str to str or None
        Distribution name to ``"exact"`` or ``"major.minor"``.

    Returns
    -------
    dict of str to str
        The rules, with each name canonicalized (``"Scikit_Learn"`` becomes
        ``"scikit-learn"``).

    Raises
    ------
    ValueError
        If a rule is neither ``"exact"`` nor ``"major.minor"``, or names a package
        ``VERSION_RULES`` already covers: a caller may add rules, never change one.
    """
    checked = {str(canonicalize_name(package)): rule for package, rule in (rules or {}).items()}
    built_in = sorted(set(checked) & set(VERSION_RULES))
    if built_in:
        msg = f"Version rules for {built_in} are built in and cannot be changed."
        raise ValueError(msg)
    unknown = {package: rule for package, rule in checked.items() if rule not in _RULE_NAMES}
    if unknown:
        msg = f"Each version rule must be one of {list(_RULE_NAMES)}, got {unknown}."
        raise ValueError(msg)
    return checked


def held_versions(type_names: Iterable[str], extra_version_rules: Mapping[str, str] | None) -> dict[str, str]:
    """Return the installed version of each declared package the saved types come from.

    Parameters
    ----------
    type_names : iterable of str
        Fully qualified type names in the saved file.
    extra_version_rules : mapping of str to str or None
        Caller-declared rules; only their package names are used here.

    Returns
    -------
    dict of str to str
        Canonical distribution name to installed version, for each declared package
        that provides the top-level module of at least one of ``type_names``.
    """
    rules = validate_version_rules(extra_version_rules)
    if not rules:
        return {}
    held_modules = {name.partition(".")[0] for name in type_names}
    held: dict[str, str] = {}
    for module, distributions in packages_distributions().items():
        if module not in held_modules:
            continue
        for distribution in distributions:
            package = canonicalize_name(distribution)
            if package in rules:
                held[package] = version(package)
    return held


def _installed_version(package: str) -> str:
    """Return the installed version of ``package``, or a marker when it is not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return _NOT_INSTALLED


def _major_minor(text: str) -> tuple[int, ...] | str:
    """Return the (major, minor) release of a version, or the text itself if unparseable."""
    try:
        return Version(text).release[:2]
    except InvalidVersion:
        # An unparseable version can only be compared as a whole.
        return text


def compare_versions(
    recorded: Mapping[str, str],
    installed: Mapping[str, str] | None = None,
    extra_version_rules: Mapping[str, str] | None = None,
) -> list[VersionMismatch]:
    """Compare recorded versions with installed ones under ``VERSION_RULES`` and the caller's rules.

    Parameters
    ----------
    recorded : mapping of str to str
        Versions recorded in the model's flavour configuration.
    installed : mapping of str to str or None, default=None
        Versions to compare against. ``None`` reads the installed packages.
    extra_version_rules : mapping of str to str or None, default=None
        Rules for packages outside ``VERSION_RULES``, compared only when the model
        recorded them.

    Returns
    -------
    list of VersionMismatch
        One entry per compared package that breaks its rule. Empty when compatible.

    Examples
    --------
    >>> saved = {"yohou": "0.1.0a13", "scikit-learn": "1.9.1", "polars": "1.44.2"}
    >>> now = {"yohou": "0.1.0a13", "scikit-learn": "1.9.2", "polars": "1.45.0"}
    >>> [str(m) for m in compare_versions(saved, now)]
    ['polars: saved with 1.44.2, installed 1.45.0 (major and minor versions must match)']
    """
    rules = {**VERSION_RULES, **validate_version_rules(extra_version_rules)}
    if installed is not None:
        current = dict(installed)
    else:
        current = installed_versions()
        current.update({package: _installed_version(package) for package in rules})
    mismatches = []
    for package, rule in rules.items():
        if rule is None or package not in recorded:
            continue
        saved, now = str(recorded[package]), current.get(package, _NOT_INSTALLED)
        differs = saved != now if rule == "exact" else _major_minor(saved) != _major_minor(now)
        if differs:
            mismatches.append(VersionMismatch(package, saved, now, rule))
    return mismatches


def describe_mismatches(mismatches: Iterable[VersionMismatch]) -> str:
    """Return the error and warning text for a set of version mismatches.

    Parameters
    ----------
    mismatches : iterable of VersionMismatch
        The mismatches to describe.

    Returns
    -------
    str
        A message naming each package with both versions, what to do, and the
        ``strict=False`` escape.
    """
    listed = "\n".join(f"  - {m}" for m in mismatches)
    return (
        "This model was saved with package versions that differ from the installed ones:\n"
        f"{listed}\n"
        "Loading it could fail, or succeed and predict differently. Refit and save the "
        "forecaster under the installed versions, or install the versions it was saved "
        "with. To load it anyway, pass strict=False."
    )
