"""Record package versions at save time and compare them at load time."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version

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


def _major_minor(text: str) -> tuple[int, ...] | str:
    """Return the (major, minor) release of a version, or the text itself if unparseable."""
    try:
        return Version(text).release[:2]
    except InvalidVersion:
        # An unparseable version can only be compared as a whole.
        return text


def compare_versions(recorded: Mapping[str, str], installed: Mapping[str, str] | None = None) -> list[VersionMismatch]:
    """Compare recorded versions with installed ones under ``VERSION_RULES``.

    Parameters
    ----------
    recorded : mapping of str to str
        Versions recorded in the model's flavour configuration.
    installed : mapping of str to str or None, default=None
        Versions to compare against. ``None`` reads the installed packages.

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
    current = dict(installed) if installed is not None else installed_versions()
    mismatches = []
    for package, rule in VERSION_RULES.items():
        if rule is None or package not in recorded:
            continue
        saved, now = str(recorded[package]), current[package]
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
