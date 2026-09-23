"""Errors and warnings raised by the yohou MLflow flavour."""

from __future__ import annotations

from collections.abc import Sequence

from yohou_mlflow._versions import VersionMismatch, describe_mismatches


class YohouMlflowError(Exception):
    """Base class for every error this package raises on purpose."""


class UntrustedTypesError(YohouMlflowError):
    """A saved forecaster contains types outside the trust policy.

    Parameters
    ----------
    types : sequence of str
        Fully qualified names of the types outside the policy.
    when : {"save", "load"}
        Whether the problem was found while saving or while loading.

    Attributes
    ----------
    types : tuple of str
        Fully qualified names of the types outside the policy, sorted.
    """

    def __init__(self, types: Sequence[str], when: str) -> None:
        self.types = tuple(sorted(types))
        listed = "\n".join(f"  - {name}" for name in self.types)
        if when == "save":
            action = "It would be refused when loaded, so it was not saved."
        else:
            action = "Nothing from the file was constructed."
        super().__init__(
            f"The forecaster contains types outside yohou-mlflow's trust policy:\n{listed}\n"
            f"{action} If you trust these types, pass them in `extra_trusted_types`, "
            "both when saving and every time the model is loaded."
        )


class VersionMismatchError(YohouMlflowError):
    """The installed packages differ from those the model was saved with.

    Parameters
    ----------
    mismatches : sequence of VersionMismatch
        One entry per package whose installed version breaks the comparison rule.

    Attributes
    ----------
    mismatches : tuple of VersionMismatch
        The mismatches, in the order they were found.
    """

    def __init__(self, mismatches: Sequence[VersionMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__(describe_mismatches(self.mismatches))


class SaveVerificationError(YohouMlflowError):
    """A just-written model could not be loaded back, or loaded back differently."""


class FormatVersionError(YohouMlflowError):
    """The model was saved in a format version this release cannot read."""


class VersionMismatchWarning(UserWarning):
    """A model was loaded with `strict=False` despite package version mismatches."""
