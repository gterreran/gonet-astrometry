"""Lightweight filesystem discovery for native GONet image inputs.

Discovery identifies candidate original GONet ``.jpg`` files from explicit
file paths and directories without parsing their pixel payloads. Native image
loading is deliberately deferred to the caller so command-line and GUI
workflows can inspect large observing directories cheaply.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

GONET_SUFFIXES = frozenset({".jpg"})
"""Filename suffixes accepted as candidate native GONet images."""


@dataclass(frozen=True)
class DiscoveryResult:
    """Result of scanning user-provided files and directories.

    Parameters
    ----------
    files
        Unique candidate GONet files found during discovery.
    directories
        Directories that were searched.
    missing
        Input paths that did not exist.
    unsupported
        Existing explicit files whose suffix is not supported.
    """

    files: tuple[Path, ...]
    directories: tuple[Path, ...]
    missing: tuple[Path, ...]
    unsupported: tuple[Path, ...]

    @property
    def summary(self) -> str:
        """Return a concise user-facing discovery summary."""
        file_word = "file" if len(self.files) == 1 else "files"
        message = f"Discovered {len(self.files)} candidate GONet {file_word}."
        if self.directories:
            directory_word = "folder" if len(self.directories) == 1 else "folders"
            message += f" Searched {len(self.directories)} {directory_word}."

        details: list[str] = []
        if self.missing:
            details.append(f"{len(self.missing)} missing path(s)")
        if self.unsupported:
            details.append(f"{len(self.unsupported)} unsupported file(s)")
        if details:
            message = f"{message} Ignored " + " and ".join(details) + "."
        return message


def parse_source_paths(value: str | None) -> tuple[Path, ...]:
    """Parse newline- or semicolon-separated paths into path objects."""
    if value is None:
        return ()

    entries = value.replace(";", "\n").splitlines()
    return tuple(Path(entry.strip()).expanduser() for entry in entries if entry.strip())


def discover_gonet_files(
    source_paths: Iterable[Path],
    *,
    recursive: bool = True,
) -> DiscoveryResult:
    """Discover candidate native GONet files without loading image data.

    Parameters
    ----------
    source_paths
        Explicit files or directories supplied by the user.
    recursive
        Whether directory searches include nested subdirectories.

    Returns
    -------
    DiscoveryResult
        Candidate files and diagnostic information about searched inputs.

    Notes
    -----
    Discovery checks only filesystem state and the ``.jpg`` suffix. A candidate
    is validated as a real native GONet file only when the Wizard loader parses
    it.
    """
    files: set[Path] = set()
    directories: list[Path] = []
    missing: list[Path] = []
    unsupported: list[Path] = []

    for raw_path in source_paths:
        path = raw_path.expanduser()
        if not path.exists():
            missing.append(path)
            continue

        normalized = path.resolve()
        if normalized.is_file():
            if _is_candidate_file(normalized):
                files.add(normalized)
            else:
                unsupported.append(normalized)
            continue

        if normalized.is_dir():
            directories.append(normalized)
            iterator = normalized.rglob("*") if recursive else normalized.iterdir()
            for candidate in iterator:
                if candidate.is_file() and _is_candidate_file(candidate):
                    files.add(candidate.resolve())
            continue

        unsupported.append(normalized)

    return DiscoveryResult(
        files=tuple(sorted(files, key=lambda item: str(item).casefold())),
        directories=tuple(directories),
        missing=tuple(missing),
        unsupported=tuple(unsupported),
    )


def _is_candidate_file(path: Path) -> bool:
    """Return whether ``path`` resembles an original GONet image file."""
    return (
        path.suffix.lower() in GONET_SUFFIXES
        and not path.name.startswith(".")
        and not path.name.startswith("._")
    )
