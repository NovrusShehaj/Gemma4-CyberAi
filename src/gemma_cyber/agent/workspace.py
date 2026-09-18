"""The workspace jail — a real security boundary, not a convention (plan §12).

Every path a tool touches passes through :meth:`Workspace.resolve_in_jail`,
which canonicalises first and compares second. Canonicalising first is the whole
point: ``docs/../../../etc/passwd`` and a symlink named ``notes.txt`` pointing at
``/etc/shadow`` both look workspace-relative until ``realpath`` is applied.

What lives here: root discovery, canonicalisation, ignore rules, binary/size
limits, and the cheap repo stub. What does *not* live here: the permission mode
and the sensitive-path denylist. Those are policy and belong to
`permissions.py`, which is the only module allowed to answer "may this happen".
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma_cyber.agent.errors import ToolError, WorkspaceEscape

try:  # pathspec ships with the [agent] extra; degrade to built-in rules without it.
    import pathspec
except ImportError:  # pragma: no cover - exercised only in a base install
    pathspec = None  # type: ignore[assignment]

__all__ = ["Workspace", "WorkspaceLimits", "discover_root"]

#: Directories that are always skipped regardless of .gitignore. `.git` is the
#: important one: it holds credentials in `config`, packed history, and hooks.
ALWAYS_IGNORED_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
     ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".uv-cache", "dist", "build"}
)

INSTRUCTION_FILENAMES = ("GEMMA4.md",)
INSTRUCTION_OVERLAY = Path(".gemma4") / "instructions.md"

_BINARY_SNIFF_BYTES = 8192

#: Markers that identify a project root, in probe order.
_ROOT_MARKERS = (".git",)

#: One-line project hints for the workspace stub (plan §12, "cheap").
_PROJECT_HINTS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "python"),
    ("requirements.txt", "python"),
    ("package.json", "node"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("pom.xml", "java"),
    ("Gemfile", "ruby"),
    ("Dockerfile", "docker"),
    ("docker-compose.yml", "docker-compose"),
)


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    max_file_bytes: int = 1_048_576
    tree_max_entries: int = 200
    instructions_max_bytes: int = 16_384


def discover_root(
    explicit: Path | None = None, *, start: Path | None = None, allow_home: bool = False
) -> Path:
    """Find the workspace root (plan §12): ``--cwd`` → nearest ``.git`` → cwd.

    Refuses ``$HOME`` and the filesystem root unless ``allow_home``. That refusal
    is not paternalism: a jail rooted at ``$HOME`` contains ``.ssh``, ``.aws``
    and every other repo on the machine, which makes the jail meaningless.
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise ToolError(f"workspace path is not a directory: {root.name or root}")
    else:
        cursor = (start or Path.cwd()).resolve(strict=False)
        root = cursor
        for candidate in (cursor, *cursor.parents):
            if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
                root = candidate
                break

    home = Path.home().resolve(strict=False)
    if not allow_home and (root == home or root == Path(root.anchor)):
        where = "$HOME" if root == home else "the filesystem root"
        raise WorkspaceEscape(
            f"refusing to use {where} as the workspace root: it would place "
            "credentials and unrelated repositories inside the jail. "
            "Pass --cwd <project> or, deliberately, --allow-home."
        )
    return root


class Workspace:
    """A canonicalised root plus the rules for what may be touched inside it."""

    def __init__(
        self,
        root: Path,
        *,
        limits: WorkspaceLimits | None = None,
        extra_ignore: tuple[str, ...] = (),
    ) -> None:
        # Resolve once at construction: every later comparison is against a
        # canonical root, so a symlinked root (e.g. /tmp on macOS) is not a hole.
        self.root = Path(root).expanduser().resolve(strict=False)
        self.limits = limits or WorkspaceLimits()
        self._spec = self._load_ignore_spec(extra_ignore)

    # -- ignore rules -------------------------------------------------------

    def _load_ignore_spec(self, extra: tuple[str, ...]) -> Any:
        if pathspec is None:  # pragma: no cover - base install only
            return None
        patterns: list[str] = [f"{name}/" for name in sorted(ALWAYS_IGNORED_DIRS)]
        for name in (".gitignore", ".gemma4ignore"):
            candidate = self.root / name
            try:
                if candidate.is_file():
                    patterns.extend(candidate.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
        patterns.extend(extra)
        # GitIgnoreSpec implements gitignore precedence (later negations win);
        # older pathspec releases only expose the pattern factory by name.
        factory = getattr(pathspec, "GitIgnoreSpec", None)
        if factory is not None:
            return factory.from_lines(patterns)
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)  # pragma: no cover

    def is_ignored(self, path: Path) -> bool:
        """True if ``path`` matches .gitignore / .gemma4ignore / always-ignored."""
        try:
            rel = self.relative(path)
        except WorkspaceEscape:
            return True
        parts = Path(rel).parts
        if any(part in ALWAYS_IGNORED_DIRS for part in parts):
            return True
        if self._spec is None:  # pragma: no cover - base install only
            return False
        candidates = [rel]
        if path.is_dir():
            candidates.append(rel + "/")
        return bool(any(self._spec.match_file(c) for c in candidates))

    # -- the jail -----------------------------------------------------------

    def resolve_in_jail(self, user_path: str | Path) -> Path:
        """Canonicalise ``user_path`` and require it to be inside the workspace.

        Raises :class:`WorkspaceEscape` for traversal, absolute paths outside the
        root, and symlinks whose target leaves the root. Resolution is
        ``strict=False`` so a not-yet-created file is still checked — including
        the case where its *parent* is a symlink pointing out.
        """
        raw = str(user_path).strip()
        if not raw:
            raise WorkspaceEscape("empty path")
        if "\x00" in raw:
            raise WorkspaceEscape("path contains a NUL byte")

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            # Unresolvable (symlink loop, permission on a parent) → deny. A path
            # whose location cannot be established is never allowed through.
            raise WorkspaceEscape(f"could not resolve path safely: {exc.__class__.__name__}") from exc

        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceEscape(
                f"path escapes the workspace: {raw!r} resolves outside {self.root.name}/"
            )
        return resolved

    def relative(self, path: Path) -> str:
        """Workspace-relative POSIX string, for display, hashing keys, and audit."""
        resolved = Path(path).resolve(strict=False)
        if resolved == self.root:
            return "."
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            raise WorkspaceEscape("path is outside the workspace") from None

    # -- file access --------------------------------------------------------

    def open_read_bytes(self, path: Path, *, max_bytes: int | None = None) -> bytes:
        """Read a jailed file with size and symlink-race checks.

        ``O_NOFOLLOW`` is a TOCTOU guard: :meth:`resolve_in_jail` already removed
        every symlink, so if the final component *is* a symlink by the time we
        open it, something replaced it between check and use. Fail closed.
        """
        cap = self.limits.max_file_bytes if max_bytes is None else max_bytes
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if exc.errno in (getattr(os, "ELOOP", 62), 40):
                raise WorkspaceEscape("path became a symlink during the read") from exc
            raise ToolError(f"cannot read {self.relative(path)}: {exc.strerror}") from exc
        try:
            stat = os.fstat(fd)
            if not os.path.stat.S_ISREG(stat.st_mode):  # type: ignore[attr-defined]
                raise ToolError(f"not a regular file: {self.relative(path)}")
            if stat.st_size > cap:
                raise ToolError(
                    f"file is {stat.st_size} bytes, over the {cap}-byte limit "
                    f"({self.relative(path)}); narrow the read with fs.grep"
                )
            with os.fdopen(fd, "rb", closefd=True) as fh:
                data = fh.read(cap + 1)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        if len(data) > cap:
            raise ToolError(f"file exceeds the {cap}-byte limit ({self.relative(path)})")
        return data

    @staticmethod
    def looks_binary(data: bytes) -> bool:
        """NUL byte in the first 8 KiB — the plan's heuristic (§12)."""
        return b"\x00" in data[:_BINARY_SNIFF_BYTES]

    def read_text(self, path: Path, *, max_bytes: int | None = None) -> str:
        """Read a jailed text file, refusing binaries with a usable message."""
        data = self.open_read_bytes(path, max_bytes=max_bytes)
        if self.looks_binary(data):
            raise ToolError(f"binary file, not read: {self.relative(path)}")
        return data.decode("utf-8", errors="replace")

    # -- enumeration --------------------------------------------------------

    def iter_files(self, *, include_hidden: bool = False) -> Iterator[Path]:
        """Yield non-ignored regular files under the root.

        Directory symlinks are never followed: ``os.walk(followlinks=False)`` is
        what keeps a symlinked ``vendor/ -> /`` from turning enumeration into a
        whole-filesystem scan.
        """
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in ALWAYS_IGNORED_DIRS
                and (include_hidden or not d.startswith("."))
                and not self.is_ignored(here / d)
            ]
            for name in sorted(filenames):
                if not include_hidden and name.startswith("."):
                    continue
                candidate = here / name
                if candidate.is_symlink():
                    # A file symlink is only safe if its target stays inside.
                    try:
                        self.resolve_in_jail(candidate)
                    except WorkspaceEscape:
                        continue
                if self.is_ignored(candidate):
                    continue
                yield candidate

    def tree_stub(self, *, depth: int = 2) -> str:
        """Depth-2 ignored-aware outline — the MVP's entire repo understanding.

        Deliberately cheap (plan §12): no embeddings, no whole-repo dump. The
        model discovers the rest with `fs.glob` / `fs.grep` / `fs.read`.
        """
        lines: list[str] = []
        budget = self.limits.tree_max_entries

        def walk(directory: Path, prefix: str, level: int) -> None:
            nonlocal budget
            if level > depth or budget <= 0:
                return
            try:
                entries = sorted(
                    directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                )
            except OSError:
                return
            for entry in entries:
                if budget <= 0:
                    return
                if entry.name.startswith(".") and entry.name not in {".github", ".gemma4"}:
                    continue
                if entry.is_symlink():
                    continue
                if self.is_ignored(entry):
                    continue
                budget -= 1
                if entry.is_dir():
                    lines.append(f"{prefix}{entry.name}/")
                    walk(entry, prefix + "  ", level + 1)
                else:
                    lines.append(f"{prefix}{entry.name}")

        walk(self.root, "", 1)
        if budget <= 0:
            lines.append(f"... (truncated at {self.limits.tree_max_entries} entries)")
        return "\n".join(lines)

    def project_hints(self) -> list[str]:
        """Presence-based one-liners. Not a language server (plan §12)."""
        hints: list[str] = []
        for filename, label in _PROJECT_HINTS:
            if (self.root / filename).is_file() and label not in hints:
                hints.append(label)
        return hints

    # -- project instructions ----------------------------------------------

    def instruction_files(self, *, start: Path | None = None) -> list[tuple[str, str]]:
        """Collect ``GEMMA4.md`` / ``.gemma4/instructions.md``, nearest last.

        Returns ``(label, text)`` pairs. The caller wraps these as UNTRUSTED —
        they are repository content and carry exactly zero authority (plan §19).
        Each file is size-capped so a hostile repo cannot evict the system policy
        by shipping a megabyte of "instructions".
        """
        cursor = (start or self.root).resolve(strict=False)
        try:
            self.relative(cursor)
        except WorkspaceEscape:
            cursor = self.root

        chain: list[Path] = []
        while True:
            chain.append(cursor)
            if cursor == self.root:
                break
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        chain.reverse()  # root first, nearest directory last

        collected: list[tuple[str, str]] = []
        seen: set[Path] = set()
        for directory in chain:
            for rel in (*(Path(n) for n in INSTRUCTION_FILENAMES), INSTRUCTION_OVERLAY):
                candidate = directory / rel
                if candidate in seen or not candidate.is_file():
                    continue
                seen.add(candidate)
                try:
                    text = self.read_text(
                        self.resolve_in_jail(candidate),
                        max_bytes=self.limits.instructions_max_bytes,
                    )
                except (ToolError, WorkspaceEscape):
                    continue
                collected.append((f"workspace:{self.relative(candidate)}", text))
        return collected

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace(root={self.root})"
