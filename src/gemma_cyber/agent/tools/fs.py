"""Filesystem tools: `fs.read`, `fs.glob`, `fs.grep`, `fs.edit`, `fs.write`.

The toolbelt is small on purpose (plan §9.3, §70). A 4B-class model given twenty
overlapping tools picks badly; five well-described ones with strict schemas is
the design that works.

Every path argument is declared in ``path_fields``, so `PermissionGuard` jails
and denylist-checks it *before* any code here runs. These tools still call
`resolve_in_jail` themselves — defence in depth, and because the resolved path
is what they need anyway.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from gemma_cyber.agent.errors import PatchConflict, ToolError, WorkspaceEscape
from gemma_cyber.agent.sessions import UndoStore
from gemma_cyber.agent.tools.base import Tool, ToolContext
from gemma_cyber.agent.types import AgentState, SideEffect, ToolResult

__all__ = [
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ReadFileTool",
    "WriteFileTool",
    "builtin_fs_tools",
    "sha256_bytes",
]

#: Lines of context shown either side of a grep hit and in edit previews.
_DIFF_CONTEXT = 3
_MAX_GREP_LINE_CHARS = 400


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- fs.read ----------------------------------------------------------------

class ReadArgs(BaseModel):
    path: str = Field(description="Path to a file, relative to the workspace root.")
    start_line: int | None = Field(
        default=None, ge=1, description="First line to return (1-based). Omit for the whole file."
    )
    end_line: int | None = Field(default=None, ge=1, description="Last line to return (inclusive).")


class ReadFileTool(Tool):
    name = "fs.read"
    description = (
        "Read a UTF-8 text file from the workspace. Returns numbered lines and the "
        "file's sha256, which fs.edit requires. Binary files and files above the "
        "size limit are refused."
    )
    input_model = ReadArgs
    side_effect = SideEffect.READ
    timeout_s = 15.0
    path_fields = ("path",)

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, ReadArgs)
        path = ctx.workspace.resolve_in_jail(args.path)
        if not path.exists():
            raise ToolError(f"no such file: {args.path}")
        if path.is_dir():
            raise ToolError(f"{args.path} is a directory; use fs.glob to list it")

        data = ctx.workspace.open_read_bytes(path, max_bytes=ctx.max_file_bytes)
        if ctx.workspace.looks_binary(data):
            raise ToolError(f"binary file, not read: {args.path}")

        digest = sha256_bytes(data)
        rel = ctx.workspace.relative(path)
        # Record the hash so a later fs.edit can prove it is editing what it read.
        ctx.state.record_file_hash(rel, digest)

        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = (args.start_line or 1) - 1
        end = args.end_line if args.end_line is not None else len(lines)
        if start >= len(lines) and lines:
            raise ToolError(
                f"start_line {args.start_line} is past the end of {rel} ({len(lines)} lines)"
            )
        window = lines[start:end]
        numbered = "\n".join(f"{start + i + 1:>6}\t{line}" for i, line in enumerate(window))

        header = f"{rel} ({len(lines)} lines, sha256={digest})"
        if args.start_line or args.end_line:
            header += f" [lines {start + 1}-{min(end, len(lines))}]"
        result = self.ok(f"{header}\n{numbered}")
        result.metadata = {"path": rel, "sha256": digest, "lines": len(lines)}
        return result


# -- fs.glob ----------------------------------------------------------------

class GlobArgs(BaseModel):
    pattern: str = Field(
        description="Glob relative to the workspace root, e.g. 'src/**/*.py'."
    )
    include_hidden: bool = Field(
        default=False, description="Include dotfiles and dot-directories."
    )


class GlobTool(Tool):
    name = "fs.glob"
    description = (
        "List workspace files matching a glob pattern. Honours .gitignore and "
        ".gemma4ignore. Use this to find files before reading them."
    )
    input_model = GlobArgs
    side_effect = SideEffect.READ
    timeout_s = 20.0

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, GlobArgs)
        pattern = args.pattern.strip()
        if not pattern:
            raise ToolError("pattern must not be empty")
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            # A glob is not a path, so the jail cannot resolve it — reject the
            # shapes that would try to escape instead of silently matching nothing.
            raise WorkspaceEscape("glob patterns must stay inside the workspace")

        # An explicitly dotted pattern implies the user wants hidden entries.
        include_hidden = args.include_hidden or any(
            part.startswith(".") for part in Path(pattern).parts
        )
        matches: list[str] = []
        for candidate in ctx.workspace.iter_files(include_hidden=include_hidden):
            rel = ctx.workspace.relative(candidate)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(Path(rel).name, pattern):
                matches.append(rel)
                if len(matches) >= ctx.glob_max_results:
                    break

        if not matches:
            return self.ok(f"no files match {pattern!r}")
        body = "\n".join(sorted(matches))
        if len(matches) >= ctx.glob_max_results:
            body += f"\n... (capped at {ctx.glob_max_results} results; narrow the pattern)"
        result = self.ok(body)
        result.metadata = {"count": len(matches)}
        return result


# -- fs.grep ----------------------------------------------------------------

class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str | None = Field(
        default=None, description="Optional subdirectory or file to limit the search to."
    )
    glob: str | None = Field(default=None, description="Optional filename filter, e.g. '*.py'.")
    ignore_case: bool = Field(default=False)


class GrepTool(Tool):
    name = "fs.grep"
    description = (
        "Search workspace file contents with a regular expression. Returns "
        "file:line:text hits. Skips ignored and binary files."
    )
    input_model = GrepArgs
    side_effect = SideEffect.READ
    timeout_s = 30.0
    path_fields = ("path",)

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, GrepArgs)
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc

        root = ctx.workspace.resolve_in_jail(args.path) if args.path else ctx.workspace.root
        hits = self._ripgrep(args, ctx, root)
        if hits is None:
            hits = self._python_scan(regex, args, ctx, root)

        if not hits:
            return self.ok(f"no matches for {args.pattern!r}")
        capped = len(hits) >= ctx.grep_max_results
        body = "\n".join(hits[: ctx.grep_max_results])
        if capped:
            body += f"\n... (capped at {ctx.grep_max_results} matches; narrow the pattern)"
        result = self.ok(body)
        result.metadata = {"count": len(hits)}
        return result

    def _ripgrep(self, args: GrepArgs, ctx: ToolContext, root: Path) -> list[str] | None:
        """Use ripgrep when it is on PATH; return None to fall back to Python.

        ripgrep is an optimisation, never a requirement (plan §24) — it is not
        added as a system dependency and its absence changes nothing observable.
        """
        binary = shutil.which("rg")
        if not binary:
            return None
        argv = [
            binary, "--line-number", "--no-heading", "--color", "never",
            "--max-count", str(ctx.grep_max_results), "--max-filesize", "1M",
            "--no-follow",
        ]
        if args.ignore_case:
            argv.append("--ignore-case")
        if args.glob:
            argv += ["--glob", args.glob]
        argv += ["--regexp", args.pattern, "--", str(root)]
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell, fixed binary
                argv, capture_output=True, text=True, timeout=20, check=False,
                cwd=str(ctx.workspace.root),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode not in (0, 1):
            return None
        return self._relativise(completed.stdout.splitlines(), ctx)

    def _relativise(self, lines: list[str], ctx: ToolContext) -> list[str]:
        out: list[str] = []
        root = str(ctx.workspace.root) + os.sep
        for line in lines:
            cleaned = line.replace(root, "")
            out.append(cleaned[:_MAX_GREP_LINE_CHARS])
        return out

    def _python_scan(
        self, regex: re.Pattern[str], args: GrepArgs, ctx: ToolContext, root: Path
    ) -> list[str]:
        hits: list[str] = []
        candidates = (
            [root] if root.is_file()
            else [p for p in ctx.workspace.iter_files() if str(p).startswith(str(root))]
        )
        for candidate in candidates:
            if ctx.cancel.is_cancelled():
                break
            if args.glob and not fnmatch.fnmatch(candidate.name, args.glob):
                continue
            try:
                data = ctx.workspace.open_read_bytes(candidate, max_bytes=ctx.max_file_bytes)
            except (ToolError, WorkspaceEscape, OSError):
                continue
            if ctx.workspace.looks_binary(data):
                continue
            rel = ctx.workspace.relative(candidate)
            for number, line in enumerate(
                data.decode("utf-8", errors="replace").splitlines(), start=1
            ):
                if regex.search(line):
                    hits.append(f"{rel}:{number}:{line[:_MAX_GREP_LINE_CHARS]}")
                    if len(hits) >= ctx.grep_max_results:
                        return hits
        return hits


# -- fs.edit ----------------------------------------------------------------

class EditArgs(BaseModel):
    path: str = Field(description="File to edit, relative to the workspace root.")
    old_string: str = Field(
        min_length=1,
        description="Exact text to replace. Must appear EXACTLY ONCE in the file.",
    )
    new_string: str = Field(description="Replacement text.")
    expected_sha256: str = Field(
        description="The sha256 fs.read returned for this file. The edit is refused if it changed."
    )


class EditFileTool(Tool):
    name = "fs.edit"
    description = (
        "Replace a unique block of text in a workspace file. Requires the sha256 "
        "from a previous fs.read of that file; if the file changed since, the "
        "edit is refused and you must re-read it."
    )
    input_model = EditArgs
    side_effect = SideEffect.WORKSPACE_WRITE
    timeout_s = 20.0
    mutating = True
    path_fields = ("path",)

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, EditArgs)
        path = ctx.workspace.resolve_in_jail(args.path)
        rel = ctx.workspace.relative(path)
        if not path.is_file():
            raise ToolError(f"no such file: {rel}")

        original = ctx.workspace.open_read_bytes(path, max_bytes=ctx.max_file_bytes)
        actual = sha256_bytes(original)
        if actual != args.expected_sha256.strip().lower():
            # Intended failure mode: the user edited the file in another editor
            # between the model's read and this apply (plan §13).
            raise PatchConflict(
                f"{rel} changed since it was read (expected sha256 "
                f"{args.expected_sha256[:12]}…, found {actual[:12]}…). "
                "Re-read the file with fs.read and retry the edit."
            )

        if ctx.workspace.looks_binary(original):
            raise ToolError(f"refusing to edit a binary file: {rel}")

        text = original.decode("utf-8")
        occurrences = text.count(args.old_string)
        if occurrences == 0:
            raise PatchConflict(
                f"old_string was not found in {rel}. Copy it verbatim from fs.read output."
            )
        if occurrences > 1:
            raise PatchConflict(
                f"old_string appears {occurrences} times in {rel}; it must be unique. "
                "Include more surrounding context to disambiguate."
            )

        updated = text.replace(args.old_string, args.new_string, 1)
        if ctx.undo_dir is not None:
            UndoStore(Path(ctx.undo_dir), ctx.session_id).snapshot(rel, original)
        self._atomic_write(path, updated.encode("utf-8"))

        new_digest = sha256_bytes(updated.encode("utf-8"))
        ctx.state.record_file_hash(rel, new_digest)
        if path not in ctx.state.files_touched:
            ctx.state.files_touched.append(path)

        diff = _unified_diff(text, updated, rel)
        result = self.ok(
            f"edited {rel}\nnew sha256={new_digest}\n\n{diff}"
        )
        result.metadata = {"path": rel, "sha256": new_digest}
        return result

    def preview(self, args: BaseModel, state: AgentState) -> str:
        """Diff shown in the approval prompt. Best effort; never blocks approval."""
        assert isinstance(args, EditArgs)
        before = args.old_string
        after = args.new_string
        return _unified_diff(before + "\n", after + "\n", args.path)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Temp file in the same directory, then `os.replace` (plan §13.5).

        `O_NOFOLLOW` on the temp file stops a planted symlink from redirecting
        the write; the original file's permission bits are carried over so an
        edit does not silently make a script non-executable.
        """
        tmp = path.with_name(f".{path.name}.gemma4.tmp")
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            mode = 0o644
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ToolError(f"could not write file: {exc.strerror}") from exc


# -- fs.write ---------------------------------------------------------------

class WriteArgs(BaseModel):
    path: str = Field(description="New file to create, relative to the workspace root.")
    content: str = Field(description="Full contents of the new file.")


class WriteFileTool(Tool):
    name = "fs.write"
    description = (
        "Create a NEW file in the workspace. Fails if the path already exists — "
        "use fs.edit to change an existing file."
    )
    input_model = WriteArgs
    side_effect = SideEffect.WORKSPACE_WRITE
    timeout_s = 20.0
    mutating = True
    path_fields = ("path",)

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, WriteArgs)
        path = ctx.workspace.resolve_in_jail(args.path)
        rel = ctx.workspace.relative(path)
        if path.exists():
            # Create-only in MVP (plan §26). Overwrite is what fs.edit's hash
            # check exists to make safe; a convenience overwrite here would
            # discard that entire guarantee.
            raise ToolError(
                f"{rel} already exists. fs.write only creates new files; "
                "use fs.read then fs.edit to change it."
            )
        data = args.content.encode("utf-8")
        if len(data) > ctx.max_file_bytes:
            raise ToolError(f"content exceeds the {ctx.max_file_bytes}-byte limit")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except FileExistsError as exc:
            raise ToolError(f"{rel} already exists") from exc
        except OSError as exc:
            raise ToolError(f"could not create {rel}: {exc.strerror}") from exc

        digest = sha256_bytes(data)
        ctx.state.record_file_hash(rel, digest)
        if path not in ctx.state.files_touched:
            ctx.state.files_touched.append(path)
        result = self.ok(f"created {rel} ({len(data)} bytes)\nsha256={digest}")
        result.metadata = {"path": rel, "sha256": digest, "created": True}
        return result

    def preview(self, args: BaseModel, state: AgentState) -> str:
        assert isinstance(args, WriteArgs)
        head = args.content.splitlines()[:20]
        body = "\n".join(f"+{line}" for line in head)
        if len(args.content.splitlines()) > 20:
            body += "\n+... (truncated preview)"
        return f"--- /dev/null\n+++ {args.path}\n{body}"


def _unified_diff(before: str, after: str, label: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{label}", tofile=f"b/{label}", n=_DIFF_CONTEXT,
    )
    return "".join(diff).rstrip()


def builtin_fs_tools() -> list[Tool]:
    """The MVP filesystem toolbelt, in the order the model sees it."""
    return [ReadFileTool(), GlobTool(), GrepTool(), EditFileTool(), WriteFileTool()]
