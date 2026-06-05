# Fixing graphify git hooks on isolated installs (uv tool / pipx), Windows-prone

**Applies to:** any repository where `graphify hook install` was run **and** graphify
is installed in an **isolated environment** (`uv tool install graphifyy` or
`pipx install graphifyy`). The bug is most visible on Windows but the root cause is
the isolation, not the OS.

---

## TL;DR

The `post-commit` / `post-checkout` graphify hooks try to find a Python that can
`import graphify`. With an isolated install, graphify lives in its own venv and is
**not** importable from the `python` / `python3` on `PATH`, so the hook silently
`exit 0`s and the graph is never rebuilt. The fix makes the hook **prefer the
interpreter graphify already recorded** in `graphify-out/.graphify_python` (the
isolated venv's python), which can always import graphify.

---

## Symptoms

- Every `git commit` prints: `warning: command substitution: ignored null byte in input`
  (from the hook reading a binary launcher).
- `graphify-out/graph.json` never updates after commits (stale mtime).
- No rebuild log at `~/.cache/graphify-rebuild.log`.

## Quick diagnosis

Run these from the repo root (Git Bash / sh):

```sh
# 1. Hooks installed?
graphify hook status

# 2. Does a bare python on PATH have graphify? (the hook's fallback)
python  -c "import graphify" 2>&1 || echo "python: NO graphify"
python3 -c "import graphify" 2>&1 || echo "python3: NO graphify"

# 3. Is the graph stale relative to your last commit?
ls -l graphify-out/graph.json      # compare mtime to: git log -1 --format=%cd
```

If step 2 reports **NO graphify** but the `graphify` CLI works, you have this bug.

## Root cause

`uv tool install` / `pipx install` create an **isolated virtual environment** per
tool (e.g. `…/uv/tools/graphifyy/Scripts/python.exe`). graphify is importable only
from that venv. The launcher placed on `PATH` (`~/.local/bin/graphify[.exe]`) is an
entry-point, **not** an interpreter you can `import` from.

The stock hook tries, in order:

1. Read the shebang of the `graphify` launcher to extract its venv python — on
   Windows the launcher is a compiled `.exe` with no readable shebang (hence the
   *null byte* warning), so this yields nothing.
2. Fall back to `python` / `python3` on `PATH` — the **system** interpreter, which
   has no graphify → `ModuleNotFoundError`.
3. No interpreter found → `exit 0` (silent no-op).

graphify itself works fine — only the hook's interpreter heuristic fails, because it
doesn't look inside the isolated venv. The `/graphify` pipeline already records the
correct interpreter in `graphify-out/.graphify_python`; the fix just reads it first.

---

## Fix — automated (recommended)

Save this as `patch_graphify_hooks.py` and run it once **per repository** with any
Python (it only does text replacement — it does **not** need graphify):

```python
#!/usr/bin/env python3
"""Patch graphify git hooks to prefer the recorded interpreter.

Idempotent. Run from anywhere inside the target git repo:
    python patch_graphify_hooks.py
"""
import subprocess
import sys
from pathlib import Path

OLD = '''# Detect the correct Python interpreter (handles pipx, venv, system installs)
GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
if [ -n "$GRAPHIFY_BIN" ]; then
    case "$GRAPHIFY_BIN" in
        *.exe) _SHEBANG="" ;;
        *)     _SHEBANG=$(head -1 "$GRAPHIFY_BIN" | sed 's/^#![[:space:]]*//') ;;
    esac
    case "$_SHEBANG" in
        */env\\ *) GRAPHIFY_PYTHON="${_SHEBANG#*/env }" ;;
        *)         GRAPHIFY_PYTHON="$_SHEBANG" ;;
    esac
    # Allowlist: only keep characters valid in a filesystem path to prevent
    # injection if the shebang contains shell metacharacters
    case "$GRAPHIFY_PYTHON" in
        *[!a-zA-Z0-9/_.@-]*) GRAPHIFY_PYTHON="" ;;
    esac
    if [ -n "$GRAPHIFY_PYTHON" ] && ! "$GRAPHIFY_PYTHON" -c "import graphify" 2>/dev/null; then
        GRAPHIFY_PYTHON=""
    fi
fi'''

NEW = '''# Prefer the interpreter recorded by the /graphify pipeline — handles
# uv-tool / pipx isolated installs where graphify is NOT importable from a
# bare python/python3 on PATH. Reading it first also avoids head-reading a
# non-.exe binary shim (the "ignored null byte" warning).
GRAPHIFY_PYTHON=""
_GRAPHIFY_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -n "$_GRAPHIFY_ROOT" ] && [ -f "$_GRAPHIFY_ROOT/graphify-out/.graphify_python" ]; then
    _REC=$(tr -d '\\r\\n' < "$_GRAPHIFY_ROOT/graphify-out/.graphify_python")
    if [ -n "$_REC" ] && "$_REC" -c "import graphify" 2>/dev/null; then
        GRAPHIFY_PYTHON="$_REC"
    fi
fi

# Detect the correct Python interpreter (handles pipx, venv, system installs)
if [ -z "$GRAPHIFY_PYTHON" ]; then
    GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
    if [ -n "$GRAPHIFY_BIN" ]; then
        case "$GRAPHIFY_BIN" in
            *.exe) _SHEBANG="" ;;
            *)     _SHEBANG=$(head -1 "$GRAPHIFY_BIN" | sed 's/^#![[:space:]]*//') ;;
        esac
        case "$_SHEBANG" in
            */env\\ *) GRAPHIFY_PYTHON="${_SHEBANG#*/env }" ;;
            *)         GRAPHIFY_PYTHON="$_SHEBANG" ;;
        esac
        # Allowlist: only keep characters valid in a filesystem path to prevent
        # injection if the shebang contains shell metacharacters
        case "$GRAPHIFY_PYTHON" in
            *[!a-zA-Z0-9/_.@-]*) GRAPHIFY_PYTHON="" ;;
        esac
        if [ -n "$GRAPHIFY_PYTHON" ] && ! "$GRAPHIFY_PYTHON" -c "import graphify" 2>/dev/null; then
            GRAPHIFY_PYTHON=""
        fi
    fi
fi'''

MARKER = "Prefer the interpreter recorded by the /graphify pipeline"


def hooks_dir() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def main() -> int:
    hooks = hooks_dir()
    patched, skipped, missing = [], [], []
    for name in ("post-commit", "post-checkout"):
        hook = hooks / name
        if not hook.exists():
            missing.append(name)
            continue
        text = hook.read_text(encoding="utf-8")
        if "graphify" not in text:
            skipped.append(f"{name} (not a graphify hook)")
            continue
        if MARKER in text:
            skipped.append(f"{name} (already patched)")
            continue
        if OLD not in text:
            skipped.append(f"{name} (detection block not found — graphify version differs; patch manually)")
            continue
        hook.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        patched.append(name)

    for n in patched:
        print(f"patched: {n}")
    for s in skipped:
        print(f"skipped: {s}")
    for m in missing:
        print(f"absent:  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run:

```sh
python patch_graphify_hooks.py
```

It is idempotent (re-running is a no-op) and only touches hooks that carry the exact
stock detection block. If it reports *"detection block not found"*, your graphify
version emits a different hook — apply the **manual** fix below.

---

## Fix — manual

In **both** `.git/hooks/post-commit` and `.git/hooks/post-checkout`, find this block:

```sh
# Detect the correct Python interpreter (handles pipx, venv, system installs)
GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
if [ -n "$GRAPHIFY_BIN" ]; then
    ...
fi
```

Replace it with: a *recorded-interpreter* preamble, then the **same** block wrapped
in `if [ -z "$GRAPHIFY_PYTHON" ]; then ... fi`:

```sh
# Prefer the interpreter recorded by the /graphify pipeline.
GRAPHIFY_PYTHON=""
_GRAPHIFY_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -n "$_GRAPHIFY_ROOT" ] && [ -f "$_GRAPHIFY_ROOT/graphify-out/.graphify_python" ]; then
    _REC=$(tr -d '\r\n' < "$_GRAPHIFY_ROOT/graphify-out/.graphify_python")
    if [ -n "$_REC" ] && "$_REC" -c "import graphify" 2>/dev/null; then
        GRAPHIFY_PYTHON="$_REC"
    fi
fi

# Detect the correct Python interpreter (fallback, only if not already found)
if [ -z "$GRAPHIFY_PYTHON" ]; then
    GRAPHIFY_BIN=$(command -v graphify 2>/dev/null)
    if [ -n "$GRAPHIFY_BIN" ]; then
        # ... original detection body, unchanged, indented one extra level ...
    fi
fi
```

The two essential changes: (1) set `GRAPHIFY_PYTHON` from
`graphify-out/.graphify_python` first; (2) guard the original detection with
`[ -z "$GRAPHIFY_PYTHON" ]` so it only runs as a fallback (this also skips the
binary-shim read that caused the null-byte warning).

> Requires that `graphify-out/.graphify_python` exists — it is written by the
> `/graphify` pipeline. If absent, run a graph build once (`graphify update .` or
> the `/graphify` skill) to create it, or set `GRAPHIFY_PYTHON` to the venv python
> directly (e.g. `…/uv/tools/graphifyy/Scripts/python.exe`).

---

## Verify

```sh
# 1. No null-byte warning + a rebuild is launched (post-commit, invoked as git does):
sh .git/hooks/post-commit

# 2. The background rebuild actually ran:
tail ~/.cache/graphify-rebuild.log      # expect: "Rebuilt: N nodes, M edges ..."

# 3. The graph is now fresh:
ls -l graphify-out/graph.json           # mtime should be ~now

# 4. post-checkout (real branch switch):
git switch -c _hooktest && git switch - && git branch -D _hooktest
tail ~/.cache/graphify-rebuild.log      # expect: "Branch switched - launching ..."
```

---

## Caveats

- **Hooks are not version-controlled.** `.git/hooks/` is local to each clone, so this
  patch must be applied **per repository, per machine**. It is not shared by pushing.
- **`graphify hook install` does *not* overwrite an existing hook** — it is a no-op
  when a graphify hook is already present (`already installed`), so it will not clobber
  your patch. However, `graphify hook uninstall` followed by `graphify hook install`,
  or a fresh clone, regenerates **stock** hooks; re-run `patch_graphify_hooks.py` then
  (it's idempotent and safe to keep around).
- **Depends on `graphify-out/.graphify_python`.** Keep it present (the `/graphify`
  pipeline maintains it). If you wipe `graphify-out/`, rebuild once to regenerate it.
- **Package vs module name:** the tool is installed as `graphifyy` (PyPI/uv name) but
  imported as `graphify`. The patch checks importability of `graphify`, which is
  correct.

## Upstream note

This is a graphify-side limitation: the installed hook's interpreter heuristic does
not consult `graphify-out/.graphify_python` and assumes a non-isolated install with a
readable launcher shebang. Preferring the recorded interpreter out of the box would
fix it for all uv-tool / pipx / Windows users. Worth reporting upstream.
