---
name: Quality Check
description: Verify code quality with ruff (lint/format) and mypy (types) before committing/pushing.
---

# Quality Check Skill

This skill ensures that the codebase meets quality standards before pushing changes. It runs:
1.  **Ruff**: Linter and Formatter to fix style issues and bugs.
2.  **Mypy**: Static type checker to catch type errors.

## Usage

Always run this skill before pushing to GitHub to ensure CI/CD will pass.

## Steps

### 1. Run Ruff (Lint & Format)

Run Ruff to auto-fix linting issues and format code.

```powershell
.venv\Scripts\activate
ruff check . --fix
ruff format .
```

### 2. Run Mypy (Type Check)

Run MyPy to verify type safety.

```powershell
.venv\Scripts\activate
mypy src/
```

### 3. Verify Output

-   **Ruff**: Should report "All checks passed!" or "Found X errors (fixed)".
-   **Mypy**: Should report "Success: no issues found in X source files".

If any errors remain, **FIX THEM** before pushing.
