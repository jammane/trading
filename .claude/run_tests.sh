#!/usr/bin/env bash
# Run pytest after git commit / push.
# Output is captured by Claude Code and fed back into the conversation,
# so test failures are visible to Claude immediately and can be self-corrected.

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$REPO_DIR"

# Locate pytest (prefer venv binary, fall back to system)
PYTEST_CMD=""
if [ -f "$REPO_DIR/.venv/bin/pytest" ]; then
    PYTEST_CMD="$REPO_DIR/.venv/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then
    PYTEST_CMD="pytest"
else
    exit 0   # pytest not installed; skip silently
fi

echo ""
echo "══════════════════════ pytest ══════════════════════"
"$PYTEST_CMD" tests/ -v --tb=short 2>&1
PYTEST_EXIT=$?
echo "════════════════════════════════════════════════════"

if [ $PYTEST_EXIT -ne 0 ]; then
    echo ""
    echo "PYTEST_FAILED — fix the failures above before pushing."
fi

exit 0   # always exit 0 so the hook does not block the commit
