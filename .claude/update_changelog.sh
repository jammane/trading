#!/usr/bin/env bash
# Auto-update CHANGELOG.md after each git commit made by Claude.
# Prepends a new entry, then amends the commit to include it.
set -e

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
CHANGELOG="$REPO_DIR/CHANGELOG.md"

# Bail out if CHANGELOG doesn't exist yet
[ -f "$CHANGELOG" ] || exit 0

# Get latest commit details
HASH=$(git -C "$REPO_DIR" log -1 --format="%h")
DATE=$(git -C "$REPO_DIR" log -1 --format="%ci" | cut -d' ' -f1)
MSG=$(git -C "$REPO_DIR" log -1 --format="%s")

# Skip if this hash is already recorded (prevents double-logging on repeated amends)
if grep -qF "[$HASH]" "$CHANGELOG" 2>/dev/null; then
    exit 0
fi

# Prepend the new entry immediately after the first --- separator line
TMPFILE=$(mktemp)
awk -v hash="$HASH" -v date="$DATE" -v msg="$MSG" '
    /^---$/ && !done {
        print
        print ""
        print "## [" hash "] \342\200\224 " date
        print msg
        done=1
        next
    }
    { print }
' "$CHANGELOG" > "$TMPFILE"
mv "$TMPFILE" "$CHANGELOG"

# Stage and amend the current commit to include the changelog update
git -C "$REPO_DIR" add "$CHANGELOG"
git -C "$REPO_DIR" commit --amend --no-edit
