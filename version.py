"""
version.py — Single source of truth for the project version string.

Versioning scheme:
  Pre-paper-trading:  0.BREAKING.FEATURE.BUILD
  Post-paper-trading: MAJOR.FEATURE.BUILD  (starts at 1.0.0)

  BREAKING — incremented when a change loses backward compatibility
              (e.g. model file format change, state.json schema change);
              resets all trailing digits to 0.
  FEATURE   — incremented for any new capability or significant improvement;
              resets BUILD to 0.
  BUILD     — incremented for bug fixes and minor changes within a FEATURE.
  0.        — prefix present until paper trading performance is deemed acceptable;
              replaced by 1. on first production promotion.
"""

VERSION = "0.1.0.4"
