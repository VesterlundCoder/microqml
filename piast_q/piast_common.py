"""Shared helpers for connecting to the PCSS PIAST-Q trapped-ion quantum computer.

Access is via the alpha `pcss-qapi[aqt]` library (Qiskit AQT Provider backend).
The token is read from the environment (loaded from a gitignored `.env` file) so
it is never hardcoded or committed. It can also be entered interactively.

Machine constraints (per the PCSS onboarding email):
- 200 shots per circuit execution (SHOTS_MAX).
- ~250 jobs per hour; keep total usage to a few hours per week.
- Best quality: Mon 13:00-17:00 CEST, Tue-Thu 10:00-17:00 CEST.
- Fri 09:00-17:00 CEST is a technical break.
"""
from __future__ import annotations

import os
from pathlib import Path

SHOTS_MAX = 200  # hard limit enforced by PIAST-Q per circuit execution.

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_token() -> str | None:
    """Return the PIAST-Q token from `.env` / environment, or None if unset."""
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
    except Exception:
        pass
    token = os.environ.get("PIAST_Q_TOKEN")
    if token and token.strip() and token.strip() != "paste-your-token-here":
        return token.strip()
    return None


def login(token: str | None = None) -> None:
    """Authenticate to PIAST-Q.

    If ``token`` is None we try the environment; failing that, the underlying
    ``AuthorizationService.login`` will prompt interactively in the terminal.
    """
    from pcss_qapi import AuthorizationService

    token = token or load_token()
    # login(None) triggers the library's interactive prompt.
    AuthorizationService.login(token)


def get_backend(direct: bool = True):
    """Return a PIAST-Q backend. Requires a prior successful ``login``."""
    from pcss_qapi.aqt.provider import PCSS_AQTProvider

    provider = PCSS_AQTProvider()
    if direct:
        return provider.get_direct_access_backend()
    return provider.get_backend()
