"""Connectivity check for PIAST-Q.

Logs in with the token and fetches a backend, then prints backend identity and
configuration. Run this first to confirm the token works.

    python connect.py
"""
from __future__ import annotations

import sys

from piast_common import get_backend, load_token, login


def main() -> int:
    token = load_token()
    if token:
        print(f"Token loaded from .env/env (len={len(token)}). Authenticating...")
    else:
        print("No PIAST_Q_TOKEN found; you will be prompted for the token.")

    try:
        login(token)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Login failed: {type(exc).__name__}: {exc}")
        return 1
    print("[OK] Authenticated.")

    try:
        backend = get_backend(direct=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Could not obtain backend: {type(exc).__name__}: {exc}")
        return 1

    print("[OK] Got direct-access backend.")
    print(f"  backend      : {backend}")
    for attr in ("name", "num_qubits", "version"):
        val = getattr(backend, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:  # noqa: BLE001
                val = "<callable>"
        if val is not None:
            print(f"  {attr:<12}: {val}")

    # Try to surface configuration / target if the AQT backend exposes it.
    for meth in ("configuration", "target"):
        obj = getattr(backend, meth, None)
        if callable(obj):
            try:
                print(f"  {meth}(): {obj()}")
            except Exception:  # noqa: BLE001
                pass

    print("\nConnection check complete. If you reached here, the token works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
