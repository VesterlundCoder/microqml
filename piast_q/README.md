# PIAST-Q access (PCSS trapped-ion quantum computer)

Client setup for running circuits on **PIAST-Q** via the alpha `pcss-qapi[aqt]`
library (Qiskit AQT Provider backend). Built for the CMF quantum-search project
(`../QUANTUM_CMF_ALGORITHM.md`, QPU target "Piast-Q").

## Why a dedicated venv

`pcss-qapi` requires **Python >= 3.11** and `qiskit-aqt-provider` requires
**< 3.14**, so the workspace `cy-env` (3.10) and brew `python3.14` both fail.
This project uses **Python 3.12.13** (pyenv) in a local `.venv`.

Installed (see `requirements.txt`): `pcss-qapi 0.2.2`, `qiskit 1.4.6`,
`qiskit-aqt-provider 1.14.0` (the email's "version for authorization: 1.14"),
`qiskit-aer`, `aqt-connector`, `auth0-python`.

## One-time setup

```bash
# Recreate the venv if needed:
~/.pyenv/versions/3.12.13/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Token

The token arrives in a separate PCSS email. Store it in a gitignored `.env`:

```bash
cp .env.example .env
# then edit .env and paste the token into PIAST_Q_TOKEN=...
```

Alternatively, leave `.env` unset and the scripts will prompt for the token in
the terminal (the library caches it after first login).

## Usage

```bash
# 1. Confirm the token works and inspect the backend:
.venv/bin/python connect.py

# 2. Run the canonical Bell-state job (200 shots):
.venv/bin/python bell_test.py
```

## PIAST-Q operating constraints (from the PCSS onboarding email)

- **200 shots** per circuit execution (`SHOTS_MAX`). Larger statistics = repeat
  and aggregate.
- ~**250 jobs/hour**; keep total usage to **a few hours per week** (no week-long runs).
- **Best quality windows (CEST):** Mon 13:00-17:00; Tue-Thu 10:00-17:00.
- **Fri 09:00-17:00 CEST**: technical break. Off-window results may vary in quality.
- Jobs sent during calibration are queued and returned once the system is back.
- Publishing results requires an acknowledgment formula (email PCSS for the
  grant/project number).

## Files

- `piast_common.py` - shared `login()` / `get_backend()` helpers + `SHOTS_MAX`.
- `connect.py` - connectivity + backend-info check.
- `bell_test.py` - Bell-state smoke test (200 shots).
- `.env.example` - token template (copy to `.env`).
