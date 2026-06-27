#!/usr/bin/env bash
# setup_env.command — DOUBLE-CLICK this file in Finder to set up the project environment.
# (macOS opens Terminal and runs it.) Safe to run more than once.
set -euo pipefail
cd "$(dirname "$0")"                    # run from the repo folder, wherever it was launched from

# Keep the Terminal window open at the end (on success OR error) so the output stays readable.
trap 'echo; read -n 1 -s -r -p "Press any key to close this window..."; echo' EXIT

PY=.venv/bin/python

# 1) Create .venv if it does not exist yet (uses the system Python 3).
if [ ! -x "$PY" ]; then
  echo "[1/4] Creating .venv ..."; python3 -m venv .venv
else
  echo "[1/4] .venv already exists ($("$PY" -V))."
fi

# 2) pip is old (21.x) -> upgrade it so it understands modern torch wheels.
echo "[2/4] Upgrading pip ..."
"$PY" -m pip install --upgrade pip --quiet

# 3) Install exactly what requirements.txt declares (incl. torch; arm64 -> MPS build).
echo "[3/4] Installing libraries from requirements.txt ..."
"$PY" -m pip install -r requirements.txt

# 4) Verify: imports succeed, print versions, run one torch op.
echo "[4/4] Verifying:"
"$PY" - <<'CHECK'
import numpy, sklearn, pandas, torch
print(f"     numpy {numpy.__version__} | sklearn {sklearn.__version__} | pandas {pandas.__version__}")
print(f"     torch {torch.__version__}")
print(f"     torch matmul ok: {tuple((torch.randn(3,3) @ torch.randn(3,3)).shape)}")
print(f"     MPS (Mac GPU) available: {torch.backends.mps.is_available()}")
CHECK
echo "Done. Environment is ready."
