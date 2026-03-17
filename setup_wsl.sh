#!/bin/bash
set -euo pipefail

# ==========================================================
#  APD Intelligibility Estimator - WSL Environment Setup
#
#  Usage:
#    wsl bash setup_wsl.sh
#
#  What this does:
#    1. Install system packages (build tools, audio libs)
#    2. Install Python 3.10 + venv
#    3. Create virtualenv, install all Python deps (including pesq)
#    4. Verify everything works
#    5. Print next steps
# ==========================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step() { echo -e "\n${GREEN}=== $1 ===${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------
step "1/6  System packages"
# ----------------------------------------------------------
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    gcc g++ \
    python3 python3-dev python3-venv python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    > /dev/null 2>&1

log "System packages installed"

# Python version check
PYTHON=python3
PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 9) )); then
    fail "Python >= 3.9 required, found $PY_VER"
fi
log "Python $PY_VER"

# ----------------------------------------------------------
step "2/6  Virtual environment"
# ----------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    warn "Existing venv found, reusing: $VENV_DIR"
else
    $PYTHON -m venv "$VENV_DIR"
    log "Created venv: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
log "venv activated, pip upgraded"

# ----------------------------------------------------------
step "3/6  PyTorch (CUDA)"
# ----------------------------------------------------------
# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log "NVIDIA driver detected: $CUDA_VER"

    # Install PyTorch with CUDA support
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1
    log "PyTorch + CUDA installed"
else
    warn "No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q 2>&1 | tail -1
    log "PyTorch (CPU) installed"
fi

# ----------------------------------------------------------
step "4/6  Python dependencies"
# ----------------------------------------------------------
pip install numpy scipy soundfile pystoi pyroomacoustics -q 2>&1 | tail -1
log "Core deps installed"

# pesq needs C compiler - should work on WSL/Linux
pip install pesq -q 2>&1 | tail -1 && log "pesq installed" || warn "pesq build failed (STOI fallback will be used)"

# ----------------------------------------------------------
step "5/6  Verify imports"
# ----------------------------------------------------------
$PYTHON -c "
import sys
ok = True
pkgs = ['torch', 'torchaudio', 'numpy', 'scipy', 'soundfile', 'pystoi', 'pyroomacoustics']
for p in pkgs:
    try:
        __import__(p)
        print(f'  {p:25s} OK')
    except ImportError:
        print(f'  {p:25s} MISSING')
        ok = False

# optional
try:
    __import__('pesq')
    print(f'  {\"pesq\":25s} OK')
except ImportError:
    print(f'  {\"pesq\":25s} MISSING (optional)')

# CUDA check
import torch
if torch.cuda.is_available():
    print(f'  {\"CUDA\":25s} {torch.cuda.get_device_name(0)}')
else:
    print(f'  {\"CUDA\":25s} NOT AVAILABLE')

if not ok:
    sys.exit(1)
"
log "All imports verified"

# ----------------------------------------------------------
step "6/6  Quick model test"
# ----------------------------------------------------------
$PYTHON -c "
import sys, torch
sys.path.insert(0, '.')
from model.model_definition import create_model
from training.export_apd import export_apd, validate_export
from training.loss import APDLoss

model = create_model(overparameterized=True)
x = torch.randn(2, 1, 16000)
out = model(x)
print(f'  Model params:  {sum(p.numel() for p in model.parameters()):,}')
print(f'  Forward pass:  {out.squeeze().tolist()}')

criterion = APDLoss()
loss, _ = criterion(out.squeeze(), torch.rand(2))
print(f'  Loss:          {loss.item():.4f}')
print(f'  All checks passed')
"
log "Model test passed"

# ----------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Activate venv:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run full pipeline:"
echo "    python run_pipeline.py"
echo ""
echo "  Quick test (small dataset):"
echo "    python run_pipeline.py --small"
echo ""
echo "  Download data only:"
echo "    python download_data.py"
echo ""
