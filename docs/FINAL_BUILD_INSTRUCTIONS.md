# FINAL BUILD & OPERATIONAL INSTRUCTIONS — QUANT_VVF

**Target Platforms:** Windows 10/11 x86_64, Linux (Ubuntu 22.04 LTS / RHEL 9)  
**Native Toolchains Required:**
- Python $\ge$ 3.11.0 (with pip and virtualenv)
- Rust Toolchain $\ge$ 1.100.0 (`cargo`, `rustc`, `rustfmt`, `clippy`)
- C++20 Compiler (Clang++ $\ge$ 17.0 or GCC $\ge$ 13.0 supporting `-std=c++20`)

---

## 1. Environment Setup

### Windows Toolchain Configuration
Ensure Clang, GCC (for Mingw runtime), and Cargo are on the active PATH:
```powershell
$clangPath = (Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "clang++.exe" | Where-Object { $_.FullName -like "*llvm-mingw*\bin*" }).DirectoryName
$gccPath = (Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "gcc.exe" | Where-Object { $_.FullName -like "*mingw64\bin*" }).DirectoryName
$env:PATH = "$clangPath;$gccPath;$env:USERPROFILE\.cargo\bin;" + $env:PATH
```

### Python Dependencies
Install required numerical and financial modeling packages:
```bash
python -m pip install -r requirements.txt
python -m pip install scikit-learn
```

---

## 2. Compiling & Executing Native Test Suites

### 2.1 C++20 Low-Latency Execution & Pre-Trade Risk Engine
Compile and execute the seventh remediation test suite (including durable WAL recovery):
```powershell
cd hft/tests
clang++ -std=c++20 -O2 -I../include test_seventh_remediation.cpp -o test_seventh.exe
.\test_seventh.exe
```

Compile and execute the fifth remediation regression suite (order collision and ledger invariants):
```powershell
cd hft/tests
clang++ -std=c++20 -O2 -I../include test_fifth_remediation.cpp -o test_fifth.exe
.\test_fifth.exe
```

### 2.2 Rust 2024 Starbase SBE Ingestion Gateway
Run all unit tests, format checks, and strict linter audits:
```powershell
cd rust_hft
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
```

### 2.3 Python Quantitative Research & Integration Test Suite
Execute the complete automated test suite (146 tests covering P1–P5, Options, Backtesting, Portfolio, JARVIS, and Adapters):
```powershell
pytest tests/ -v
```

---

## 3. Operational Modes & Safety Guards

The platform provides multiple isolated execution modes configured via environment variables:

| Operating Mode | Configuration Variable | Behavior & Gatekeeping |
| :--- | :--- | :--- |
| **Paper Trading (Default)** | Default mode | Matches orders against simulated local liquidity book. Zero risk of live loss. |
| **Testnet Mode** | `DERIBIT_USE_TESTNET=TRUE` | Connects to `test.deribit.com`. Requires testnet API keys. |
| **Live Production Mode** | `DERIBIT_LIVE_AUTHORIZED=TRUE` | **Physically locked.** Requires explicit operator token. Fails closed if missing. |

---

## 4. Launching the Trading Terminal Frontend

To launch the local quantitative trading dashboard:
```bash
# From repository root
cd frontend
# Open in default web browser
start index.html   # On Windows
xdg-open index.html # On Linux
```
