# Stokes Multipliers of x^{2M} Anharmonic Oscillators

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.31796332-blue)](https://doi.org/10.6084/m9.figshare.31796332.v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code for the paper:

> **J. Zhou**, "Stokes Multipliers of x^{2M} Anharmonic Oscillators: Exact Results for M = 2–11" (March 2026).  
> Figshare preprint: [doi:10.6084/m9.figshare.31796332.v1](https://doi.org/10.6084/m9.figshare.31796332.v1)

## Overview

This repository contains Python code for computing **Stokes multipliers** of the one-dimensional anharmonic oscillators

$$H = \frac{p^2}{2} + \frac{x^2}{2} + g\,x^{2M}$$

using high-precision perturbation theory, Richardson extrapolation, and the PSLQ integer relation algorithm.

### Key Results

| M | \|C_M\| | Closed Form | φ(M−1)/2 |
|---|---------|-------------|----------|
| 2 | 0.43989 68135 81545 | √(6/π³) | 0 |
| 3 | 0.57315 91682 50756 | 4√2/π² | 0 |
| 4 | 0.74005 14982 59358 506… | **None found** (30 digits) | 1 |
| 5 | 0.91376 01170 24928 | 8√3/(Γ(1/4)·π^{5/4}) | 1 |
| 6 | 1.08996 63616 43951 | — | 2 |
| 7 | 1.26735 98646 78474 | C⁶·Γ(1/3)⁹·π⁶ = 2²⁰·3³ | 1 |
| 8 | 1.44540 95063 62104 | — | 3 |
| 9 | 1.62385 95507 60926 | — | 2 |
| 10 | 1.80257 17672 30778 | — | 3 |
| 11 | 1.98146 49381 17015 | — | 2 |

**New exact results:** C₅ and C₇ closed forms discovered via PSLQ.  
**Universal formula:** Instanton action A_M = S₀^{M−1} with S₀ = 2^{−1/(M−1)} · B(1/(M−1), 3/2) / (M−1).

## Repository Structure

```
stokes-multipliers/
├── README.md              # This file
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── src/
│   ├── bender_wu.py       # Rayleigh-Schrödinger perturbation theory recursion
│   ├── richardson.py      # Richardson extrapolation & parameter extraction
│   ├── pslq_search.py     # PSLQ closed-form search routines
│   ├── borel_pade.py      # Borel-Padé summation (auxiliary)
│   └── utils.py           # Utility functions
├── scripts/
│   ├── compute_all.py     # Reproduce all results from the paper
│   ├── compute_single.py  # Compute C_M for a single M value
│   └── verify_closed.py   # Verify all closed-form identities
└── data/
    └── results.json       # Pre-computed numerical results
```

## Installation

```bash
git clone https://github.com/JackZH26/stokes-multipliers.git
cd stokes-multipliers
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- [mpmath](http://mpmath.org/) ≥ 1.3 (arbitrary-precision arithmetic)

## Usage

### Quick Start: Verify All Closed Forms

```bash
python scripts/verify_closed.py
```

This verifies the four exact identities:
- C₂² · π³ = 6
- C₃² · π⁴ = 32
- C₅⁴ · Γ(1/4)⁴ · π⁵ = 36864
- C₇⁶ · Γ(1/3)⁹ · π⁶ = 28311552

### Compute C_M for a Single M

```bash
# M=5, 300 orders, 150-digit precision
python scripts/compute_single.py --M 5 --orders 300 --dps 150
```

### Reproduce All Paper Results

```bash
# Warning: M=4 at full precision takes ~1 hour and ~1 GB RAM
python scripts/compute_all.py
```

### Use as a Library

```python
from src.bender_wu import compute_coefficients
from src.richardson import richardson_extrapolate
from mpmath import mp, mpf, gamma, sqrt, pi

mp.dps = 200

# Compute 500 perturbation coefficients for the quartic oscillator (M=2)
coeffs = compute_coefficients(M=2, N_max=500, dps=200)

# Extract Stokes multiplier using known exact A and b
A = mpf(1) / 3
C_seq = []
for k in range(20, len(coeffs)):
    Ck = coeffs[k] * (-A)**k / gamma(k + mpf('0.5'))
    C_seq.append(Ck)

C = richardson_extrapolate(C_seq[50:], N_terms=40)
print(f"|C_2| = {mp.nstr(abs(C), 25)}")
print(f"sqrt(6/pi^3) = {mp.nstr(sqrt(6/pi**3), 25)}")
```

## Method

### 1. Perturbation Coefficients (`bender_wu.py`)

Computes the Rayleigh-Schrödinger perturbation series E(g) = Σ aₖ gᵏ by recursively building the perturbed wave function in the harmonic oscillator basis. The perturbation V = x^{2M} is applied by 2M successive applications of the position operator x = (a + a†)/√2.

**Complexity:** O(M²k²) per order k, with vectors of dimension O(Mk).

### 2. Richardson Extrapolation (`richardson.py`)

Extracts the instanton action A_M, the Gamma-shift parameter b, and the Stokes multiplier C_M from the large-order behavior:

$$a_k \sim C_M \cdot (-1)^{k+1} \cdot A_M^{-k} \cdot \Gamma((M{-}1)k + \tfrac{1}{2})$$

Richardson extrapolation of order N eliminates the first N subleading corrections, dramatically accelerating convergence.

### 3. PSLQ Search (`pslq_search.py`)

Searches for integer relations among {ln|C_M|, ln Γ(1/(M−1)), ln π, ln 2, ln 3, 1} using the PSLQ algorithm from mpmath. The basis is chosen based on the number of independent Gamma transcendentals, determined by φ(M−1)/2.

## Computational Resources

| M | Orders | Precision (dps) | Time | RAM | C_M digits |
|---|--------|----------------|------|-----|------------|
| 2 | 500 | 200 | ~1 min | 50 MB | 24 |
| 3 | 500 | 200 | ~2 min | 80 MB | 20 |
| 4 | 1200 | 300 | ~55 min | 1 GB | 30 |
| 5 | 600 | 200 | ~5 min | 150 MB | 20 |
| 7 | 400 | 180 | ~8 min | 200 MB | 22 |
| 11 | 250 | 160 | ~10 min | 300 MB | 20 |

Benchmarked on a single-core VPS (x86_64, Python 3.11, mpmath 1.3).

## Citation

If you use this code, please cite:

```bibtex
@article{Zhou2026stokes,
  author  = {Zhou, Jian},
  title   = {Stokes Multipliers of $x^{2M}$ Anharmonic Oscillators: Exact Results for $M = 2$--$11$},
  year    = {2026},
  month   = {March},
  doi     = {10.6084/m9.figshare.31796332.v1},
  url     = {https://doi.org/10.6084/m9.figshare.31796332.v1},
  note    = {Preprint on Figshare}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
