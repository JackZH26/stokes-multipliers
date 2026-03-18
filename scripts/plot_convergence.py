#!/usr/bin/env python3
"""Generate convergence plot for M=4 Stokes multiplier extraction."""
from mpmath import mp, mpf, gamma, sqrt, pi, log10, fabs, binomial, factorial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mp.dps = 50

# Load M=4 coefficients
print("Loading M=4 coefficients...")
coeffs = []
fname = '../data/coefficients/M4_coeffs_400.txt'
with open(fname) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                coeffs.append(mpf(parts[1]))
print(f"Loaded {len(coeffs)} coefficients from {fname}")

M = 4
p = M - 1
S0 = (mpf(1)/2)**(mpf(1)/p) * gamma(mpf(1)/p) * sqrt(pi) / (2*p*gamma(mpf(1)/p + mpf(3)/2))
A = S0**p
C_best = mpf('-0.74005149825935850640151149162202')
print(f"A_4 = {mp.nstr(A, 25)}")

# Build C_k sequence
C_raw_mp = []
k_values = []
for k in range(15, len(coeffs)):
    gval = gamma(p * mpf(k) + mpf('0.5'))
    Ck = coeffs[k] * ((-A)**k) / gval
    C_raw_mp.append(Ck)
    k_values.append(k)

C_raw_float = [float(c) for c in C_raw_mp]
print(f"C_k sequence: {len(C_raw_mp)} values (k={k_values[0]}..{k_values[-1]})")

# Richardson extrapolation
def richardson(seq_mp, start_idx, N):
    """Richardson extrapolation of order N from start_idx in seq_mp."""
    mp.dps = 50
    result = mpf(0)
    for j in range(N+1):
        idx = start_idx + j
        if idx >= len(seq_mp):
            return None
        k = idx + 15  # actual k
        bcoeff = binomial(N, j)
        sign = mpf(-1)**(N - j)
        result += sign * bcoeff * mpf(k)**N * seq_mp[idx]
    result /= factorial(N)
    return result

def digits_agree(val, ref):
    if val is None:
        return 0
    diff = fabs(val - ref)
    if diff == 0:
        return 30
    return max(0, float(-log10(fabs(diff / ref))))

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1, 1]})

# Upper: raw C_k sequence
ax1.plot(k_values[::2], C_raw_float[::2], 'b.', markersize=1.5, alpha=0.6, label=r'$\widetilde{C}_k$ (raw)')
ax1.axhline(y=float(C_best), color='r', linestyle='--', linewidth=1.0, 
            label=r'Best estimate $C_4$')
ax1.set_xlabel('Order $k$', fontsize=12)
ax1.set_ylabel(r'$\widetilde{C}_k$', fontsize=12)
ax1.set_title(r'Convergence of Stokes multiplier extraction: $M = 4$', fontsize=13)
ax1.legend(fontsize=10, loc='upper right')
yc = float(C_best)
ax1.set_ylim([yc - 0.015, yc + 0.015])
ax1.grid(True, alpha=0.3)

# Lower: digits of precision vs Richardson order for two starting points
for k0_actual, color, marker, label_prefix in [
    (200, 'green', 'o', r'$k_0 = 200$'),
    (150, 'blue', 's', r'$k_0 = 150$'),
    (100, 'orange', '^', r'$k_0 = 100$'),
]:
    k0_idx = k0_actual - 15  # index in C_raw_mp
    if k0_idx < 0:
        continue
    N_vals = list(range(5, 121, 5))
    digits = []
    N_valid = []
    for N in N_vals:
        if k0_idx + N < len(C_raw_mp):
            r = richardson(C_raw_mp, k0_idx, N)
            d = digits_agree(r, C_best)
            digits.append(d)
            N_valid.append(N)
    if digits:
        ax2.plot(N_valid, digits, color=color, marker=marker, markersize=3, 
                 linewidth=1.0, label=label_prefix)

ax2.set_xlabel('Richardson order $N$', fontsize=12)
ax2.set_ylabel('Reliable digits', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 25])

plt.tight_layout()
outpath = '../paper/convergence_M4.pdf'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved to {outpath}")
plt.close()
