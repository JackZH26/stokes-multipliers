#!/usr/bin/env python3
"""
Compute the Stokes multiplier C_M for a single value of M.

Usage:
    python compute_single.py --M 5 --orders 300 --dps 150
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mpmath import mp, mpf, pi, sqrt, gamma, pslq, log
from src.bender_wu import compute_coefficients
from src.richardson import richardson_extrapolate


def instanton_action(M, dps=50):
    """Compute exact instanton action A_M = S0^{M-1}."""
    old = mp.dps; mp.dps = dps + 20
    p = M - 1
    S0 = (mpf(1)/2)**(mpf(1)/p) * gamma(mpf(1)/p) * sqrt(pi) / (2*p*gamma(mpf(1)/p + mpf(3)/2))
    A = S0**p
    mp.dps = old
    return +A


def compute_CM(M, N_max=300, dps=150, verbose=True):
    """Full pipeline: coefficients -> Richardson -> C_M."""
    mp.dps = dps

    A = instanton_action(M, dps=dps)
    if verbose:
        print(f"M={M}, A = {mp.nstr(A, 25)}")

    # Compute perturbation coefficients
    if verbose:
        print(f"Computing {N_max} orders at {dps} dps...")
    coeffs = compute_coefficients(M=M, N_max=N_max, dps=dps)
    if verbose:
        print(f"Got {len(coeffs)} coefficients")

    # Extract C_M sequence
    p = M - 1
    C_vals = []
    for k in range(15, len(coeffs)):
        gval = gamma(p * mpf(k) + mpf('0.5'))
        Ck = coeffs[k] * ((-A)**k) / gval
        C_vals.append(Ck)

    # Richardson extrapolation from multiple windows
    if verbose:
        print("\nRichardson extrapolation:")
    best = None
    for start, nt in [(40, 30), (60, 40), (80, 50), (100, 40)]:
        if start + nt <= len(C_vals):
            cr = richardson_extrapolate(C_vals[start:], N_terms=nt)
            if verbose:
                print(f"  start={start+15}, N={nt}: {mp.nstr(cr, 28)}")
            if best is None:
                best = cr

    if best is not None:
        if verbose:
            print(f"\nC_{M} = {mp.nstr(best, 30)}")
            print(f"|C_{M}| = {mp.nstr(abs(best), 30)}")
        return best
    else:
        print("ERROR: Not enough data for Richardson extrapolation")
        return None


def pslq_search(C, M, dps=20, verbose=True):
    """Search for closed form using PSLQ."""
    old = mp.dps; mp.dps = dps
    p = M - 1
    absC = abs(+C)

    lC = log(absC)
    lpi = log(pi)
    l2 = log(mpf(2))
    l3 = log(mpf(3))

    if p >= 3:
        lG = log(gamma(mpf(1)/p))
        basis = [lC, lG, lpi, l2, l3, mpf(1)]
        label = f"G(1/{p})"
    else:
        basis = [lC, lpi, l2, l3, mpf(1)]
        label = "pi only"

    if verbose:
        print(f"\nPSLQ search ({label}, {dps} dps):")

    for mc in [100, 200, 500, 1000]:
        rel = pslq(basis, maxcoeff=mc)
        if rel and rel[0] != 0:
            if verbose:
                print(f"  FOUND (maxcoeff={mc}): {rel}")
            mp.dps = old
            return rel
        elif verbose:
            print(f"  maxcoeff={mc}: no relation")

    mp.dps = old
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Stokes multiplier C_M")
    parser.add_argument("--M", type=int, required=True, help="Potential power: V = x^{2M}")
    parser.add_argument("--orders", type=int, default=300, help="Number of perturbation orders")
    parser.add_argument("--dps", type=int, default=150, help="Decimal digits of precision")
    parser.add_argument("--pslq", action="store_true", help="Run PSLQ closed-form search")
    args = parser.parse_args()

    C = compute_CM(args.M, N_max=args.orders, dps=args.dps)

    if C is not None and args.pslq:
        pslq_search(C, args.M, dps=min(20, args.dps - 5))
