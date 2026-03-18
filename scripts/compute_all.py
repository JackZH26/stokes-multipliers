#!/usr/bin/env python3
"""
Reproduce all results from the paper.

Computes Stokes multipliers C_M for M = 2 through 11.
WARNING: Full computation (especially M=4 at 1200 orders) takes ~2 hours total.

For a quick check, use --quick flag (lower precision, fewer orders).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mpmath import mp, mpf, pi, sqrt, gamma, nstr
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


# Computation parameters from the paper
PAPER_PARAMS = {
    2:  {"orders": 500,  "dps": 200},
    3:  {"orders": 500,  "dps": 200},
    4:  {"orders": 1200, "dps": 300},
    5:  {"orders": 600,  "dps": 200},
    6:  {"orders": 500,  "dps": 200},
    7:  {"orders": 400,  "dps": 180},
    8:  {"orders": 300,  "dps": 180},
    9:  {"orders": 300,  "dps": 180},
    10: {"orders": 250,  "dps": 160},
    11: {"orders": 250,  "dps": 160},
}

QUICK_PARAMS = {M: {"orders": min(200, v["orders"]), "dps": min(100, v["dps"])}
                for M, v in PAPER_PARAMS.items()}


def compute_one(M, orders, dps):
    """Compute C_M with given parameters."""
    mp.dps = dps
    p = M - 1
    A = instanton_action(M, dps=dps)

    print(f"\n{'='*60}")
    print(f"M = {M} | orders = {orders} | dps = {dps}")
    print(f"A_{M} = {nstr(A, 25)}")
    print(f"{'='*60}")

    coeffs = compute_coefficients(M=M, N_max=orders, dps=dps)
    print(f"Computed {len(coeffs)} coefficients")

    # Build C_k sequence
    C_vals = []
    for k in range(15, len(coeffs)):
        gval = gamma(p * mpf(k) + mpf('0.5'))
        Ck = coeffs[k] * ((-A)**k) / gval
        C_vals.append(Ck)

    # Richardson from multiple windows
    print("Richardson extrapolation:")
    estimates = []
    for start, nt in [(40, 30), (60, 40), (80, 50), (100, 40)]:
        if start + nt <= len(C_vals):
            cr = richardson_extrapolate(C_vals[start:], N_terms=nt)
            print(f"  s={start+15}, N={nt}: {nstr(cr, 28)}")
            estimates.append(cr)

    best = estimates[0] if estimates else C_vals[-1]
    print(f"\nC_{M} = {nstr(best, 30)}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode (lower precision)")
    parser.add_argument("--M", type=int, nargs="+", help="Specific M values (default: 2-11)")
    args = parser.parse_args()

    params = QUICK_PARAMS if args.quick else PAPER_PARAMS
    M_values = args.M if args.M else list(range(2, 12))

    results = {}
    for M in M_values:
        p = params.get(M, {"orders": 200, "dps": 100})
        C = compute_one(M, p["orders"], p["dps"])
        results[M] = {
            "C": nstr(C, 30),
            "absC": nstr(abs(C), 30),
            "A": nstr(instanton_action(M, dps=50), 30),
        }

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'M':>3} | {'|C_M|':>32}")
    print("-" * 40)
    for M in sorted(results.keys()):
        print(f"{M:>3} | {results[M]['absC']}")

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'results.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {outpath}")
