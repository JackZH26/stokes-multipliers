"""
Rayleigh-Schrödinger perturbation theory for anharmonic oscillators.

Computes perturbation coefficients a_k for:
    H = p²/2 + x²/2 + g·x^{2M}
    E(g) = Σ_{k=0}^∞ a_k g^k

Convention: ℏ=1, m=1, ω=1, so H₀ = p²/2 + x²/2 with E_n = n + 1/2.
Matrix element of x: ⟨m|x|n⟩ = √(n/2) δ_{m,n-1} + √((n+1)/2) δ_{m,n+1}
"""

from mpmath import mp, mpf, sqrt, mpf as _mpf


def compute_coefficients(M=2, N_max=100, dps=50):
    """
    Compute perturbation coefficients a_k for H = p²/2 + x²/2 + g·x^{2M}.

    Parameters
    ----------
    M : int
        Determines the anharmonic potential x^{2M}. M=2 → quartic, M=3 → sextic, etc.
    N_max : int
        Maximum perturbation order (compute a_0, a_1, ..., a_{N_max}).
    dps : int
        Decimal digits of precision for the computation.

    Returns
    -------
    list of mpf
        Coefficients [a_0, a_1, ..., a_{N_max}].
    """
    old_dps = mp.dps
    mp.dps = dps + 30  # guard digits

    ZERO = mpf(0)
    ONE = mpf(1)
    HALF = ONE / 2

    # Maximum quantum number we'll encounter:
    # At order k, |ψ^(k)⟩ has support up to n = 2Mk.
    # Applying V = x^{2M} extends by 2M more.
    nmax = 2 * M * (N_max + 1) + 2

    # Precompute sqrt coefficients for x operator
    # x|n⟩ = sq_down[n]|n-1⟩ + sq_up[n]|n+1⟩
    sq_down = [ZERO] * (nmax + 2)  # sqrt(n/2)
    sq_up = [ZERO] * (nmax + 2)    # sqrt((n+1)/2)
    for n in range(nmax + 2):
        sq_down[n] = sqrt(mpf(n) / 2)
        sq_up[n] = sqrt(mpf(n + 1) / 2)

    def apply_x(vec, sz):
        """Apply x operator to vector vec (of useful size sz). Returns (new_vec, new_sz)."""
        new_sz = min(sz + 1, nmax + 1)
        result = [ZERO] * (nmax + 2)
        for n in range(sz):
            c = vec[n]
            if not c:
                continue
            if n > 0:
                result[n - 1] += c * sq_down[n]
            if n + 1 <= nmax:
                result[n + 1] += c * sq_up[n]
        return result, new_sz

    def apply_V(vec, sz):
        """Apply V = x^{2M} to a vector."""
        v, s = vec, sz
        for _ in range(2 * M):
            v, s = apply_x(v, s)
        return v, s

    # RS perturbation theory
    # E_n = n + 1/2, so E_0 - E_n = -n
    E0 = HALF
    coeffs = [E0]  # a_0

    # Store |ψ^(k)⟩ as (vector, useful_size)
    # |ψ^(0)⟩ = |0⟩
    psi_v = []
    psi_s = []
    v0 = [ZERO] * (nmax + 2)
    v0[0] = ONE
    psi_v.append(v0)
    psi_s.append(1)

    for k in range(1, N_max + 1):
        # Step 1: E^(k) = ⟨0|V|ψ^(k-1)⟩
        Vpsi, Vsz = apply_V(psi_v[k - 1], psi_s[k - 1])
        ak = Vpsi[0]
        coeffs.append(ak)

        # Step 2: Compute |ψ^(k)⟩
        # numerator_n = ⟨n|V|ψ^(k-1)⟩ - Σ_{j=1}^{k-1} a_j · ⟨n|ψ^(k-j)⟩
        # Then ψ^(k)_n = numerator_n / (E_0 - E_n) = numerator_n / (-n)  for n ≠ 0

        # Start with V|ψ^(k-1)⟩
        num = list(Vpsi)
        effective_sz = Vsz

        # Subtract correction terms
        for j in range(1, k):
            aj = coeffs[j]
            pv = psi_v[k - j]
            ps = psi_s[k - j]
            for n in range(ps):
                if pv[n]:
                    num[n] -= aj * pv[n]

        # Divide by energy denominator
        psik = [ZERO] * (nmax + 2)
        for n in range(1, effective_sz):
            if num[n]:
                psik[n] = num[n] / mpf(-n)

        psi_v.append(psik)
        psi_s.append(effective_sz)

        # Progress reporting for long runs
        if N_max >= 50 and k % 50 == 0:
            print(f"  order {k}/{N_max} done")

    mp.dps = dps
    result = [+c for c in coeffs]  # round to target precision
    mp.dps = old_dps
    return result


def compute_coefficients_table(M=2, N_max=100, dps=50):
    """Compute and return a formatted table of coefficients."""
    coeffs = compute_coefficients(M=M, N_max=N_max, dps=dps)
    lines = [f"M={M}, N_max={N_max}, dps={dps}"]
    lines.append("-" * 60)
    for i, c in enumerate(coeffs):
        lines.append(f"a_{i} = {mp.nstr(c, 15)}")
    return "\n".join(lines), coeffs


# Known exact values for validation (M=2, H = p²/2 + x²/2 + gx⁴)
KNOWN_M2 = {
    0: (1, 2),       # 1/2
    1: (3, 4),       # 3/4
    2: (-21, 8),     # -21/8
    3: (333, 16),    # 333/16
    4: (-30885, 128),  # -30885/128
}


def validate_M2(coeffs, tol=None):
    """Validate M=2 coefficients against known exact values."""
    if tol is None:
        tol = mpf(10) ** (-(mp.dps - 10))

    results = []
    all_pass = True
    for k, (num, den) in KNOWN_M2.items():
        if k >= len(coeffs):
            break
        expected = mpf(num) / mpf(den)
        actual = coeffs[k]
        err = abs(actual - expected)
        ok = err < tol
        if not ok:
            all_pass = False
        results.append((k, expected, actual, err, ok))

    return all_pass, results


if __name__ == "__main__":
    mp.dps = 50
    print("Computing M=2 coefficients...")
    coeffs = compute_coefficients(M=2, N_max=10, dps=50)

    print("\nM=2 perturbation coefficients:")
    for i, c in enumerate(coeffs):
        print(f"  a_{i} = {mp.nstr(c, 40)}")

    print("\nValidation:")
    ok, results = validate_M2(coeffs)
    for k, expected, actual, err, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  a_{k}: expected={mp.nstr(expected,20)}, got={mp.nstr(actual,20)}, err={mp.nstr(err,5)} [{status}]")

    print(f"\nOverall: {'ALL PASSED' if ok else 'SOME FAILED'}")
