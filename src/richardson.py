"""
Richardson extrapolation and large-order analysis tools.

Tools for extracting instanton action A, parameter b, and Stokes multiplier C
from the large-order behavior of perturbation coefficients:

    a_k ~ C · (-A)^{-k} · Γ(k + b + 1) · [1 + c₁/k + c₂/k² + ...]
"""

from mpmath import mp, mpf, gamma, power, fac, pi, sqrt, binomial, matrix


def richardson_extrapolate(seq, N_terms=None):
    """
    Richardson extrapolation to accelerate convergence of seq[k] → L.

    Uses the standard formula that eliminates 1/k, 1/k², ..., 1/k^m corrections:
        T_n^(m) = Σ_{j=0}^{m} (-1)^j C(m,j) (n+j+1)^m s_{n+j}
                / Σ_{j=0}^{m} (-1)^j C(m,j) (n+j+1)^m

    Parameters
    ----------
    seq : list of mpf
        Sequence converging to some limit L.
    N_terms : int or None
        Order of extrapolation (number of correction terms to eliminate).
        Default: min(20, len(seq)//2).

    Returns
    -------
    mpf
        Extrapolated limit estimate.
    """
    if N_terms is None:
        N_terms = min(20, len(seq) // 2)

    n_seq = len(seq)
    if N_terms <= 0 or n_seq < N_terms + 1:
        return seq[-1] if seq else mpf(0)

    # Use the last N_terms+1 values of the sequence
    m = N_terms  # order of extrapolation
    start = n_seq - m - 1  # starting index in seq

    numerator = mpf(0)
    denominator = mpf(0)

    for j in range(m + 1):
        binom_coeff = binomial(m, j)
        sign = mpf(-1) ** j
        idx = start + j
        weight = mpf(idx + 1) ** m  # (n+j+1)^m where n=start
        term = sign * binom_coeff * weight

        numerator += term * seq[idx]
        denominator += term

    if denominator == 0:
        return seq[-1]

    return numerator / denominator


def richardson_extrapolate_sequence(seq, max_order=None):
    """
    Compute a sequence of Richardson extrapolants of increasing order.

    Returns list of (order, extrapolated_value) pairs.
    """
    if max_order is None:
        max_order = min(30, len(seq) // 2)

    results = []
    for m in range(1, max_order + 1):
        val = richardson_extrapolate(seq, N_terms=m)
        results.append((m, val))

    return results


def extract_instanton_action(coeffs, dps=200):
    """
    Extract the instanton action A from the ratio test.

    For a_k ~ C · (-A)^{-k} · Γ(k+b+1):
        r_k = a_k / a_{k-1} → -1/A · (k + b)  as k → ∞
        So A_k = -k / r_k → A  as k → ∞

    Parameters
    ----------
    coeffs : list of mpf
        Perturbation coefficients [a_0, a_1, ...].
    dps : int
        Working precision.

    Returns
    -------
    dict with keys:
        'A_sequence': list of successive A_k estimates
        'A_extrapolated': Richardson-extrapolated value of A
        'ratio_sequence': the raw ratios r_k
    """
    old_dps = mp.dps
    mp.dps = dps

    n = len(coeffs)
    ratios = []
    A_seq = []

    for k in range(1, n):
        if coeffs[k - 1] == 0:
            continue
        rk = coeffs[k] / coeffs[k - 1]
        ratios.append(rk)
        # A_k = -k / r_k
        if rk != 0:
            Ak = mpf(-k) / rk
            A_seq.append(Ak)

    # Richardson extrapolate A_seq
    A_extrap = mpf(0)
    if len(A_seq) > 4:
        A_extrap = richardson_extrapolate(A_seq)

    mp.dps = old_dps
    return {
        'A_sequence': A_seq,
        'A_extrapolated': +A_extrap,
        'ratio_sequence': ratios,
    }


def extract_b_parameter(coeffs, A, dps=200):
    """
    Extract the parameter b from the ratio test.

    r_k = a_k / a_{k-1} → -(k + b) / A
    So b_k = -A · r_k - k → b  as k → ∞

    Parameters
    ----------
    coeffs : list of mpf
        Perturbation coefficients.
    A : mpf
        Instanton action (already determined).
    dps : int
        Working precision.

    Returns
    -------
    dict with 'b_sequence' and 'b_extrapolated'.
    """
    old_dps = mp.dps
    mp.dps = dps

    n = len(coeffs)
    b_seq = []

    for k in range(1, n):
        if coeffs[k - 1] == 0:
            continue
        rk = coeffs[k] / coeffs[k - 1]
        bk = -A * rk - k
        b_seq.append(bk)

    b_extrap = mpf(0)
    if len(b_seq) > 4:
        b_extrap = richardson_extrapolate(b_seq)

    mp.dps = old_dps
    return {
        'b_sequence': b_seq,
        'b_extrapolated': +b_extrap,
    }


def extract_stokes_multiplier(coeffs, A, b, dps=200):
    """
    Extract the Stokes multiplier (leading prefactor C) from:

        a_k ~ C · (-A)^{-k} · Γ(k + b + 1)

    Construct C_k = a_k · (-A)^k / Γ(k + b + 1) and Richardson extrapolate.

    Parameters
    ----------
    coeffs : list of mpf
        Perturbation coefficients.
    A : mpf
        Instanton action.
    b : mpf
        Gamma function shift parameter.
    dps : int
        Working precision.

    Returns
    -------
    dict with 'C_sequence' and 'C_extrapolated'.
    """
    old_dps = mp.dps
    mp.dps = dps

    n = len(coeffs)
    C_seq = []

    for k in range(n):
        gamma_val = gamma(mpf(k) + b + 1)
        if gamma_val == 0:
            continue
        neg_A_pow_k = power(-A, k)
        Ck = coeffs[k] * neg_A_pow_k / gamma_val  # should be: Ck = a_k / ((-A)^{-k} Γ(k+b+1)) = a_k (-A)^k / Γ(k+b+1)
        # Wait: a_k ~ C · (-A)^{-k} · Γ(k+b+1)
        # So C_k = a_k / [(-A)^{-k} · Γ(k+b+1)] = a_k · (-A)^k / Γ(k+b+1)
        C_seq.append(Ck)

    C_extrap = mpf(0)
    if len(C_seq) > 4:
        C_extrap = richardson_extrapolate(C_seq)

    mp.dps = old_dps
    return {
        'C_sequence': C_seq,
        'C_extrapolated': +C_extrap,
    }


def full_large_order_analysis(coeffs, dps=200):
    """
    Perform full large-order analysis: extract A, b, and C.

    Returns dict with all extracted parameters and sequences.
    """
    old_dps = mp.dps
    mp.dps = dps

    # Step 1: Extract instanton action A
    A_result = extract_instanton_action(coeffs, dps=dps)
    A = A_result['A_extrapolated']

    # Step 2: Extract b parameter
    b_result = extract_b_parameter(coeffs, A, dps=dps)
    b = b_result['b_extrapolated']

    # Step 3: Extract Stokes multiplier C
    C_result = extract_stokes_multiplier(coeffs, A, b, dps=dps)

    mp.dps = old_dps
    return {
        'A': +A,
        'b': +b,
        'C': +C_result['C_extrapolated'],
        'A_details': A_result,
        'b_details': b_result,
        'C_details': C_result,
    }


if __name__ == "__main__":
    # Quick test with M=2 known large-order behavior
    from mpmath import mpf
    mp.dps = 30

    # Generate a test sequence: s_k = L + c₁/k + c₂/k²
    L = mpf("3.14159265358979323846")
    test_seq = [L + mpf(1)/k + mpf(2)/k**2 for k in range(1, 50)]
    result = richardson_extrapolate(test_seq, N_terms=10)
    print(f"Test Richardson: expected π ≈ {L}")
    print(f"  Got: {result}")
    print(f"  Error: {abs(result - L)}")
