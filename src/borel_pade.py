"""
Borel transform + Padé approximant + Borel summation.

For a divergent series E(g) = Σ a_k g^k with a_k ~ C·(-A)^{-k}·Γ(k+b+1),
the Borel transform B(t) = Σ b_k t^k (where b_k = a_k / Γ(k+1)) is convergent
for |t| < A. Padé approximants analytically continue B(t) beyond this radius,
and the Borel sum is recovered via Laplace transform:

    E_Borel(g) = ∫_0^∞ e^{-t} B(g·t) dt
"""

from mpmath import mp, mpf, gamma, fac, quad, exp, inf, matrix, lu_solve, power


def borel_transform(coeffs):
    """
    Compute Borel-transformed coefficients.

    b_k = a_k / Γ(k+1) = a_k / k!

    Parameters
    ----------
    coeffs : list of mpf
        Original perturbation coefficients [a_0, a_1, ...].

    Returns
    -------
    list of mpf
        Borel coefficients [b_0, b_1, ...].
    """
    borel = []
    for k, ak in enumerate(coeffs):
        bk = ak / fac(k)
        borel.append(bk)
    return borel


def pade_approximant(coeffs, N=None):
    """
    Compute [N/N] Padé approximant from power series coefficients.

    Given f(z) = Σ_{k=0}^{2N} c_k z^k + O(z^{2N+1}),
    find P(z)/Q(z) with deg(P)=N, deg(Q)=N, Q(0)=1, matching f to order 2N.

    Parameters
    ----------
    coeffs : list of mpf
        Power series coefficients [c_0, c_1, ..., c_{2N}] (need at least 2N+1).
    N : int or None
        Order of Padé. Default: (len(coeffs)-1)//2.

    Returns
    -------
    (p_coeffs, q_coeffs) : tuple of lists of mpf
        Numerator and denominator polynomial coefficients.
        P(z) = p_0 + p_1 z + ... + p_N z^N
        Q(z) = 1 + q_1 z + ... + q_N z^N
    """
    if N is None:
        N = (len(coeffs) - 1) // 2

    c = coeffs
    if len(c) < 2 * N + 1:
        raise ValueError(f"Need at least {2*N+1} coefficients, got {len(c)}")

    # Solve for denominator coefficients q_1, ..., q_N
    # from the equations:
    # Σ_{j=0}^{min(k,N)} q_j c_{k-j} = 0  for k = N+1, ..., 2N
    # where q_0 = 1

    # This gives N linear equations in N unknowns q_1, ..., q_N:
    # Σ_{j=1}^{N} q_j c_{k-j} = -c_k  for k = N+1, ..., 2N

    if N == 0:
        return [c[0]], [mpf(1)]

    A_mat = matrix(N, N)
    b_vec = matrix(N, 1)

    for i in range(N):
        k = N + 1 + i
        b_vec[i, 0] = -c[k]
        for j in range(1, N + 1):
            idx = k - j
            if 0 <= idx < len(c):
                A_mat[i, j - 1] = c[idx]

    # Solve A q = b
    q_vec = lu_solve(A_mat, b_vec)

    # Denominator: Q(z) = 1 + q_1 z + ... + q_N z^N
    q_coeffs = [mpf(1)]
    for j in range(N):
        q_coeffs.append(q_vec[j, 0])

    # Numerator: P(z) = Σ_{k=0}^{N} p_k z^k
    # where p_k = Σ_{j=0}^{k} q_j c_{k-j}
    p_coeffs = []
    for k in range(N + 1):
        pk = mpf(0)
        for j in range(min(k, N) + 1):
            pk += q_coeffs[j] * c[k - j]
        p_coeffs.append(pk)

    return p_coeffs, q_coeffs


def eval_polynomial(coeffs, z):
    """Evaluate polynomial Σ c_k z^k using Horner's method."""
    if not coeffs:
        return mpf(0)
    result = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * z + coeffs[k]
    return result


def eval_pade(p_coeffs, q_coeffs, z):
    """Evaluate Padé approximant P(z)/Q(z) at point z."""
    return eval_polynomial(p_coeffs, z) / eval_polynomial(q_coeffs, z)


def borel_sum(coeffs, g, N_pade=None, dps=50):
    """
    Compute the Borel sum of Σ a_k g^k via Borel-Padé method.

    Steps:
    1. Borel transform: b_k = a_k / k!
    2. Padé approximant of B(t) = Σ b_k t^k
    3. Laplace integral: E(g) = ∫_0^∞ e^{-t} B_Padé(g·t) dt

    For g > 0, the integration is along the positive real axis.
    For g < 0, one may need lateral Borel sums (above/below the singularities).

    Parameters
    ----------
    coeffs : list of mpf
        Perturbation coefficients [a_0, a_1, ...].
    g : mpf
        Coupling constant value.
    N_pade : int or None
        Order of Padé approximant. Default: (len(coeffs)-1)//2.
    dps : int
        Working precision.

    Returns
    -------
    mpf
        Borel-summed value E(g).
    """
    old_dps = mp.dps
    mp.dps = dps + 20

    g = mpf(g)

    # Step 1: Borel transform
    borel_coeffs = borel_transform(coeffs)

    # Step 2: Padé approximant
    if N_pade is None:
        N_pade = (len(borel_coeffs) - 1) // 2
    p, q = pade_approximant(borel_coeffs, N=N_pade)

    # Step 3: Laplace integral
    def integrand(t):
        return exp(-t) * eval_pade(p, q, g * t)

    result = quad(integrand, [0, inf], method='tanh-sinh')

    mp.dps = dps
    result = +result
    mp.dps = old_dps
    return result


def borel_sum_lateral(coeffs, g, N_pade=None, dps=50, direction=+1):
    """
    Lateral Borel sum: integrate slightly above (+1) or below (-1) the real axis.

    For g < 0, the Borel transform has singularities on the positive real t-axis.
    The lateral sums avoid these by deforming the contour:
        E±(g) = ∫_0^{∞±iε} e^{-t} B_Padé(g·t) dt

    The imaginary part of the difference gives the non-perturbative ambiguity.

    Parameters
    ----------
    direction : +1 or -1
        +1 for contour above real axis, -1 for below.
    """
    old_dps = mp.dps
    mp.dps = dps + 20

    g = mpf(g)

    borel_coeffs = borel_transform(coeffs)
    if N_pade is None:
        N_pade = (len(borel_coeffs) - 1) // 2
    p, q = pade_approximant(borel_coeffs, N=N_pade)

    # Integrate along a slightly rotated contour
    eps = mpf(10) ** (-dps // 2) * direction

    def integrand(t):
        z = t * (1 + eps * 1j)
        return exp(-z) * eval_pade(p, q, g * z) * (1 + eps * 1j)

    result = quad(integrand, [0, inf], method='tanh-sinh')

    mp.dps = dps
    result = +result
    mp.dps = old_dps
    return result


if __name__ == "__main__":
    mp.dps = 30

    # Test with M=2 quartic oscillator
    # Known: E(g=0.1) ≈ 0.559146 (from exact diagonalization / Borel sum)
    import sys
    sys.path.insert(0, '.')
    from bender_wu import compute_coefficients

    print("Computing M=2 coefficients for Borel sum test...")
    coeffs = compute_coefficients(M=2, N_max=40, dps=40)

    g_test = mpf("0.1")
    result = borel_sum(coeffs, g_test, dps=30)
    print(f"Borel sum at g={g_test}: {result}")
    print(f"Direct partial sum (10 terms): {sum(coeffs[k] * g_test**k for k in range(11))}")
