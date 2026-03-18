"""
PSLQ closed-form search for Stokes multipliers and related constants.

Uses mpmath's pslq() to search for integer relations among a target value
and a basis of mathematical constants. Employs a layered search strategy
with progressively larger constant bases.
"""

from mpmath import mp, mpf, pi, euler, log, zeta, gamma, sqrt, power, catalan


def constant_basis_layer(layer, M=None, dps=200):
    """
    Return a list of (name, value) pairs for a constant basis at given layer.

    Layer 0: Basic constants (π, ln2)
    Layer 1: + Gamma function values at rational points related to M
    Layer 2: + Zeta values, Catalan's constant
    Layer 3: + Products and powers of above

    Parameters
    ----------
    layer : int
        Complexity layer (0-3).
    M : int or None
        Potential parameter M from x^{2M}, used for Gamma function arguments.
    dps : int
        Working precision.

    Returns
    -------
    list of (str, mpf)
        Named constants.
    """
    old_dps = mp.dps
    mp.dps = dps + 10

    basis = [("1", mpf(1))]

    # Layer 0: π and ln2
    basis.append(("pi", pi))
    basis.append(("ln2", log(2)))

    if layer >= 1 and M is not None:
        # Gamma function at rational points related to M
        for denom in [M, M - 1, 2 * M, 2 * (M - 1)]:
            if denom > 0:
                for numer in range(1, denom):
                    name = f"Gamma({numer}/{denom})"
                    val = gamma(mpf(numer) / mpf(denom))
                    # Avoid duplicates
                    if not any(n == name for n, _ in basis):
                        basis.append((name, val))

        # sqrt(π) often appears
        basis.append(("sqrt(pi)", sqrt(pi)))

    if layer >= 2:
        # Zeta values
        for n in range(2, 6):
            basis.append((f"zeta({n})", zeta(n)))
        basis.append(("catalan", catalan))
        basis.append(("euler_gamma", euler))

    if layer >= 3:
        # Products and powers
        basis.append(("pi^2", pi ** 2))
        basis.append(("pi*ln2", pi * log(2)))
        basis.append(("ln2^2", log(2) ** 2))
        if M is not None and M >= 2:
            basis.append(("pi/sqrt(2)", pi / sqrt(2)))

    mp.dps = old_dps
    return basis


def pslq_search(target, basis_values, dps=200, maxcoeff=1000, maxsteps=None):
    """
    Search for an integer relation: n_0·target + n_1·v_1 + ... + n_k·v_k = 0.

    If found, returns the integer coefficients. The closed-form expression is:
        target = -(n_1·v_1 + ... + n_k·v_k) / n_0

    Parameters
    ----------
    target : mpf
        The value to express as a closed form.
    basis_values : list of mpf
        The constant basis values.
    dps : int
        Working precision.
    maxcoeff : int
        Maximum allowed coefficient in PSLQ result.
    maxsteps : int or None
        Maximum PSLQ iterations.

    Returns
    -------
    list of int or None
        Integer relation [n_0, n_1, ...] or None if not found.
    """
    old_dps = mp.dps
    mp.dps = dps

    x = [target] + list(basis_values)

    kwargs = {"maxcoeff": maxcoeff}
    if maxsteps is not None:
        kwargs["maxsteps"] = maxsteps

    try:
        rel = mp.pslq(x, **kwargs)
    except Exception:
        rel = None

    mp.dps = old_dps
    return rel


def format_relation(names, relation):
    """
    Format a PSLQ integer relation as a human-readable closed-form expression.

    Given relation [n_0, n_1, ..., n_k] meaning n_0·target + Σ n_i·basis_i = 0,
    express target = -(Σ n_i·basis_i) / n_0.
    """
    if relation is None:
        return "No relation found"

    n0 = relation[0]
    if n0 == 0:
        return "Degenerate relation (coefficient of target is 0)"

    terms = []
    for i, (name, ni) in enumerate(zip(names[1:], relation[1:])):
        if ni == 0:
            continue
        coeff = mpf(-ni) / mpf(n0)
        if coeff == int(coeff):
            coeff = int(coeff)
        terms.append(f"({coeff})·{name}")

    if not terms:
        return "target = 0"

    return "target = " + " + ".join(terms)


def layered_search(target, M=None, dps=200, max_layer=3, maxcoeff=1000):
    """
    Perform layered PSLQ search with increasing constant basis complexity.

    Starts with simple constants and progressively adds more complex ones.
    Returns as soon as a relation is found.

    Parameters
    ----------
    target : mpf
        Value to find closed form for.
    M : int
        Anharmonic oscillator parameter.
    dps : int
        Working precision.
    max_layer : int
        Maximum layer to try (0-3).
    maxcoeff : int
        Maximum coefficient in PSLQ.

    Returns
    -------
    dict or None
        Result with 'layer', 'relation', 'names', 'formatted' keys.
    """
    old_dps = mp.dps

    for layer in range(max_layer + 1):
        mp.dps = dps
        basis = constant_basis_layer(layer, M=M, dps=dps)
        names = [name for name, _ in basis]
        values = [val for _, val in basis]

        rel = pslq_search(target, values, dps=dps, maxcoeff=maxcoeff)

        if rel is not None:
            formatted = format_relation(["target"] + names, rel)
            mp.dps = old_dps
            return {
                "layer": layer,
                "relation": rel,
                "names": ["target"] + names,
                "formatted": formatted,
            }

    mp.dps = old_dps
    return None


if __name__ == "__main__":
    mp.dps = 50

    # Test: verify that PSLQ can find A_2 = 4/3
    target = mpf(4) / 3
    basis = [mpf(1)]
    rel = pslq_search(target, basis, dps=50)
    print(f"PSLQ for 4/3: relation = {rel}")
    # Expected: [3, -4] meaning 3·target + (-4)·1 = 0, so target = 4/3

    # Test: verify A_3 = π/2
    target = pi / 2
    result = layered_search(target, M=3, dps=50)
    print(f"PSLQ for π/2: {result}")
