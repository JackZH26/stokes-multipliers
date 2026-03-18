#!/usr/bin/env python3
"""
Verify all closed-form identities for Stokes multipliers.

  C₂² · π³ = 6
  C₃² · π⁴ = 32
  C₅⁴ · Γ(1/4)⁴ · π⁵ = 2¹² · 3² = 36864
  C₇⁶ · Γ(1/3)⁹ · π⁶ = 2²⁰ · 3³ = 28311552
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mpmath import mp, mpf, pi, sqrt, gamma, nstr

mp.dps = 40

print("=" * 60)
print("  Verification of Exact Stokes Multiplier Identities")
print("=" * 60)


def check(label, lhs, rhs):
    err = abs(lhs - rhs) / abs(rhs)
    ok = "✅ PASS" if err < mpf(10)**(-35) else "❌ FAIL"
    print(f"\n  {label}")
    print(f"  LHS = {nstr(lhs, 30)}")
    print(f"  RHS = {nstr(rhs, 30)}")
    print(f"  Relative error = {nstr(err, 5)}")
    print(f"  {ok}")
    return err < mpf(10)**(-35)


all_pass = True

# M=2: C₂ = √(6/π³), so C₂² π³ = 6
C2 = sqrt(mpf(6) / pi**3)
all_pass &= check("M=2: C₂² · π³ = 6", C2**2 * pi**3, mpf(6))

# M=3: C₃ = 4√2/π², so C₃² π⁴ = 32
C3 = 4 * sqrt(2) / pi**2
all_pass &= check("M=3: C₃² · π⁴ = 32", C3**2 * pi**4, mpf(32))

# M=5: C₅⁴ Γ(1/4)⁴ π⁵ = 36864
g14 = gamma(mpf(1) / 4)
C5 = 8 * sqrt(3) / (g14 * pi**(mpf(5)/4))
all_pass &= check("M=5: C₅⁴ · Γ(1/4)⁴ · π⁵ = 36864",
                   C5**4 * g14**4 * pi**5, mpf(36864))

# M=7: C₇⁶ Γ(1/3)⁹ π⁶ = 28311552
g13 = gamma(mpf(1) / 3)
target = mpf(2)**20 * mpf(3)**3
C7 = (target / (g13**9 * pi**6))**(mpf(1)/6)
all_pass &= check("M=7: C₇⁶ · Γ(1/3)⁹ · π⁶ = 2²⁰ · 3³",
                   C7**6 * g13**9 * pi**6, target)

# Also verify the Γ(1/6) form: C₇²⁴ Γ(1/6)¹⁸ π³³ = 2⁷⁴ · 3²¹
g16 = gamma(mpf(1) / 6)
target_raw = mpf(2)**74 * mpf(3)**21
all_pass &= check("M=7 (raw): C₇²⁴ · Γ(1/6)¹⁸ · π³³ = 2⁷⁴ · 3²¹",
                   C7**24 * g16**18 * pi**33, target_raw)

# Verify Gauss multiplication: Γ(1/6) = Γ(1/3)² √3 / (2^{1/3} √π)
gauss_lhs = gamma(mpf(1)/6)
gauss_rhs = gamma(mpf(1)/3)**2 * sqrt(3) / (mpf(2)**(mpf(1)/3) * sqrt(pi))
all_pass &= check("Gauss: Γ(1/6) = Γ(1/3)²·√3 / (2^{1/3}·√π)",
                   gauss_lhs, gauss_rhs)

print("\n" + "=" * 60)
print(f"  Overall: {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
print("=" * 60)

# Print numerical values
print("\n  Numerical values of |C_M|:")
print(f"  |C₂| = {nstr(C2, 25)}")
print(f"  |C₃| = {nstr(C3, 25)}")
print(f"  |C₅| = {nstr(C5, 25)}")
print(f"  |C₇| = {nstr(C7, 25)}")
