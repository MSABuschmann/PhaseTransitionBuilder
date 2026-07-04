#pragma once

// Filon quadrature for highly oscillatory integrals.
//
// Computes simultaneously:
//   I_cos = ∫_a^b g(u) cos(ω u) du
//   I_sin = ∫_a^b g(u) sin(ω u) du
//
// using the classical Filon rule on N uniform sub-intervals (N must be even).
// The oscillatory factor is integrated exactly; only the smooth envelope g(u)
// is sampled on the N+1 grid.  For ω → 0 the rule reduces to Simpson's rule.
//
// Reference: Filon (1928); Abramowitz & Stegun §25.4.

#include <cmath>

// Minimum number of sub-intervals (must be even).
// The actual N per integral is raised automatically to keep ~8 panels per
// Bessel-function oscillation cycle in g(u); see integral_u_filon in
// filon_integrator.cpp.
// Set FILON_N_TEST > 0 at compile time (e.g. -DFILON_N_TEST=32768) to
// override FILON_N_MIN for convergence testing.
#ifdef FILON_N_TEST
static constexpr int FILON_N_MIN = FILON_N_TEST;
#else
static constexpr int FILON_N_MIN = 2048;
#endif

// ---------------------------------------------------------------------------
// Filon coefficients
// ---------------------------------------------------------------------------

inline void filon_coeffs(double theta,
                          double &alpha, double &beta, double &gamma) {
    if (std::abs(theta) < 1e-6) {
        // Taylor series to avoid catastrophic cancellation near θ = 0.
        // α → 2θ³/45,  β → 2/3 + 2θ²/15,  γ → 4/3 − 2θ²/15
        const double th2 = theta * theta;
        alpha = 2.0 * th2 * theta / 45.0;
        beta  = 2.0 / 3.0 + 2.0 * th2 / 15.0;
        gamma = 4.0 / 3.0 - 2.0 * th2 / 15.0;
    } else {
        const double s   = std::sin(theta);
        const double c   = std::cos(theta);
        const double th2 = theta * theta;
        const double th3 = th2 * theta;
        alpha = (th2 + theta * s * c - 2.0 * s * s) / th3;
        beta  = 2.0 * (theta * (1.0 + c * c) - 2.0 * s * c) / th3;
        gamma = 4.0 * (s - theta * c) / th3;
    }
}

// ---------------------------------------------------------------------------
// Core Filon rule
// g_vals[i] = g(a + i·h)  for  i = 0 … N  (N+1 values, N even)
// ---------------------------------------------------------------------------

inline void filon_cos_sin(const double * __restrict__ g_vals, int N,
                           double a, double h, double omega,
                           double &I_cos, double &I_sin) {
    double alpha, beta, gamma_f;
    filon_coeffs(omega * h, alpha, beta, gamma_f);

    // Even-index weighted sums (trapezoidal endpoint weight = ½)
    double Se_c = 0.5 * (g_vals[0] * std::cos(omega * a)
                       + g_vals[N] * std::cos(omega * (a + N * h)));
    double Se_s = 0.5 * (g_vals[0] * std::sin(omega * a)
                       + g_vals[N] * std::sin(omega * (a + N * h)));
    for (int i = 2; i < N; i += 2) {
        const double wu = omega * (a + i * h);
        Se_c += g_vals[i] * std::cos(wu);
        Se_s += g_vals[i] * std::sin(wu);
    }

    // Odd-index sums
    double So_c = 0.0, So_s = 0.0;
    for (int i = 1; i < N; i += 2) {
        const double wu = omega * (a + i * h);
        So_c += g_vals[i] * std::cos(wu);
        So_s += g_vals[i] * std::sin(wu);
    }

    const double g0 = g_vals[0], gN = g_vals[N];
    const double wu0 = omega * a, wuN = omega * (a + N * h);

    I_cos = h * (alpha * (gN * std::sin(wuN) - g0 * std::sin(wu0))
                 + beta * Se_c + gamma_f * So_c);
    I_sin = h * (alpha * (-gN * std::cos(wuN) + g0 * std::cos(wu0))
                 + beta * Se_s + gamma_f * So_s);
}
