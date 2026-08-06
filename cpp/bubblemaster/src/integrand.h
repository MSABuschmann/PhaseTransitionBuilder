// Fast Bessel functions and cutoff/integrand functions.
// Carried from the reference BubbleMaster implementation.

#pragma once

#include <cmath>

inline double fast_bessel_j0(double x) {
    double ax = std::fabs(x);
    if (ax < 8.0) {
        double y = x * x;
        double num = 57568490574.0
            + y * (-13362590354.0
            + y * (651619640.7
            + y * (-11214424.18
            + y * (77392.33017
            + y * (-184.9052456)))));
        double den = 57568490411.0
            + y * (1029532985.0
            + y * (9494680.718
            + y * (59272.64853
            + y * (267.8532712 + y))));
        return num / den;
    } else {
        double z  = 8.0 / ax;
        double y  = z * z;
        double xx = ax - 0.785398164;
        return std::sqrt(0.636619772 / ax) *
               (std::cos(xx) * (1.0 + y * (-0.1098628627e-2
                    + y * (0.2734510407e-4
                    + y * (-0.2073370639e-5 + y * 0.2093887211e-6))))
              - z * std::sin(xx) * (0.1562499995e-1
                    + y * (0.1430488765e-3
                    + y * (-0.6911147651e-5 + y * (0.7621095161e-6)))));
    }
}

inline double fast_bessel_j1(double x) {
    double ax = std::fabs(x), ans;
    if (ax < 8.0) {
        double y = x * x;
        double num = x * (72362614232.0
            + y * (-7895059235.0
            + y * (242396853.1
            + y * (-2972611.439
            + y * (15704.48260 + y * (-30.16036606))))));
        double den = 144725228442.0
            + y * (2300535178.0
            + y * (18583304.74
            + y * (99447.43394 + y * (376.9991397 + y))));
        ans = num / den;
    } else {
        double z  = 8.0 / ax;
        double y  = z * z;
        double xx = ax - 2.356194491;
        ans = std::sqrt(0.636619772 / ax) *
              (std::cos(xx) * (1.0 + y * (0.183105e-2
                   + y * (-0.3516396496e-4
                   + y * (0.2457520174e-5 + y * (-0.240337019e-6)))))
             - z * std::sin(xx) * (0.04687499995
                   + y * (-0.2002690873e-3
                   + y * (0.8449199096e-5 + y * (-0.88228987e-6)))));
        if (x < 0.) ans = -ans;
    }
    return ans;
}

inline double fast_bessel_j2(double x) {
    return (2.0 / x) * fast_bessel_j1(x) - fast_bessel_j0(x);
}
inline double fast_bessel_j0p2(double x) {
    return (2.0 / x) * fast_bessel_j1(x);
}
inline double fast_bessel_j0m2(double x) {
    return 2. * fast_bessel_j0(x) - (2.0 / x) * fast_bessel_j1(x);
}

inline double in_bessel(double w, double Sqrt1mkk, double s, double u,
                        double sign) {
    return w * Sqrt1mkk * s * std::sqrt(u * u + sign);
}

// Cutoff function C1: 1 for t < t_cut, 0 for t >= t_max, smooth interpolation
inline double C1(double t, double t_cut, double t_m, double t_0,
                 double t_max, int type) {
    if (t < t_cut)  return 1.;
    if (t >= t_max) return 0.;
    switch (type) {
    case 0: {
        double arg = 2. * (t - t_m) + t_0;
        return std::exp(-arg * arg * std::log(2.) / (t_0 * t_0));
    }
    case 1: {
        double s = std::sin(M_PI * (2. * (t_m - t) + t_0) / (4. * t_0));
        return s * s;
    }
    case 2: {
        double arg = 2. * (t_m - t) + t_0;
        return (t + t_0 - t_m) * arg * arg / (2. * t_0 * t_0 * t_0);
    }
    default: return 0.;
    }
}

// Core integrands (params = {w, Sqrt1mkk, s, sign, t_cut, t_m, t_0, t_max, cutoff_type}).
// These omit the C1 cutoff window; the windowed wrappers below multiply it back in.
// Kept as a single source of truth so the plain (unwindowed) and windowed variants
// used by the incremental-in-time u-integral can never drift apart.
inline double core_xx_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return (u*u+d[3]) * std::cos(d[0]*d[2]*u) * fast_bessel_j0m2(ib);
}
inline double core_xx_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return (u*u+d[3]) * std::sin(d[0]*d[2]*u) * fast_bessel_j0m2(ib);
}
inline double core_yy_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return (u*u+d[3]) * std::cos(d[0]*d[2]*u) * fast_bessel_j0p2(ib);
}
inline double core_yy_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return (u*u+d[3]) * std::sin(d[0]*d[2]*u) * fast_bessel_j0p2(ib);
}
inline double core_zz_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return std::cos(d[0]*d[2]*u) * fast_bessel_j0(ib);
}
inline double core_zz_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return std::sin(d[0]*d[2]*u) * fast_bessel_j0(ib);
}
inline double core_xz_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return d[3] * std::cos(d[0]*d[2]*u) * fast_bessel_j1(ib) * std::sqrt(u*u+d[3]);
}
inline double core_xz_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    double ib = in_bessel(d[0], d[1], d[2], u, d[3]);
    return d[3] * std::sin(d[0]*d[2]*u) * fast_bessel_j1(ib) * std::sqrt(u*u+d[3]);
}

// Windowed variants: core * C1(t_cut, t_m, t_0, t_max, cutoff_type). Used for the
// bootstrap (i_t==0) call and for the transition-window piece of each later step.
double integrand_xx_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_xx_real(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_xx_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_xx_imag(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_yy_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_yy_real(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_yy_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_yy_imag(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_zz_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_zz_real(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_zz_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_zz_imag(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_xz_real(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_xz_real(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}
double integrand_xz_imag(double u, void *p) {
    double *d = static_cast<double *>(p);
    return core_xz_imag(u, p) * C1(d[2]*u, d[4], d[5], d[6], d[7], (int)d[8]);
}

// Plain (unwindowed) variants: used to integrate the C1==1 "plateau" region
// incrementally between consecutive time steps, without paying for the C1
// evaluation (which is identically 1 there anyway).
double integrand_xx_real_plain(double u, void *p) { return core_xx_real(u, p); }
double integrand_xx_imag_plain(double u, void *p) { return core_xx_imag(u, p); }
double integrand_yy_real_plain(double u, void *p) { return core_yy_real(u, p); }
double integrand_yy_imag_plain(double u, void *p) { return core_yy_imag(u, p); }
double integrand_zz_real_plain(double u, void *p) { return core_zz_real(u, p); }
double integrand_zz_imag_plain(double u, void *p) { return core_zz_imag(u, p); }
double integrand_xz_real_plain(double u, void *p) { return core_xz_real(u, p); }
double integrand_xz_imag_plain(double u, void *p) { return core_xz_imag(u, p); }
