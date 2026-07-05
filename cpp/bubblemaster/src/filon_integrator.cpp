#include "filon_integrator.h"
#include "filon.h"
#include "integrand.h"
#include "interpolator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <omp.h>

// ---------------------------------------------------------------------------
// Constructor
// Copies the field snapshots into a flat NUMA-friendly layout using a
// parallel first-touch loop with schedule(static), matching the schedule
// used in k_integral so each thread primarily accesses its own pages.
// ---------------------------------------------------------------------------

FilonIntegrator::FilonIntegrator(const std::vector<std::vector<double>> &input_phi,
                                 const Setup &setup, int param)
    : n_k_(setup.n_k), n_w_(setup.n_w),
      ds_(setup.ds * setup.how_often_ds),
      dz_(std::abs(setup.z[1] - setup.z[0])),
      t_cut_base_(setup.t_cut), t_m_base_(setup.t_m),
      t_max_base_(setup.t_max), t_0_(setup.t_0),
      d_(setup.d), cutoff_type_(setup.cutoff_type),
      z_(setup.z), wlist_(setup.wlist), times_(setup.times),
      n_min_(param <= 0 ? FILON_N_MIN : (param + 1) & ~1)
{
    n_z_ = z_.size();
    n_s_ = input_phi.size();
    slist_ = linspace(0., (n_s_ - 1) * ds_, static_cast<int>(n_s_));

    // First-touch copy: each thread writes the rows it will later read.
    phi_.resize(n_s_ * n_z_);
#pragma omp parallel for schedule(static)
    for (int i_s = 0; i_s < static_cast<int>(n_s_); ++i_s)
        for (int i_z = 0; i_z < static_cast<int>(n_z_); ++i_z)
            phi_[i_s * n_z_ + i_z] = input_phi[i_s][i_z];

    SetPhi2();

    std::cout << "FilonIntegrator: n_z=" << n_z_ << " n_s=" << n_s_
              << " n_w=" << n_w_ << " n_k=" << n_k_
              << "  FILON_N_MIN=" << n_min_ << " per segment (split at t_cut/s)\n"
              << "  phi_ flat layout: "
              << (n_s_ * n_z_ * 8) / (1 << 20) << " MB per array\n"
              << "  t_cut=" << t_cut_base_ << " t_m=" << t_m_base_
              << " t_max=" << t_max_base_ << "\n\n";
}

// ---------------------------------------------------------------------------
// Compute for one time index
// ---------------------------------------------------------------------------

std::vector<double> FilonIntegrator::Compute(int i_t) const {
    double shift = t_m_base_ - times_[i_t];
    double t_cut = t_cut_base_ - shift;
    double t_m   = t_m_base_   - shift;
    double t_max = t_max_base_ - shift;

    std::cout << "Compute i_t=" << i_t
              << " shift=" << shift
              << " t_cut=" << t_cut
              << " t_m="   << t_m << "\n";

    std::vector<double> result(n_w_);
    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        result[i_w] = k_integral(wlist_[i_w], t_cut, t_m, t_max);
        std::cout << "  w=" << wlist_[i_w]
                  << " spec=" << result[i_w] << "\n";
    }
    return result;
}

// ---------------------------------------------------------------------------
// Filon u-integral: all four stress-tensor components simultaneously
// ---------------------------------------------------------------------------

void FilonIntegrator::integral_u_filon(double s, double Sqrt1mkk, double w,
                                        double sign, double umin,
                                        double t_cut, double t_m, double t_max,
                                        double &zz_r, double &zz_i,
                                        double &xx_r, double &xx_i,
                                        double &yy_r, double &yy_i,
                                        double &xz_r, double &xz_i) const {
    // C1 = 0 for t = s*u >= t_max → integrand is zero above u = t_max/s.
    // Truncating at u_top avoids wasting panels in the dead zone.
    const double u_top   = t_max / s;
    // C1 has a C¹ but not C² junction at t = t_cut (second derivative jumps
    // where the flat-1 region meets the smooth formula).  Splitting at the
    // corresponding u = t_cut/s restores O(h⁴) Filon convergence on each
    // sub-interval instead of O(h²) for the unsplit integral.
    const double u_split = t_cut / s;

    if (u_top <= umin) {
        // Entire interval lies in the dead zone.
        zz_r = zz_i = xx_r = xx_i = yy_r = yy_i = xz_r = xz_i = 0.;
        return;
    }

    const double omega = w * s;

    // Core Filon evaluation on [a, b].
    auto run_segment = [&](double a, double b,
                           double &r_zz, double &i_zz,
                           double &r_xx, double &i_xx,
                           double &r_yy, double &i_yy,
                           double &r_xz, double &i_xz) {
        if (a >= b) {
            r_zz = i_zz = r_xx = i_xx = r_yy = i_yy = r_xz = i_xz = 0.;
            return;
        }
        const double u2s_b = b * b + sign;
        const double ib_b  = (u2s_b > 0.) ? w * Sqrt1mkk * s * std::sqrt(u2s_b) : 0.;
        int N = std::max(n_min_, (int)(8.0 * ib_b / (2.0 * M_PI)) + 2);
        N = (N + 1) & ~1;

        const double h = (b - a) / N;

        std::vector<double> g_zz(N + 1), g_xx(N + 1), g_yy(N + 1), g_xz(N + 1);
        for (int i = 0; i <= N; ++i) {
            const double u       = a + i * h;
            const double u2s     = u * u + sign;
            const double u2s_pos = (u2s > 0.) ? u2s : 0.;
            const double ib      = w * Sqrt1mkk * s * std::sqrt(u2s_pos);
            const double c1      = C1(s * u, t_cut, t_m, t_0_, t_max, cutoff_type_);

            const double bj0 = fast_bessel_j0(ib);
            const double bj1 = fast_bessel_j1(ib);
            double bj0m2, bj0p2;
            if (ib < 1e-14) {
                bj0m2 = 1.0;
                bj0p2 = 1.0;
            } else {
                const double two_over_ib = 2.0 / ib;
                bj0m2 = 2.0 * bj0 - two_over_ib * bj1;
                bj0p2 = two_over_ib * bj1;
            }

            g_zz[i] = bj0  * c1;
            g_xx[i] = u2s  * bj0m2 * c1;
            g_yy[i] = u2s  * bj0p2 * c1;
            g_xz[i] = sign * std::sqrt(u2s_pos) * bj1 * c1;
        }

        filon_cos_sin(g_zz.data(), N, a, h, omega, r_zz, i_zz);
        filon_cos_sin(g_xx.data(), N, a, h, omega, r_xx, i_xx);
        filon_cos_sin(g_yy.data(), N, a, h, omega, r_yy, i_yy);
        filon_cos_sin(g_xz.data(), N, a, h, omega, r_xz, i_xz);
    };

    if (u_split > umin && u_split < u_top) {
        double lo_zz_r, lo_zz_i, lo_xx_r, lo_xx_i, lo_yy_r, lo_yy_i, lo_xz_r, lo_xz_i;
        double hi_zz_r, hi_zz_i, hi_xx_r, hi_xx_i, hi_yy_r, hi_yy_i, hi_xz_r, hi_xz_i;
        run_segment(umin,    u_split, lo_zz_r, lo_zz_i, lo_xx_r, lo_xx_i, lo_yy_r, lo_yy_i, lo_xz_r, lo_xz_i);
        run_segment(u_split, u_top,   hi_zz_r, hi_zz_i, hi_xx_r, hi_xx_i, hi_yy_r, hi_yy_i, hi_xz_r, hi_xz_i);
        zz_r = lo_zz_r + hi_zz_r;  zz_i = lo_zz_i + hi_zz_i;
        xx_r = lo_xx_r + hi_xx_r;  xx_i = lo_xx_i + hi_xx_i;
        yy_r = lo_yy_r + hi_yy_r;  yy_i = lo_yy_i + hi_yy_i;
        xz_r = lo_xz_r + hi_xz_r;  xz_i = lo_xz_i + hi_xz_i;
    } else {
        run_segment(umin, u_top, zz_r, zz_i, xx_r, xx_i, yy_r, yy_i, xz_r, xz_i);
    }
}

// ---------------------------------------------------------------------------
// k integral
// schedule(static) matches the first-touch schedule so the s-loop accesses
// NUMA-local rows.
// ---------------------------------------------------------------------------

double FilonIntegrator::k_integral(double w,
                                    double t_cut, double t_m, double t_max) const {
    double int_k = 0.;
    auto klist   = linspace(0., 1., static_cast<int>(n_k_));
    const double dk = klist[1] - klist[0];
    std::vector<double> intk(n_k_, 0.);

    for (std::size_t i_k = 0; i_k < n_k_; ++i_k) {
        const double k        = klist[i_k];
        if (k == 1. || k == -1.) continue;
        const double k_sq     = k * k;
        const double Onemkk   = 1. - k_sq;
        const double Sqrt1mkk = std::sqrt(Onemkk);
        const double TwokSqrt = 2. * k * Sqrt1mkk;

        double int_s_zz_real    = 0., int_s_zz_imag    = 0.;
        double int_s_xandy_real = 0., int_s_xandy_imag = 0.;
        double int_s_xz_real    = 0., int_s_xz_imag    = 0.;

#pragma omp parallel for schedule(static) \
    reduction(+:int_s_zz_real, int_s_zz_imag, \
                int_s_xandy_real, int_s_xandy_imag, \
                int_s_xz_real, int_s_xz_imag)
        for (int i_s = 0; i_s < static_cast<int>(n_s_); ++i_s) {
            const double s     = slist_[i_s];
            const double s_off = s - 0.5 * ds_;
            if (s == 0.) continue;

            // u-integrals via Filon
            double zz_r1, zz_i1, xx_r1, xx_i1, yy_r1, yy_i1, xz_r1, xz_i1;
            double zz_r2, zz_i2, xx_r2, xx_i2, yy_r2, yy_i2, xz_r2, xz_i2;
            integral_u_filon(s, Sqrt1mkk, w, -1., 1.,
                             t_cut, t_m, t_max,
                             zz_r1, zz_i1, xx_r1, xx_i1, yy_r1, yy_i1, xz_r1, xz_i1);
            integral_u_filon(s, Sqrt1mkk, w, +1., 0.,
                             t_cut, t_m, t_max,
                             zz_r2, zz_i2, xx_r2, xx_i2, yy_r2, yy_i2, xz_r2, xz_i2);

            double zzx_r1, zzx_i1, xx_off_r1, xx_off_i1, yy_off_r1, yy_off_i1, xz_off_r1, xz_off_i1;
            double zzx_r2, zzx_i2, xx_off_r2, xx_off_i2, yy_off_r2, yy_off_i2, xz_off_r2, xz_off_i2;
            integral_u_filon(s_off, Sqrt1mkk, w, -1., 1.,
                             t_cut, t_m, t_max,
                             zzx_r1, zzx_i1, xx_off_r1, xx_off_i1, yy_off_r1, yy_off_i1, xz_off_r1, xz_off_i1);
            integral_u_filon(s_off, Sqrt1mkk, w, +1., 0.,
                             t_cut, t_m, t_max,
                             zzx_r2, zzx_i2, xx_off_r2, xx_off_i2, yy_off_r2, yy_off_i2, xz_off_r2, xz_off_i2);

            // z-integrals — all reads are NUMA-local with schedule(static)
            double iz1_zz = 0., iz2_zz = 0.;
            for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                iz1_zz += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi_);
                iz2_zz += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi2_);
            }

            double iz1_xa = 0., iz2_xa = 0.;
            for (int i_z = 0; i_z < static_cast<int>(n_z_); ++i_z) {
                const double fz = (i_z == 0 || i_z == static_cast<int>(n_z_) - 1) ? 0.5 : 1.;
                iz1_xa += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi_);
                iz2_xa += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi2_);
            }

            double iz1_xz = 0., iz2_xz = 0.;
            for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                iz1_xz += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi_);
                iz2_xz += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi2_);
            }

            // Accumulate
            const double fac    = (i_s == 0 || i_s == static_cast<int>(n_s_) - 1)
                                  ? 0.5 : 1.;
            const double pre_zz = fac * i_s * i_s * ds_ * ds_ * ds_;
            int_s_zz_real += pre_zz * (zz_r1 * iz1_zz + zz_r2 * iz2_zz);
            int_s_zz_imag += pre_zz * (zz_i1 * iz1_zz + zz_i2 * iz2_zz);

            const double pre_xa = 0.5 * s_off * s_off * ds_;
            int_s_xandy_real += pre_xa * ((xx_off_r1*k_sq - yy_off_r1)*iz1_xa
                                        + (xx_off_r2*k_sq - yy_off_r2)*iz2_xa);
            int_s_xandy_imag += pre_xa * ((xx_off_i1*k_sq - yy_off_i1)*iz1_xa
                                        + (xx_off_i2*k_sq - yy_off_i2)*iz2_xa);

            const double pre_xz = -s_off * s_off * ds_;
            int_s_xz_real += pre_xz * (xz_off_r1 * iz1_xz + xz_off_r2 * iz2_xz);
            int_s_xz_imag += pre_xz * (xz_off_i1 * iz1_xz + xz_off_i2 * iz2_xz);
        }

        const double re = int_s_zz_real*Onemkk + int_s_xandy_real - TwokSqrt*int_s_xz_real;
        const double im = int_s_zz_imag*Onemkk + int_s_xandy_imag - TwokSqrt*int_s_xz_imag;
        intk[i_k] = (re*re + im*im) * w*w*w * 2.*M_PI;

        const double fk = (i_k == 0 || i_k == n_k_ - 1) ? 1. : 2.;
        int_k += intk[i_k] * dk * fk;
    }
    return int_k;
}

// ---------------------------------------------------------------------------
// Two-bubble reference field — parallel first-touch with schedule(static)
// so phi2_ pages land on the same NUMA domains as the matching phi_ rows.
// ---------------------------------------------------------------------------

void FilonIntegrator::SetPhi2() {
    // Find peak of initial snapshot (sequential — cheap)
    auto max_it = std::max_element(phi_.begin(), phi_.begin() + n_z_);
    int  phimid = static_cast<int>(max_it - phi_.begin());

    std::vector<double> z_new(z_.begin() + phimid, z_.end());
    double z0 = z_new[0];
    for (double &zi : z_new) zi -= z0;

    std::vector<double> phi0_new(phi_.begin() + phimid, phi_.begin() + n_z_);
    Interpolator phi0_interp(z_new, phi0_new);

    phi2_.resize(n_s_ * n_z_);

#pragma omp parallel for schedule(static)
    for (int i_s = 0; i_s < static_cast<int>(n_s_); ++i_s) {
        const double s_val = i_s * ds_;
        for (int i_z = 0; i_z < static_cast<int>(n_z_); ++i_z) {
            const double z_val = i_z * dz_;
            const double r1 = std::sqrt(s_val*s_val + (z_val - d_/2.)*(z_val - d_/2.));
            const double r2 = std::sqrt(s_val*s_val + (z_val + d_/2.)*(z_val + d_/2.));
            phi2_[i_s * n_z_ + i_z] = phi0_interp(r1) + phi0_interp(r2);
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::vector<double> FilonIntegrator::linspace(double a, double b, int n) {
    std::vector<double> v(n);
    double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) v[i] = a + i * step;
    return v;
}

std::vector<double> FilonIntegrator::geomspace(double a, double b, std::size_t n) {
    if (n == 0) return {};
    std::vector<double> v(n);
    double la = std::log10(a), lb = std::log10(b);
    double step = (lb - la) / (n - 1);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::pow(10., la + i * step);
    return v;
}
