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
              << "  N=adaptive(64 panels/osc, floor=" << n_min_ << ") split at t_cut/s\n"
              << "  phi_ flat layout: "
              << (n_s_ * n_z_ * 8) / (1 << 20) << " MB per array\n"
              << "  t_cut=" << t_cut_base_ << " t_m=" << t_m_base_
              << " t_max=" << t_max_base_ << "\n\n";

    const std::size_t total = n_w_ * n_k_ * n_s_;
    cum_zz1_re_.assign(total, 0.); cum_zz1_im_.assign(total, 0.);
    cum_zz2_re_.assign(total, 0.); cum_zz2_im_.assign(total, 0.);
    cum_xx1_re_.assign(total, 0.); cum_xx1_im_.assign(total, 0.);
    cum_xx2_re_.assign(total, 0.); cum_xx2_im_.assign(total, 0.);
    cum_yy1_re_.assign(total, 0.); cum_yy1_im_.assign(total, 0.);
    cum_yy2_re_.assign(total, 0.); cum_yy2_im_.assign(total, 0.);
    cum_xz1_re_.assign(total, 0.); cum_xz1_im_.assign(total, 0.);
    cum_xz2_re_.assign(total, 0.); cum_xz2_im_.assign(total, 0.);
}

// ---------------------------------------------------------------------------
// Compute for one time index
//
// The plateau portion of the u-integral is accumulated incrementally across
// calls (see cum_* members and integral_u_filon_incremental), so this
// mutates persisted state and must be called with strictly increasing i_t
// starting at 0.
// ---------------------------------------------------------------------------

std::vector<double> FilonIntegrator::Compute(int i_t) {
    if (i_t != last_i_t_processed_ + 1) {
        throw std::runtime_error(
            "FilonIntegrator::Compute: i_t must be called in strictly "
            "increasing order starting at 0 (plateau u-integral is "
            "accumulated incrementally); got i_t=" + std::to_string(i_t) +
            " after last_i_t_processed_=" + std::to_string(last_i_t_processed_));
    }

    double shift = t_m_base_ - times_[i_t];
    double t_cut = t_cut_base_ - shift;
    double t_m   = t_m_base_   - shift;
    double t_max = t_max_base_ - shift;
    const double t_cut_prev = t_cut_prev_;

    std::cout << "Compute i_t=" << i_t
              << " shift=" << shift
              << " t_cut=" << t_cut
              << " t_m="   << t_m << "\n";

    std::vector<double> result(n_w_);
    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        result[i_w] = k_integral(i_w, t_cut, t_m, t_max, t_cut_prev);
    }

    t_cut_prev_ = t_cut;
    last_i_t_processed_ = i_t;
    return result;
}

// ---------------------------------------------------------------------------
// Filon u-integral: all four stress-tensor components simultaneously.
//
// Incremental version: the plateau [umin, u_split] doesn't depend on i_t
// (C1==1 there identically, since t=s*u < t_cut throughout), so instead of
// re-running Filon over the whole growing [umin, u_split] every time step, we
// only integrate the new slice since the previous step's u_split
// (t_cut_prev/s) and add it into the persisted cum_* state. The transition
// window [u_split, u_top] has fixed width (set by t_max-t_cut) and just
// slides in u as t_cut advances, so it's recomputed fresh each step — same
// as before, just no longer paying for the plateau's ever-growing range.
// ---------------------------------------------------------------------------

void FilonIntegrator::integral_u_filon_incremental(
        double s, double Sqrt1mkk, double w,
        double sign, double umin,
        double t_cut, double t_m, double t_max, double t_cut_prev,
        double &cum_zz_r, double &cum_zz_i,
        double &cum_xx_r, double &cum_xx_i,
        double &cum_yy_r, double &cum_yy_i,
        double &cum_xz_r, double &cum_xz_i,
        double &zz_r, double &zz_i,
        double &xx_r, double &xx_i,
        double &yy_r, double &yy_i,
        double &xz_r, double &xz_i) const {
    // C1 = 0 for t = s*u >= t_max → integrand is zero above u = t_max/s.
    const double u_top   = t_max / s;
    // C1 has a C¹ but not C² junction at t = t_cut (second derivative jumps
    // where the flat-1 region meets the smooth formula).  Splitting at the
    // corresponding u = t_cut/s restores O(h⁴) Filon convergence on each
    // sub-interval instead of O(h²) for the unsplit integral.
    const double u_split = t_cut / s;

    if (u_top <= umin) {
        // Entire interval lies in the dead zone — cum_* is untouched (stays
        // at whatever it already was, 0 if this s has never left the dead
        // zone, which is the only way it can be exactly here since u_top
        // only grows with i_t).
        zz_r = cum_zz_r; zz_i = cum_zz_i;
        xx_r = cum_xx_r; xx_i = cum_xx_i;
        yy_r = cum_yy_r; yy_i = cum_yy_i;
        xz_r = cum_xz_r; xz_i = cum_xz_i;
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
        const double u2s_a = a * a + sign;
        const double ib_a  = (u2s_a > 0.) ? w * Sqrt1mkk * s * std::sqrt(u2s_a) : 0.;
        const double u2s_b = b * b + sign;
        const double ib_b  = (u2s_b > 0.) ? w * Sqrt1mkk * s * std::sqrt(u2s_b) : 0.;
        // N scales with the number of Bessel oscillations in this sub-interval
        // (ib_b - ib_a) rather than the absolute ib_b at the endpoint, so each
        // segment is resolved at ~64 panels/oscillation regardless of where it sits.
        int N = std::max(n_min_, (int)(64.0 * (ib_b - ib_a) / (2.0 * M_PI)) + 2);
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

    const double plateau_now  = std::max(umin, u_split);
    const double plateau_prev = std::max(umin, t_cut_prev / s);

    if (plateau_now > plateau_prev) {
        double d_zz_r, d_zz_i, d_xx_r, d_xx_i, d_yy_r, d_yy_i, d_xz_r, d_xz_i;
        run_segment(plateau_prev, plateau_now,
                    d_zz_r, d_zz_i, d_xx_r, d_xx_i, d_yy_r, d_yy_i, d_xz_r, d_xz_i);
        cum_zz_r += d_zz_r; cum_zz_i += d_zz_i;
        cum_xx_r += d_xx_r; cum_xx_i += d_xx_i;
        cum_yy_r += d_yy_r; cum_yy_i += d_yy_i;
        cum_xz_r += d_xz_r; cum_xz_i += d_xz_i;
    }

    double w_zz_r = 0., w_zz_i = 0., w_xx_r = 0., w_xx_i = 0.;
    double w_yy_r = 0., w_yy_i = 0., w_xz_r = 0., w_xz_i = 0.;
    if (u_top > plateau_now) {
        run_segment(plateau_now, u_top,
                    w_zz_r, w_zz_i, w_xx_r, w_xx_i, w_yy_r, w_yy_i, w_xz_r, w_xz_i);
    }

    zz_r = cum_zz_r + w_zz_r;  zz_i = cum_zz_i + w_zz_i;
    xx_r = cum_xx_r + w_xx_r;  xx_i = cum_xx_i + w_xx_i;
    yy_r = cum_yy_r + w_yy_r;  yy_i = cum_yy_i + w_yy_i;
    xz_r = cum_xz_r + w_xz_r;  xz_i = cum_xz_i + w_xz_i;
}

// ---------------------------------------------------------------------------
// ComputeAmplitude: like Compute but retains pre-squaring A(w, cos_theta)
// ---------------------------------------------------------------------------

AmplitudeResult FilonIntegrator::ComputeAmplitude(int i_t) {
    if (i_t != last_i_t_processed_ + 1) {
        throw std::runtime_error(
            "FilonIntegrator::ComputeAmplitude: i_t must be called in "
            "strictly increasing order starting at 0 (plateau u-integral is "
            "accumulated incrementally); got i_t=" + std::to_string(i_t) +
            " after last_i_t_processed_=" + std::to_string(last_i_t_processed_));
    }

    double shift = t_m_base_ - times_[i_t];
    double t_cut = t_cut_base_ - shift;
    double t_m   = t_m_base_  - shift;
    double t_max = t_max_base_ - shift;
    const double t_cut_prev = t_cut_prev_;

    std::cout << "ComputeAmplitude i_t=" << i_t
              << " shift=" << shift
              << " t_cut=" << t_cut
              << " t_m="   << t_m   << "\n";

    AmplitudeResult res;
    res.w      = wlist_;
    res.klist  = linspace(0., 1., static_cast<int>(n_k_));
    res.spectrum.resize(n_w_);
    res.amp_re.resize(n_w_ * n_k_, 0.);
    res.amp_im.resize(n_w_ * n_k_, 0.);

    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        std::vector<double> row_re(n_k_, 0.), row_im(n_k_, 0.);
        res.spectrum[i_w] = k_integral(i_w, t_cut, t_m, t_max, t_cut_prev,
                                       &row_re, &row_im);
        std::copy(row_re.begin(), row_re.end(),
                  res.amp_re.begin() + i_w * n_k_);
        std::copy(row_im.begin(), row_im.end(),
                  res.amp_im.begin() + i_w * n_k_);
    }

    t_cut_prev_ = t_cut;
    last_i_t_processed_ = i_t;
    return res;
}

// ---------------------------------------------------------------------------
// k integral
// schedule(static) matches the first-touch schedule so the s-loop accesses
// NUMA-local rows.
// ---------------------------------------------------------------------------

double FilonIntegrator::k_integral(std::size_t i_w,
                                    double t_cut, double t_m, double t_max,
                                    double t_cut_prev,
                                    std::vector<double> *out_amp_re,
                                    std::vector<double> *out_amp_im) {
    const double w = wlist_[i_w];
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

        // Base index into the persisted cumulative-plateau cache for this (i_w, i_k)
        const std::size_t z_base = (i_w * n_k_ + i_k) * n_s_;

        double int_s_zz_real    = 0., int_s_zz_imag    = 0.;
        double int_s_xandy_real = 0., int_s_xandy_imag = 0.;
        double int_s_xz_real    = 0., int_s_xz_imag    = 0.;

        double *cum_zz1_re = cum_zz1_re_.data() + z_base, *cum_zz1_im = cum_zz1_im_.data() + z_base;
        double *cum_zz2_re = cum_zz2_re_.data() + z_base, *cum_zz2_im = cum_zz2_im_.data() + z_base;
        double *cum_xx1_re = cum_xx1_re_.data() + z_base, *cum_xx1_im = cum_xx1_im_.data() + z_base;
        double *cum_xx2_re = cum_xx2_re_.data() + z_base, *cum_xx2_im = cum_xx2_im_.data() + z_base;
        double *cum_yy1_re = cum_yy1_re_.data() + z_base, *cum_yy1_im = cum_yy1_im_.data() + z_base;
        double *cum_yy2_re = cum_yy2_re_.data() + z_base, *cum_yy2_im = cum_yy2_im_.data() + z_base;
        double *cum_xz1_re = cum_xz1_re_.data() + z_base, *cum_xz1_im = cum_xz1_im_.data() + z_base;
        double *cum_xz2_re = cum_xz2_re_.data() + z_base, *cum_xz2_im = cum_xz2_im_.data() + z_base;

#pragma omp parallel for schedule(static) \
    reduction(+:int_s_zz_real, int_s_zz_imag, \
                int_s_xandy_real, int_s_xandy_imag, \
                int_s_xz_real, int_s_xz_imag)
        for (int i_s = 0; i_s < static_cast<int>(n_s_); ++i_s) {
            const double s     = slist_[i_s];
            const double s_off = s - 0.5 * ds_;
            if (s == 0.) continue;

            // u-integrals via Filon (incremental — see integral_u_filon_incremental).
            // Each call computes all four (zz,xx,yy,xz) stress-tensor components
            // together (Filon's cost is bundled per evaluation point), but only
            // zz is used from the "s" calls and only xx/yy/xz from the "s_off"
            // calls below (matches the original code, which also discarded the
            // other half of each call's output). The unused outputs' cum_* state
            // is therefore intentionally NOT persisted — fresh throwaway locals
            // (reset to 0 each call) stand in for it, since a call that starts
            // from 0 every time and whose result is discarded can't corrupt
            // anything, and reusing a persisted slot across two different
            // physical points (s vs s_off) here would corrupt real state.
            double zz_r1, zz_i1, xx_r1, xx_i1, yy_r1, yy_i1, xz_r1, xz_i1;
            double zz_r2, zz_i2, xx_r2, xx_i2, yy_r2, yy_i2, xz_r2, xz_i2;
            double junk1_re = 0., junk1_im = 0., junk2_re = 0., junk2_im = 0.;
            double junk3_re = 0., junk3_im = 0.;
            integral_u_filon_incremental(s, Sqrt1mkk, w, -1., 1.,
                             t_cut, t_m, t_max, t_cut_prev,
                             cum_zz1_re[i_s], cum_zz1_im[i_s],
                             junk1_re, junk1_im, junk2_re, junk2_im, junk3_re, junk3_im,
                             zz_r1, zz_i1, xx_r1, xx_i1, yy_r1, yy_i1, xz_r1, xz_i1);
            junk1_re = junk1_im = junk2_re = junk2_im = junk3_re = junk3_im = 0.;
            integral_u_filon_incremental(s, Sqrt1mkk, w, +1., 0.,
                             t_cut, t_m, t_max, t_cut_prev,
                             cum_zz2_re[i_s], cum_zz2_im[i_s],
                             junk1_re, junk1_im, junk2_re, junk2_im, junk3_re, junk3_im,
                             zz_r2, zz_i2, xx_r2, xx_i2, yy_r2, yy_i2, xz_r2, xz_i2);

            double zzx_r1, zzx_i1, xx_off_r1, xx_off_i1, yy_off_r1, yy_off_i1, xz_off_r1, xz_off_i1;
            double zzx_r2, zzx_i2, xx_off_r2, xx_off_i2, yy_off_r2, yy_off_i2, xz_off_r2, xz_off_i2;
            double junk_zz_re = 0., junk_zz_im = 0.;
            integral_u_filon_incremental(s_off, Sqrt1mkk, w, -1., 1.,
                             t_cut, t_m, t_max, t_cut_prev,
                             junk_zz_re, junk_zz_im,
                             cum_xx1_re[i_s], cum_xx1_im[i_s],
                             cum_yy1_re[i_s], cum_yy1_im[i_s],
                             cum_xz1_re[i_s], cum_xz1_im[i_s],
                             zzx_r1, zzx_i1, xx_off_r1, xx_off_i1, yy_off_r1, yy_off_i1, xz_off_r1, xz_off_i1);
            junk_zz_re = 0.; junk_zz_im = 0.;
            integral_u_filon_incremental(s_off, Sqrt1mkk, w, +1., 0.,
                             t_cut, t_m, t_max, t_cut_prev,
                             junk_zz_re, junk_zz_im,
                             cum_xx2_re[i_s], cum_xx2_im[i_s],
                             cum_yy2_re[i_s], cum_yy2_im[i_s],
                             cum_xz2_re[i_s], cum_xz2_im[i_s],
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
        if (out_amp_re) { (*out_amp_re)[i_k] = re; (*out_amp_im)[i_k] = im; }
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
