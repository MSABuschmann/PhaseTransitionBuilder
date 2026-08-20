#include "integrator.h"
#include "integrand.h"
#include "interpolator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <gsl/gsl_errno.h>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Integrator::Integrator(const std::vector<std::vector<double>> &input_phi,
                       const Setup &setup, int param)
    : n_k_(setup.n_k), n_w_(setup.n_w),
      ds_(setup.ds * setup.how_often_ds),
      dz_(std::abs(setup.z[1] - setup.z[0])),
      t_cut_base_(setup.t_cut), t_m_base_(setup.t_m),
      t_max_base_(setup.t_max), t_0_(setup.t_0),
      d_(setup.d), cutoff_type_(setup.cutoff_type),
      z_(setup.z), wlist_(setup.wlist), times_(setup.times),
      phi_(input_phi),
      gsl_limit_(param <= 0 ? 1000 : param)
{
    n_z_ = z_.size();
    n_s_ = phi_.size();
    slist_ = linspace(0., (n_s_ - 1) * ds_, static_cast<int>(n_s_));
    SetPhi2();

    std::cout << "Integrator: n_z=" << n_z_ << " n_s=" << n_s_
              << " n_w=" << n_w_ << " n_k=" << n_k_
              << "  gsl_limit_=" << gsl_limit_ << "\n"
              << "  t_cut=" << t_cut_base_ << " t_m=" << t_m_base_
              << " t_max=" << t_max_base_ << "\n\n";

    PrecomputeZIntegrals();

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
// The u-integral is accumulated incrementally across calls (see cum_* members
// and integral_u_quad_incremental), so this mutates persisted state and must
// be called with strictly increasing i_t starting at 0.
// ---------------------------------------------------------------------------

std::vector<double> Integrator::Compute(int i_t) {
    if (i_t != last_i_t_processed_ + 1) {
        throw std::runtime_error(
            "Integrator::Compute: i_t must be called in strictly increasing "
            "order starting at 0 (u-integral is accumulated incrementally); "
            "got i_t=" + std::to_string(i_t) +
            " after last_i_t_processed_=" + std::to_string(last_i_t_processed_));
    }

    double shift  = t_m_base_ - times_[i_t];
    double t_cut  = t_cut_base_ - shift;
    double t_m    = t_m_base_  - shift;
    double t_max  = t_max_base_ - shift;
    const double t_cut_prev = t_cut_prev_;

    std::cout << "Compute i_t=" << i_t
              << " shift=" << shift
              << " t_cut=" << t_cut
              << " t_m="   << t_m   << "\n";

    std::vector<double> result(n_w_);
    std::vector<double> eps_rel = geomspace(1e-5, 1e-7, n_w_);
    const bool omega_timings = std::getenv("BM_OMEGA_TIMINGS") != nullptr;

    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        const auto omega_start = std::chrono::steady_clock::now();
        result[i_w] = k_integral(i_w, eps_rel[i_w], t_cut, t_m, t_max, t_cut_prev);
        if (omega_timings) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - omega_start).count();
            std::cout << "Omega timing: i_t=" << i_t
                      << " i_w=" << i_w
                      << " omega=" << wlist_[i_w]
                      << " seconds=" << seconds << "\n";
        }
    }

    t_cut_prev_ = t_cut;
    last_i_t_processed_ = i_t;
    return result;
}

// ---------------------------------------------------------------------------
// ComputeAmplitude: like Compute but retains pre-squaring A(w, cos_theta).
// Same incremental-state / ordering requirement as Compute.
// ---------------------------------------------------------------------------

AmplitudeResult Integrator::ComputeAmplitude(int i_t) {
    if (i_t != last_i_t_processed_ + 1) {
        throw std::runtime_error(
            "Integrator::ComputeAmplitude: i_t must be called in strictly "
            "increasing order starting at 0 (u-integral is accumulated "
            "incrementally); got i_t=" + std::to_string(i_t) +
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

    std::vector<double> eps_rel = geomspace(1e-5, 1e-7, n_w_);

    AmplitudeResult res;
    res.w      = wlist_;
    res.klist  = linspace(0., 1., static_cast<int>(n_k_));
    res.spectrum.resize(n_w_);
    res.amp_re.resize(n_w_ * n_k_, 0.);
    res.amp_im.resize(n_w_ * n_k_, 0.);

    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        std::vector<double> row_re(n_k_, 0.), row_im(n_k_, 0.);
        res.spectrum[i_w] = k_integral(i_w, eps_rel[i_w],
                                       t_cut, t_m, t_max, t_cut_prev,
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
// ---------------------------------------------------------------------------

double Integrator::k_integral(std::size_t i_w, double eps_rel,
                               double t_cut, double t_m, double t_max,
                               double t_cut_prev,
                               std::vector<double> *out_amp_re,
                               std::vector<double> *out_amp_im) {
    const double w   = wlist_[i_w];
    double int_k = 0.;
    auto klist   = linspace(0., 1., static_cast<int>(n_k_));
    const double dk = klist[1] - klist[0];
    std::vector<double> intk(n_k_, 0.);

    for (std::size_t i_k = 0; i_k < n_k_; ++i_k) {
        const double k          = klist[i_k];
        if (k == 1. || k == -1.) continue;
        const double k_sq       = k * k;
        const double Onemkk     = 1. - k_sq;
        const double Sqrt1mkk   = std::sqrt(Onemkk);
        const double TwokSqrt   = 2. * k * Sqrt1mkk;

        // Base index into precomputed z-integral / cumulative-u-integral
        // caches for this (i_w, i_k)
        const std::size_t z_base = (i_w * n_k_ + i_k) * n_s_;

        double int_s_zz_real    = 0., int_s_zz_imag    = 0.;
        double int_s_xandy_real = 0., int_s_xandy_imag = 0.;
        double int_s_xz_real    = 0., int_s_xz_imag    = 0.;

        // Pointers so the (thread-parallel) loops below can read/write this
        // (i_w, i_k) slice of the persisted cumulative state directly.
        double *cum_zz1_re = cum_zz1_re_.data() + z_base, *cum_zz1_im = cum_zz1_im_.data() + z_base;
        double *cum_zz2_re = cum_zz2_re_.data() + z_base, *cum_zz2_im = cum_zz2_im_.data() + z_base;
        double *cum_xx1_re = cum_xx1_re_.data() + z_base, *cum_xx1_im = cum_xx1_im_.data() + z_base;
        double *cum_xx2_re = cum_xx2_re_.data() + z_base, *cum_xx2_im = cum_xx2_im_.data() + z_base;
        double *cum_yy1_re = cum_yy1_re_.data() + z_base, *cum_yy1_im = cum_yy1_im_.data() + z_base;
        double *cum_yy2_re = cum_yy2_re_.data() + z_base, *cum_yy2_im = cum_yy2_im_.data() + z_base;
        double *cum_xz1_re = cum_xz1_re_.data() + z_base, *cum_xz1_im = cum_xz1_im_.data() + z_base;
        double *cum_xz2_re = cum_xz2_re_.data() + z_base, *cum_xz2_im = cum_xz2_im_.data() + z_base;

#pragma omp parallel
        {
            gsl_set_error_handler_off();
            gsl_integration_workspace *ws =
                gsl_integration_workspace_alloc(gsl_limit_);

            gsl_function fr_xx, fi_xx, fr_yy, fi_yy,
                         fr_zz, fi_zz, fr_xz, fi_xz;
            fr_xx.function = &integrand_xx_real;
            fi_xx.function = &integrand_xx_imag;
            fr_yy.function = &integrand_yy_real;
            fi_yy.function = &integrand_yy_imag;
            fr_zz.function = &integrand_zz_real;
            fi_zz.function = &integrand_zz_imag;
            fr_xz.function = &integrand_xz_real;
            fi_xz.function = &integrand_xz_imag;

            gsl_function fr_xx_p, fi_xx_p, fr_yy_p, fi_yy_p,
                         fr_zz_p, fi_zz_p, fr_xz_p, fi_xz_p;
            fr_xx_p.function = &integrand_xx_real_plain;
            fi_xx_p.function = &integrand_xx_imag_plain;
            fr_yy_p.function = &integrand_yy_real_plain;
            fi_yy_p.function = &integrand_yy_imag_plain;
            fr_zz_p.function = &integrand_zz_real_plain;
            fi_zz_p.function = &integrand_zz_imag_plain;
            fr_xz_p.function = &integrand_xz_real_plain;
            fi_xz_p.function = &integrand_xz_imag_plain;

#pragma omp for reduction(+:int_s_zz_real,int_s_zz_imag) schedule(dynamic)
            for (int i_s = 0; i_s < static_cast<int>(slist_.size()); ++i_s) {
                const double s = slist_[i_s];
                if (s == 0.) continue;

                double ur1, ui1, ur2, ui2;
                integral_u_quad_incremental_region1(fr_zz, fi_zz, fr_zz_p, fi_zz_p, ws,
                                        s, Sqrt1mkk, w, eps_rel, ur1, ui1,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_zz1_re[i_s], cum_zz1_im[i_s]);
                integral_u_quad_incremental_region2(fr_zz, fi_zz, fr_zz_p, fi_zz_p, ws,
                                        s, Sqrt1mkk, w, eps_rel, ur2, ui2,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_zz2_re[i_s], cum_zz2_im[i_s]);

                const double iz1 = iz_zz1_[z_base + i_s];
                const double iz2 = iz_zz2_[z_base + i_s];

                const double fac = (i_s == 0 || i_s == static_cast<int>(slist_.size())-1)
                                   ? 0.5 : 1.;
                const double pre = fac * i_s * i_s * ds_ * ds_ * ds_;
                int_s_zz_real += pre * (ur1 * iz1 + ur2 * iz2);
                int_s_zz_imag += pre * (ui1 * iz1 + ui2 * iz2);
            }

#pragma omp for reduction(+:int_s_xandy_real,int_s_xandy_imag) schedule(dynamic)
            for (int i_s = 0; i_s < static_cast<int>(slist_.size()); ++i_s) {
                const double s     = slist_[i_s];
                const double s_off = s - 0.5 * ds_;
                if (s == 0.) continue;

                double urx1, uix1, urx2, uix2;
                integral_u_quad_incremental_region1(fr_xx, fi_xx, fr_xx_p, fi_xx_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, urx1, uix1,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_xx1_re[i_s], cum_xx1_im[i_s]);
                integral_u_quad_incremental_region2(fr_xx, fi_xx, fr_xx_p, fi_xx_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, urx2, uix2,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_xx2_re[i_s], cum_xx2_im[i_s]);
                double ury1, uiy1, ury2, uiy2;
                integral_u_quad_incremental_region1(fr_yy, fi_yy, fr_yy_p, fi_yy_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, ury1, uiy1,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_yy1_re[i_s], cum_yy1_im[i_s]);
                integral_u_quad_incremental_region2(fr_yy, fi_yy, fr_yy_p, fi_yy_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, ury2, uiy2,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_yy2_re[i_s], cum_yy2_im[i_s]);

                const double iz1 = iz_xa1_[z_base + i_s];
                const double iz2 = iz_xa2_[z_base + i_s];

                const double pre = 0.5 * s_off * s_off * ds_;
                int_s_xandy_real += pre * ((urx1*k_sq - ury1)*iz1 + (urx2*k_sq - ury2)*iz2);
                int_s_xandy_imag += pre * ((uix1*k_sq - uiy1)*iz1 + (uix2*k_sq - uiy2)*iz2);
            }

#pragma omp for reduction(+:int_s_xz_real,int_s_xz_imag) schedule(dynamic)
            for (int i_s = 0; i_s < static_cast<int>(slist_.size()); ++i_s) {
                const double s     = slist_[i_s];
                const double s_off = s - 0.5 * ds_;
                if (s == 0.) continue;

                double ur1, ui1, ur2, ui2;
                integral_u_quad_incremental_region1(fr_xz, fi_xz, fr_xz_p, fi_xz_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, ur1, ui1,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_xz1_re[i_s], cum_xz1_im[i_s]);
                integral_u_quad_incremental_region2(fr_xz, fi_xz, fr_xz_p, fi_xz_p, ws,
                                        s_off, Sqrt1mkk, w, eps_rel, ur2, ui2,
                                        t_cut, t_m, t_max, t_cut_prev,
                                        cum_xz2_re[i_s], cum_xz2_im[i_s]);

                const double iz1 = iz_xz1_[z_base + i_s];
                const double iz2 = iz_xz2_[z_base + i_s];

                const double pre = -s_off * s_off * ds_;
                int_s_xz_real += pre * (ur1*iz1 + ur2*iz2);
                int_s_xz_imag += pre * (ui1*iz1 + ui2*iz2);
            }

            gsl_integration_workspace_free(ws);
        } // end omp parallel

        const double re = int_s_zz_real*Onemkk + int_s_xandy_real - TwokSqrt*int_s_xz_real;
        const double im = int_s_zz_imag*Onemkk + int_s_xandy_imag - TwokSqrt*int_s_xz_imag;
        if (out_amp_re) { (*out_amp_re)[i_k] = re; (*out_amp_im)[i_k] = im; }
        intk[i_k] = (re*re + im*im) * w*w*w * 2.*M_PI;

        const double fk = (i_k == 0 || i_k == n_k_-1) ? 1. : 2.;
        int_k += intk[i_k] * dk * fk;
    }
    return int_k;
}

// ---------------------------------------------------------------------------
// GSL wrapper helpers
// ---------------------------------------------------------------------------

double Integrator::u_max(double s, double t_max_local) const {
    return 1. + t_max_local / s;
}

// Incremental replacement for the old integral_u_quad: instead of
// re-integrating the whole [umin_region, u_max] range from scratch every time
// step, split it at the plateau boundary u_cut = t_cut/s (clamped to
// umin_region):
//   - [umin_region, u_cut]: the C1==1 "plateau". f(u) here doesn't depend on
//     i_t at all, so instead of recomputing the whole thing we only integrate
//     the NEW slice since the previous step's u_cut (t_cut_prev/s) and add it
//     to the persisted running total (cum_re/cum_im, read-modify-write).
//   - [u_cut, u_max]: the transition window plus tail, fixed width (~7*t_0/s),
//     just slides in u as t_cut advances. Computed fresh each step with the
//     C1-weighted integrand — cheap since it never grows.
// Mathematically this reproduces exactly what the old single full-range
// C1-weighted integral computed (C1 is exactly 1 below t_cut by construction),
// just without redoing the plateau portion 32 times.
void Integrator::integral_u_quad_incremental(
        gsl_function &fr, gsl_function &fi,
        gsl_function &fr_plain, gsl_function &fi_plain,
        gsl_integration_workspace *ws,
        double s, double Sqrt1mkk, double w, double eps_rel,
        double sign, double umin_region,
        double &real, double &imag,
        double t_cut, double t_m, double t_max,
        double t_cut_prev,
        double &cum_re, double &cum_im) const {
    double args[9] = {w, Sqrt1mkk, s, sign,
                      t_cut, t_m, t_0_, t_max,
                      static_cast<double>(cutoff_type_)};
    fr.params = &args;
    fi.params = &args;
    fr_plain.params = &args;
    fi_plain.params = &args;

    const double plateau_now  = std::max(umin_region, t_cut / s);
    const double plateau_prev = std::max(umin_region, t_cut_prev / s);

    double err;
    if (plateau_now > plateau_prev) {
        double d_re, d_im;
        gsl_integration_qag(&fr_plain, plateau_prev, plateau_now, GSL_EPSABS, eps_rel,
                            gsl_limit_, GSL_KEY, ws, &d_re, &err);
        gsl_integration_qag(&fi_plain, plateau_prev, plateau_now, GSL_EPSABS, eps_rel,
                            gsl_limit_, GSL_KEY, ws, &d_im, &err);
        cum_re += d_re;
        cum_im += d_im;
    }
    // else: plateau boundary hasn't advanced past umin_region yet for this
    // (s, i_t) — nothing new to add; cum_re/cum_im stay at their prior value
    // (0 if the plateau hasn't started at all).

    double w_re = 0., w_im = 0.;
    const double hi = u_max(s, t_max);
    if (hi > plateau_now) {
        gsl_integration_qag(&fr, plateau_now, hi, GSL_EPSABS, eps_rel,
                            gsl_limit_, GSL_KEY, ws, &w_re, &err);
        gsl_integration_qag(&fi, plateau_now, hi, GSL_EPSABS, eps_rel,
                            gsl_limit_, GSL_KEY, ws, &w_im, &err);
    }

    real = cum_re + w_re;
    imag = cum_im + w_im;
}

void Integrator::integral_u_quad_incremental_region1(
        gsl_function &fr, gsl_function &fi,
        gsl_function &fr_plain, gsl_function &fi_plain,
        gsl_integration_workspace *ws,
        double s, double Sqrt1mkk, double w, double eps_rel,
        double &real, double &imag,
        double t_cut, double t_m, double t_max, double t_cut_prev,
        double &cum_re, double &cum_im) const {
    integral_u_quad_incremental(fr, fi, fr_plain, fi_plain, ws,
                                s, Sqrt1mkk, w, eps_rel, -1., 1.,
                                real, imag, t_cut, t_m, t_max, t_cut_prev,
                                cum_re, cum_im);
}

void Integrator::integral_u_quad_incremental_region2(
        gsl_function &fr, gsl_function &fi,
        gsl_function &fr_plain, gsl_function &fi_plain,
        gsl_integration_workspace *ws,
        double s, double Sqrt1mkk, double w, double eps_rel,
        double &real, double &imag,
        double t_cut, double t_m, double t_max, double t_cut_prev,
        double &cum_re, double &cum_im) const {
    integral_u_quad_incremental(fr, fi, fr_plain, fi_plain, ws,
                                s, Sqrt1mkk, w, eps_rel, 1., 0.,
                                real, imag, t_cut, t_m, t_max, t_cut_prev,
                                cum_re, cum_im);
}

// ---------------------------------------------------------------------------
// Precompute z-integrals (independent of t_cut / time step)
// ---------------------------------------------------------------------------

void Integrator::PrecomputeZIntegrals() {
    const std::size_t total = n_w_ * n_k_ * n_s_;
    iz_zz1_.assign(total, 0.);
    iz_zz2_.assign(total, 0.);
    iz_xa1_.assign(total, 0.);
    iz_xa2_.assign(total, 0.);
    iz_xz1_.assign(total, 0.);
    iz_xz2_.assign(total, 0.);

    auto klist = linspace(0., 1., static_cast<int>(n_k_));

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i_w = 0; i_w < static_cast<int>(n_w_); ++i_w) {
        for (int i_k = 0; i_k < static_cast<int>(n_k_); ++i_k) {
            const double w = wlist_[i_w];
            const double k = klist[i_k];
            const std::size_t base = (i_w * n_k_ + i_k) * n_s_;

            for (int i_s = 1; i_s < static_cast<int>(n_s_); ++i_s) {
                double zz1 = 0., zz2 = 0.;
                for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                    zz1 += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi_);
                    zz2 += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi2_);
                }
                iz_zz1_[base + i_s] = zz1;
                iz_zz2_[base + i_s] = zz2;

                double xa1 = 0., xa2 = 0.;
                for (int i_z = 0; i_z < static_cast<int>(n_z_); ++i_z) {
                    const double fz = (i_z == 0 || i_z == static_cast<int>(n_z_)-1) ? 0.5 : 1.;
                    xa1 += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi_);
                    xa2 += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi2_);
                }
                iz_xa1_[base + i_s] = xa1;
                iz_xa2_[base + i_s] = xa2;

                double xz1 = 0., xz2 = 0.;
                for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                    xz1 += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi_);
                    xz2 += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi2_);
                }
                iz_xz1_[base + i_s] = xz1;
                iz_xz2_[base + i_s] = xz2;
            }
        }
    }

    std::cout << "PrecomputeZIntegrals: done (" << total << " entries)\n\n";
}

// ---------------------------------------------------------------------------
// Two-bubble reference field
// ---------------------------------------------------------------------------

void Integrator::SetPhi2() {
    // Find peak of initial snapshot
    auto max_it = std::max_element(phi_[0].begin(), phi_[0].end());
    int  phimid = static_cast<int>(max_it - phi_[0].begin());

    std::vector<double> z_new(z_.begin() + phimid, z_.end());
    double z0 = z_new[0];
    for (double &zi : z_new) zi -= z0;

    std::vector<double> phi0_new(phi_[0].begin() + phimid, phi_[0].end());
    Interpolator phi0_interp(z_new, phi0_new);

    phi2_.resize(n_s_);
    for (std::size_t i_s = 0; i_s < n_s_; ++i_s) {
        phi2_[i_s].resize(n_z_);
        for (std::size_t i_z = 0; i_z < n_z_; ++i_z) {
            double s_val = i_s * ds_;
            double z_val = i_z * dz_;
            double r1 = std::sqrt(s_val*s_val + std::pow(z_val - d_/2., 2));
            double r2 = std::sqrt(s_val*s_val + std::pow(z_val + d_/2., 2));
            phi2_[i_s][i_z] = phi0_interp(r1) + phi0_interp(r2);
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::vector<double> Integrator::linspace(double a, double b, int n) {
    std::vector<double> v(n);
    double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) v[i] = a + i * step;
    return v;
}

std::vector<double> Integrator::geomspace(double a, double b, std::size_t n) {
    if (n == 0) return {};
    std::vector<double> v(n);
    double la = std::log10(a), lb = std::log10(b);
    double step = (lb - la) / (n - 1);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::pow(10., la + i * step);
    return v;
}
