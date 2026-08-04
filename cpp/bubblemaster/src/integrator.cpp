#include "integrator.h"
#include "integrand.h"
#include "interpolator.h"

#include <algorithm>
#include <cmath>
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
}

// ---------------------------------------------------------------------------
// Compute for one time index (non-destructive)
// ---------------------------------------------------------------------------

std::vector<double> Integrator::Compute(int i_t) const {
    // Apply time shift (local copy - does not modify member state)
    double shift  = t_m_base_ - times_[i_t];
    double t_cut  = t_cut_base_ - shift;
    double t_m    = t_m_base_  - shift;
    double t_max  = t_max_base_ - shift;

    std::cout << "Compute i_t=" << i_t
              << " shift=" << shift
              << " t_cut=" << t_cut
              << " t_m="   << t_m   << "\n";

    std::vector<double> result(n_w_);
    std::vector<double> eps_rel = geomspace(1e-5, 1e-7, n_w_);

    for (std::size_t i_w = 0; i_w < n_w_; ++i_w) {
        result[i_w] = k_integral(wlist_[i_w], eps_rel[i_w], t_cut, t_m, t_max);
    }
    return result;
}

// ---------------------------------------------------------------------------
// ComputeAmplitude: like Compute but retains pre-squaring A(w, cos_theta)
// ---------------------------------------------------------------------------

AmplitudeResult Integrator::ComputeAmplitude(int i_t) const {
    double shift = t_m_base_ - times_[i_t];
    double t_cut = t_cut_base_ - shift;
    double t_m   = t_m_base_  - shift;
    double t_max = t_max_base_ - shift;

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
        res.spectrum[i_w] = k_integral(wlist_[i_w], eps_rel[i_w],
                                       t_cut, t_m, t_max,
                                       &row_re, &row_im);
        std::copy(row_re.begin(), row_re.end(),
                  res.amp_re.begin() + i_w * n_k_);
        std::copy(row_im.begin(), row_im.end(),
                  res.amp_im.begin() + i_w * n_k_);
    }
    return res;
}

// ---------------------------------------------------------------------------
// k integral
// ---------------------------------------------------------------------------

double Integrator::k_integral(double w, double eps_rel,
                               double t_cut, double t_m, double t_max,
                               std::vector<double> *out_amp_re,
                               std::vector<double> *out_amp_im) const {
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

        double int_s_zz_real    = 0., int_s_zz_imag    = 0.;
        double int_s_xandy_real = 0., int_s_xandy_imag = 0.;
        double int_s_xz_real    = 0., int_s_xz_imag    = 0.;

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

#pragma omp for reduction(+:int_s_zz_real,int_s_zz_imag) schedule(dynamic)
            for (int i_s = 0; i_s < static_cast<int>(slist_.size()); ++i_s) {
                const double s = slist_[i_s];
                if (s == 0.) continue;

                double ur1, ui1, ur2, ui2;
                integral_u_quad_region1(fr_zz, fi_zz, ws, s, Sqrt1mkk, w,
                                        eps_rel, ur1, ui1, t_cut, t_m, t_max);
                integral_u_quad_region2(fr_zz, fi_zz, ws, s, Sqrt1mkk, w,
                                        eps_rel, ur2, ui2, t_cut, t_m, t_max);

                double iz1 = 0., iz2 = 0.;
                for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                    iz1 += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi_);
                    iz2 += integral_dz_zz(k, w, i_z, i_s, z_[i_z], phi2_);
                }

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
                integral_u_quad_region1(fr_xx, fi_xx, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, urx1, uix1, t_cut, t_m, t_max);
                integral_u_quad_region2(fr_xx, fi_xx, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, urx2, uix2, t_cut, t_m, t_max);
                double ury1, uiy1, ury2, uiy2;
                integral_u_quad_region1(fr_yy, fi_yy, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, ury1, uiy1, t_cut, t_m, t_max);
                integral_u_quad_region2(fr_yy, fi_yy, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, ury2, uiy2, t_cut, t_m, t_max);

                double iz1 = 0., iz2 = 0.;
                for (int i_z = 0; i_z < static_cast<int>(n_z_); ++i_z) {
                    double fz = (i_z == 0 || i_z == static_cast<int>(n_z_)-1) ? 0.5 : 1.;
                    iz1 += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi_);
                    iz2 += fz * integral_dz_xandy(k, w, i_z, i_s, z_[i_z], phi2_);
                }

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
                integral_u_quad_region1(fr_xz, fi_xz, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, ur1, ui1, t_cut, t_m, t_max);
                integral_u_quad_region2(fr_xz, fi_xz, ws, s_off, Sqrt1mkk, w,
                                        eps_rel, ur2, ui2, t_cut, t_m, t_max);

                double iz1 = 0., iz2 = 0.;
                // start at i_z=1: xz_dphi_dz accesses phi[...][i_z-1]
                for (int i_z = 1; i_z < static_cast<int>(n_z_); ++i_z) {
                    iz1 += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi_);
                    iz2 += integral_dz_xz(k, w, i_z, i_s, z_[i_z], phi2_);
                }

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

void Integrator::integral_u_quad(gsl_function &fr, gsl_function &fi,
                                  gsl_integration_workspace *ws,
                                  double s, double Sqrt1mkk, double w,
                                  double eps_rel, double sign, double umin,
                                  double &real, double &imag,
                                  double t_cut, double t_m, double t_max) const {
    double args[9] = {w, Sqrt1mkk, s, sign,
                      t_cut, t_m, t_0_, t_max,
                      static_cast<double>(cutoff_type_)};
    fr.params = &args;
    fi.params = &args;
    double err;
    gsl_integration_qag(&fr, umin, u_max(s, t_max), GSL_EPSABS, eps_rel,
                        gsl_limit_, GSL_KEY, ws, &real, &err);
    gsl_integration_qag(&fi, umin, u_max(s, t_max), GSL_EPSABS, eps_rel,
                        gsl_limit_, GSL_KEY, ws, &imag, &err);
}

void Integrator::integral_u_quad_region1(gsl_function &fr, gsl_function &fi,
                                          gsl_integration_workspace *ws,
                                          double s, double Sqrt1mkk, double w,
                                          double eps_rel,
                                          double &real, double &imag,
                                          double t_cut, double t_m, double t_max) const {
    integral_u_quad(fr, fi, ws, s, Sqrt1mkk, w, eps_rel, -1., 1.,
                    real, imag, t_cut, t_m, t_max);
}

void Integrator::integral_u_quad_region2(gsl_function &fr, gsl_function &fi,
                                          gsl_integration_workspace *ws,
                                          double s, double Sqrt1mkk, double w,
                                          double eps_rel,
                                          double &real, double &imag,
                                          double t_cut, double t_m, double t_max) const {
    integral_u_quad(fr, fi, ws, s, Sqrt1mkk, w, eps_rel, 1., 0.,
                    real, imag, t_cut, t_m, t_max);
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
