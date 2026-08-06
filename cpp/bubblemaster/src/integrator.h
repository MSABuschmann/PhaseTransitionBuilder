#pragma once

#include <cmath>
#include <gsl/gsl_integration.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "amplitude.h"
#include "setup.h"

class Integrator {
public:
    // param: GSL subinterval limit; -1 → use default (1000)
    Integrator(const std::vector<std::vector<double>> &input_phi,
               const Setup &setup, int param = -1);

    // Run GW integration for time-index i_t.
    // Returns spectrum[n_w].
    //
    // IMPORTANT: the u-integral is accumulated incrementally across time steps
    // (see cum_* members below), so i_t must be called in strictly increasing
    // order starting at 0 for a given Integrator instance — this matches how
    // main.cpp already drives it. Calling out of order throws.
    std::vector<double> Compute(int i_t);

    // Like Compute, but also returns the pre-squaring complex amplitude
    // A(w, cos_theta) for each (frequency, angle) bin. Same ordering
    // requirement as Compute.
    AmplitudeResult ComputeAmplitude(int i_t);

    const std::vector<double> &GetW()     const { return wlist_; }
    const std::vector<double> &GetSlist() const { return slist_; }
    const std::vector<double> &GetZ()     const { return z_; }

private:
    // Helpers
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    void SetPhi2();
    double k_integral(std::size_t i_w, double eps_rel,
                      double t_cut, double t_m, double t_max, double t_cut_prev,
                      std::vector<double> *out_amp_re = nullptr,
                      std::vector<double> *out_amp_im = nullptr);

    void PrecomputeZIntegrals();

    // Incremental u-integral: adds the plateau (C1==1) contribution between
    // t_cut_prev and t_cut to the persisted cum_re/cum_im state (read-modify-
    // write), then computes the transition-window piece fresh from the new
    // plateau boundary out to u_max. Returns cum + window in real/imag — same
    // total this step's u-integral would have given the old full re-integration.
    // fr/fi must be the C1-windowed integrand; fr_plain/fi_plain the matching
    // unwindowed core (see integrand.h). cum_re/cum_im are the persisted
    // per-(i_w,i_k,i_s) state slot for this quad-type and region.
    void integral_u_quad_incremental(gsl_function &fr, gsl_function &fi,
                                     gsl_function &fr_plain, gsl_function &fi_plain,
                                     gsl_integration_workspace *ws,
                                     double s, double Sqrt1mkk, double w, double eps_rel,
                                     double sign, double umin_region,
                                     double &real, double &imag,
                                     double t_cut, double t_m, double t_max,
                                     double t_cut_prev,
                                     double &cum_re, double &cum_im) const;

    void integral_u_quad_incremental_region1(
        gsl_function &fr, gsl_function &fi,
        gsl_function &fr_plain, gsl_function &fi_plain,
        gsl_integration_workspace *ws,
        double s, double Sqrt1mkk, double w, double eps_rel,
        double &real, double &imag,
        double t_cut, double t_m, double t_max, double t_cut_prev,
        double &cum_re, double &cum_im) const;

    void integral_u_quad_incremental_region2(
        gsl_function &fr, gsl_function &fi,
        gsl_function &fr_plain, gsl_function &fi_plain,
        gsl_integration_workspace *ws,
        double s, double Sqrt1mkk, double w, double eps_rel,
        double &real, double &imag,
        double t_cut, double t_m, double t_max, double t_cut_prev,
        double &cum_re, double &cum_im) const;

    double u_max(double s, double t_max_local) const;

    // inline integrand helpers (same signatures as reference)
    inline double integral_dz_xandy(double k, double w, int i_z, int i_s,
                                     double zval,
                                     const std::vector<std::vector<double>> &phi) const {
        double q = (phi[i_s][i_z] - phi[i_s - 1][i_z]) / ds_;
        return dz_ * 2.0 * std::cos(w * k * zval) * q * q;
    }
    inline double integral_dz_zz(double k, double w, int i_z, int i_s,
                                  double zval,
                                  const std::vector<std::vector<double>> &phi) const {
        double q = (phi[i_s][i_z] - phi[i_s][i_z - 1]) / dz_;
        return dz_ * 2.0 * std::cos(w * k * (zval - dz_ * 0.5)) * q * q;
    }
    inline double xz_dphi_ds(int i_s, int i_z,
                              const std::vector<std::vector<double>> &phi) const {
        return 0.5 / ds_ * (phi[i_s][i_z] - phi[i_s-1][i_z]
                          + phi[i_s][i_z-1] - phi[i_s-1][i_z-1]);
    }
    inline double xz_dphi_dz(int i_s, int i_z,
                              const std::vector<std::vector<double>> &phi) const {
        return 0.5 / dz_ * (phi[i_s][i_z] - phi[i_s][i_z-1]
                          + phi[i_s-1][i_z] - phi[i_s-1][i_z-1]);
    }
    inline double integral_dz_xz(double k, double w, int i_z, int i_s,
                                  double zval,
                                  const std::vector<std::vector<double>> &phi) const {
        return dz_ * 2.0 * std::sin(w * k * (zval - dz_ * 0.5))
             * xz_dphi_ds(i_s, i_z, phi)
             * xz_dphi_dz(i_s, i_z, phi);
    }

    // stored state
    std::size_t n_k_, n_z_, n_s_, n_w_;
    double ds_, dz_;

    double t_cut_base_, t_m_base_, t_max_base_, t_0_;
    double d_;
    int    cutoff_type_;

    std::vector<double> z_, wlist_, slist_, times_;
    std::vector<std::vector<double>> phi_;   // snapshots from evolution
    std::vector<std::vector<double>> phi2_;  // two-bubble reference field

    // Precomputed z-integrals, indexed [i_w * n_k * n_s + i_k * n_s + i_s].
    // Filled once in the constructor via PrecomputeZIntegrals().
    std::vector<double> iz_zz1_, iz_zz2_;
    std::vector<double> iz_xa1_, iz_xa2_;
    std::vector<double> iz_xz1_, iz_xz2_;

    // Persisted cumulative u-integral plateau state, same indexing as the
    // iz_* arrays above: [i_w * n_k * n_s + i_k * n_s + i_s]. Suffix 1/2
    // selects region1 (umin=1)/region2 (umin=0). Updated in place by
    // Compute()/ComputeAmplitude() as time steps advance; see
    // integral_u_quad_incremental().
    std::vector<double> cum_zz1_re_, cum_zz1_im_, cum_zz2_re_, cum_zz2_im_;
    std::vector<double> cum_xx1_re_, cum_xx1_im_, cum_xx2_re_, cum_xx2_im_;
    std::vector<double> cum_yy1_re_, cum_yy1_im_, cum_yy2_re_, cum_yy2_im_;
    std::vector<double> cum_xz1_re_, cum_xz1_im_, cum_xz2_re_, cum_xz2_im_;

    // t_cut from the previous Compute()/ComputeAmplitude() call (the lower
    // bound for this step's plateau increment). A large-but-finite negative
    // sentinel before the first call (not actual infinity — the build uses
    // -ffast-math, under which infinity is undefined behavior), so the first
    // step's "increment" naturally covers the full plateau from each region's
    // umin out to t_cut(0): t_cut_prev_/s is still hugely negative for any
    // physically sane s, so it clamps to umin_region.
    static constexpr double kNegInfSentinel = -1e30;
    double t_cut_prev_ = kNegInfSentinel;
    int    last_i_t_processed_ = -1;

    static constexpr double GSL_EPSABS = 0.;
    static constexpr int    GSL_KEY    = GSL_INTEG_GAUSS15;
    int gsl_limit_;  // set from --param or default 1000
};
