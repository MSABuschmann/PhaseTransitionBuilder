#pragma once

#include <cmath>
#include <gsl/gsl_integration.h>
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
    // Returns spectrum[n_w].  Uses local copies so it is non-destructive.
    std::vector<double> Compute(int i_t) const;

    // Like Compute, but also returns the pre-squaring complex amplitude
    // A(w, cos_theta) for each (frequency, angle) bin.
    AmplitudeResult ComputeAmplitude(int i_t) const;

    const std::vector<double> &GetW()     const { return wlist_; }
    const std::vector<double> &GetSlist() const { return slist_; }
    const std::vector<double> &GetZ()     const { return z_; }

private:
    // Helpers
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    void SetPhi2();
    double k_integral(std::size_t i_w, double eps_rel,
                      double t_cut, double t_m, double t_max,
                      std::vector<double> *out_amp_re = nullptr,
                      std::vector<double> *out_amp_im = nullptr) const;

    void PrecomputeZIntegrals();

    void integral_u_quad(gsl_function &fr, gsl_function &fi,
                         gsl_integration_workspace *ws,
                         double s, double Sqrt1mkk, double w, double eps_rel,
                         double sign, double umin,
                         double &real, double &imag,
                         double t_cut, double t_m, double t_max) const;

    void integral_u_quad_region1(gsl_function &fr, gsl_function &fi,
                                 gsl_integration_workspace *ws,
                                 double s, double Sqrt1mkk, double w,
                                 double eps_rel,
                                 double &real, double &imag,
                                 double t_cut, double t_m, double t_max) const;

    void integral_u_quad_region2(gsl_function &fr, gsl_function &fi,
                                 gsl_integration_workspace *ws,
                                 double s, double Sqrt1mkk, double w,
                                 double eps_rel,
                                 double &real, double &imag,
                                 double t_cut, double t_m, double t_max) const;

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

    static constexpr double GSL_EPSABS = 0.;
    static constexpr int    GSL_KEY    = GSL_INTEG_GAUSS15;
    int gsl_limit_;  // set from --param or default 1000
};
