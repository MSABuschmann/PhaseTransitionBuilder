#pragma once

#include <cmath>
#include <vector>

#include "amplitude.h"
#include "setup.h"

// Drop-in replacement for Integrator that uses Filon quadrature instead of
// GSL adaptive quadrature for the u-integrals.  The public interface is
// identical so main.cpp can switch between the two with a single #ifdef.
//
// NUMA layout: phi_ and phi2_ are stored as flat 1-D arrays indexed
// [i_s * n_z_ + i_z] and initialised inside an OMP parallel-for with
// schedule(static).  The s-loop in k_integral uses the same schedule,
// so each thread accesses the rows it first-touched → NUMA-local reads.

class FilonIntegrator {
public:
    // param: N_min for Filon panels; -1 → use compiled FILON_N_MIN default
    FilonIntegrator(const std::vector<std::vector<double>> &input_phi,
                    const Setup &setup, int param = -1);

    std::vector<double> Compute(int i_t) const;
    AmplitudeResult     ComputeAmplitude(int i_t) const;

    const std::vector<double> &GetW()     const { return wlist_; }
    const std::vector<double> &GetSlist() const { return slist_; }
    const std::vector<double> &GetZ()     const { return z_; }

private:
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    void SetPhi2();

    double k_integral(double w, double t_cut, double t_m, double t_max,
                      std::vector<double> *out_amp_re = nullptr,
                      std::vector<double> *out_amp_im = nullptr) const;

    void integral_u_filon(double s, double Sqrt1mkk, double w,
                          double sign, double umin,
                          double t_cut, double t_m, double t_max,
                          double &zz_r, double &zz_i,
                          double &xx_r, double &xx_i,
                          double &yy_r, double &yy_i,
                          double &xz_r, double &xz_i) const;

    double u_max(double s, double t_max_local) const {
        return 1.0 + t_max_local / s;
    }

    // z-integral helpers — flat array versions
    // phi[i_s * n_z_ + i_z]
    inline double integral_dz_xandy(double k, double w, int i_z, int i_s,
                                     double zval,
                                     const std::vector<double> &phi) const {
        double q = (phi[i_s * n_z_ + i_z] - phi[(i_s - 1) * n_z_ + i_z]) / ds_;
        return dz_ * 2.0 * std::cos(w * k * zval) * q * q;
    }
    inline double integral_dz_zz(double k, double w, int i_z, int i_s,
                                  double zval,
                                  const std::vector<double> &phi) const {
        double q = (phi[i_s * n_z_ + i_z] - phi[i_s * n_z_ + i_z - 1]) / dz_;
        return dz_ * 2.0 * std::cos(w * k * (zval - dz_ * 0.5)) * q * q;
    }
    inline double xz_dphi_ds(int i_s, int i_z,
                              const std::vector<double> &phi) const {
        return 0.5 / ds_ * (phi[ i_s      * n_z_ + i_z    ]
                          - phi[(i_s - 1) * n_z_ + i_z    ]
                          + phi[ i_s      * n_z_ + i_z - 1]
                          - phi[(i_s - 1) * n_z_ + i_z - 1]);
    }
    inline double xz_dphi_dz(int i_s, int i_z,
                              const std::vector<double> &phi) const {
        return 0.5 / dz_ * (phi[ i_s      * n_z_ + i_z    ]
                           - phi[ i_s      * n_z_ + i_z - 1]
                           + phi[(i_s - 1) * n_z_ + i_z    ]
                           - phi[(i_s - 1) * n_z_ + i_z - 1]);
    }
    inline double integral_dz_xz(double k, double w, int i_z, int i_s,
                                  double zval,
                                  const std::vector<double> &phi) const {
        return dz_ * 2.0 * std::sin(w * k * (zval - dz_ * 0.5))
             * xz_dphi_ds(i_s, i_z, phi)
             * xz_dphi_dz(i_s, i_z, phi);
    }

    std::size_t n_k_, n_z_, n_s_, n_w_;
    double ds_, dz_;

    double t_cut_base_, t_m_base_, t_max_base_, t_0_;
    double d_;
    int    cutoff_type_;

    int n_min_;  // effective FILON_N_MIN (set from --param or compile default)

    std::vector<double> z_, wlist_, slist_, times_;

    // Flat 1-D field arrays: index as phi_[i_s * n_z_ + i_z]
    std::vector<double> phi_;
    std::vector<double> phi2_;
};
