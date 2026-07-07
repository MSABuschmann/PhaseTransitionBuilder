#pragma once

#include <vector>
#include "setup.h"

// Drop-in replacement for Integrator / FilonIntegrator that runs the
// u-integration on a CUDA GPU using streaming Filon quadrature.
//
// Public interface is identical to the CPU versions so main.cpp can switch
// with a single #ifdef USE_GPU.
//
// Architecture (one CUDA thread per (i_w, i_k, i_s) triple):
//   - Streaming Filon u-integral: no array allocation, accumulators live in
//     registers.  N is chosen adaptively per call to resolve Bessel-function
//     oscillations inside g(u), same criterion as the CPU Filon version.
//   - z-integrals are computed serially within each thread.
//   - Each thread writes 6 linear contributions to a device buffer.
//   - CPU finalisation: reduce over i_s, square, k-integrate → spectrum.

class GpuIntegrator {
public:
    // param > 0: use param directly as n_floor (same as --param N for CPU Filon)
    // param <= 0: compute adaptive floor from 64*w_max*t_cut/(2π)
    GpuIntegrator(const std::vector<std::vector<double>> &input_phi,
                  const Setup &setup, int param = -1);
    ~GpuIntegrator();

    std::vector<double> Compute(int i_t) const;

    const std::vector<double> &GetW()     const { return wlist_; }
    const std::vector<double> &GetSlist() const { return slist_; }
    const std::vector<double> &GetZ()     const { return z_; }

private:
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    void build_phi2(const std::vector<std::vector<double>> &input_phi,
                    std::vector<double> &phi2_host) const;

    std::size_t n_w_, n_k_, n_s_, n_z_;
    double ds_, dz_;
    double t_cut_base_, t_m_base_, t_max_base_, t_0_, d_;
    int cutoff_type_;

    std::vector<double> wlist_, slist_, z_, times_, klist_;

    int         param_floor_ = -1;  // from constructor param; -1 → adaptive
    mutable int n_floor_ = 0;       // resolved each Compute()

    // Device arrays (allocated in constructor, freed in destructor)
    double *d_phi_    = nullptr;   // [n_s * n_z]
    double *d_phi2_   = nullptr;   // [n_s * n_z]
    double *d_z_      = nullptr;   // [n_z]
    double *d_w_      = nullptr;   // [n_w]
    double *d_k_      = nullptr;   // [n_k]
    double *d_s_      = nullptr;   // [n_s]
    double *d_intbuf_ = nullptr;   // [n_w * n_k * n_s * 6]  — reused each Compute()
};
