#pragma once

#include <vector>
#include "amplitude.h"
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
//   - z-integrals are computed once in a separate GPU kernel and reused for
//     every cutoff-time launch.
//   - Each thread writes 6 linear contributions to a device buffer.
//   - CPU finalisation: reduce over i_s, square, k-integrate → spectrum.

class GpuIntegrator {
public:
    // param > 0: use param directly as n_floor (same as --param N for CPU Filon)
    // param <= 0: default n_floor = 8192
    GpuIntegrator(const std::vector<std::vector<double>> &input_phi,
                  const Setup &setup, int param = -1,
                  int panels_per_oscillation = 64);
    ~GpuIntegrator();

    // IMPORTANT: the plateau (C1==1) portion of the u-integral is accumulated
    // incrementally across time steps in device memory (see d_cumbuf_ below),
    // so i_t must be called in strictly increasing order starting at 0 —
    // matches how main.cpp already drives it. Calling out of order throws.
    std::vector<double> Compute(int i_t);

    // Like Compute, but also returns the pre-squaring complex amplitude
    // A(w, cos_theta) for each (frequency, angle) bin. Same ordering
    // requirement as Compute — shares the same incremental device-side
    // plateau state, so the two must not be interleaved for a given i_t
    // sequence (call one or the other consistently across all i_t for a
    // given GpuIntegrator instance, same as the CPU integrators).
    AmplitudeResult ComputeAmplitude(int i_t);

    const std::vector<double> &GetW()     const { return wlist_; }
    const std::vector<double> &GetSlist() const { return slist_; }
    const std::vector<double> &GetZ()     const { return z_; }

private:
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    void build_phi2(const std::vector<std::vector<double>> &input_phi,
                    std::vector<double> &phi2_host) const;

    // Shared implementation for Compute()/ComputeAmplitude(): launches the
    // kernel, reads back intbuf, and reduces over i_s to get (re, im) and the
    // squared spectrum per (i_w, i_k). If out_amp_re/out_amp_im are non-null,
    // also writes the pre-squaring (re, im) into them (row-major
    // [i_w * n_k_ + i_k], matching AmplitudeResult::amp_re/amp_im).
    // Enforces the strictly-increasing-i_t ordering requirement and advances
    // t_cut_prev_/last_i_t_processed_ on success.
    std::vector<double> RunAndReduce(int i_t,
                                     std::vector<double> *out_amp_re,
                                     std::vector<double> *out_amp_im);

    std::size_t n_w_, n_k_, n_s_, n_z_;
    double ds_, dz_;
    double t_cut_base_, t_m_base_, t_max_base_, t_0_, d_;
    int cutoff_type_;

    std::vector<double> wlist_, slist_, z_, times_, klist_;

    int param_floor_ = -1;  // from constructor param; -1 → adaptive
    int n_floor_     = 0;   // resolved each Compute()
    int panels_per_oscillation_ = 64;

    // Device arrays (allocated in constructor, freed in destructor)
    // Layout: [iz * n_s + is] (transposed relative to the natural [is][iz]
    // input), so consecutive threads (consecutive i_s -- the fastest-varying
    // index in precompute_z_kernel) read consecutive memory for a fixed iz.
    double *d_phi_    = nullptr;   // [n_z * n_s]
    double *d_phi2_   = nullptr;   // [n_z * n_s]
    double *d_z_      = nullptr;   // [n_z]
    double *d_w_      = nullptr;   // [n_w]
    double *d_k_      = nullptr;   // [n_k]
    double *d_s_      = nullptr;   // [n_s]
    double *d_zbuf_   = nullptr;   // [n_w * n_k * n_s * 6], precomputed once
    double *d_intbuf_ = nullptr;   // [n_w * n_k * n_s * 6]  — reused each Compute()

#ifdef GW_KERNEL_TIMING
    // Diagnostic-build-only: per-thread clock64() cycle counts for
    // [u_integral, zz, xa, xz, tail], reused each RunAndReduce() call.
    // Not present in the normal production binary — see Makefile's
    // gpu_profile target and gw_kernel's timebuf parameter.
    long long *d_timebuf_ = nullptr;   // [n_w * n_k * n_s * 5]
#endif

    // Persisted cumulative plateau state for the u-integral, one 16-double
    // slot per (i_w,i_k,i_s) thread [zz1,xx1,yy1,xz1,zz2,xx2,yy2,xz2] x
    // (re,im); layout is [idx*16 + slot], idx = i_w*n_k*n_s + i_k*n_s + i_s,
    // matching gw_kernel's existing indexing. Allocated (zero-initialised) in
    // the constructor, persists across Compute() calls (unlike d_intbuf_,
    // which is a scratch buffer reset every call), freed in destructor.
    double *d_cumbuf_ = nullptr;

    // t_cut from the previous Compute() call. A large-but-finite negative
    // sentinel before the first call (not actual infinity, for consistency
    // with the CPU integrators — see their t_cut_prev_ comments); t_cut_prev_/s
    // stays hugely negative but finite for any physically sane s, so it
    // clamps to umin on the first step.
    static constexpr double kNegInfSentinel = -1e30;
    double t_cut_prev_ = kNegInfSentinel;
    int    last_i_t_processed_ = -1;
};
