#pragma once

#include <optional>
#include <vector>
#include "amplitude.h"
#include "interpolator.h"
#include "setup.h"

// Drop-in-ish replacement for Integrator / FilonIntegrator that runs the
// u-integration on a CUDA GPU using streaming Filon quadrature.
//
// Unlike the CPU integrators, this class does NOT take the complete field
// history in its constructor and does NOT expose Compute(i_t)/ComputeAmplitude(i_t)
// directly. Instead it processes the field evolution in bounded s-batches
// (see ProcessBatch) so device memory scales with a small, fixed batch size
// rather than with the full s-grid — the full history plus its z/w/k-indexed
// buffers scale roughly as gamma_ij^4 and become infeasible on a single GPU
// well before gamma_ij=32/64 (see AI_HANDOFF.md's "Planned minimal-memory GPU
// rewrite"). main.cpp streams Evolution's snapshots through an SBatchWindow
// into ProcessBatch(); once every batch has been processed, Finalize(i_t)/
// FinalizeAmplitude(i_t) read out the accumulated result per cutoff time.
//
// Architecture (one CUDA thread per (i_w, i_k, i_s_local) triple within a batch):
//   - Streaming Filon u-integral: no array allocation, accumulators live in
//     registers. N is chosen adaptively per call to resolve Bessel-function
//     oscillations inside g(u), same criterion as the CPU Filon version.
//   - z-integrals are computed once per batch in a separate GPU kernel and
//     reused for every cutoff-time launch within that batch.
//   - Each thread writes 6 LINEAR contributions per (i_t, i_w, i_k); these are
//     reduced over the batch's local i_s on the host and += into a small,
//     batch-independent accumulator (accum_) sized n_t*n_w*n_k*6. Squaring
//     (the nonlinear step) is deferred to Finalize()/FinalizeAmplitude(),
//     once every batch has contributed.
class GpuIntegrator {
public:
    // param > 0: use param directly as n_floor (same as --param N for CPU Filon)
    // param <= 0: default n_floor = 8192
    // s_batch_size: number of NEW s-slices processed per ProcessBatch() call
    //   (device buffers are sized for s_batch_size+1 to hold the halo slice).
    GpuIntegrator(const Setup &setup, int param = -1,
                  int panels_per_oscillation = 64,
                  int s_batch_size = 32);
    ~GpuIntegrator();

    GpuIntegrator(const GpuIntegrator &)            = delete;
    GpuIntegrator &operator=(const GpuIntegrator &) = delete;

    // Process one batch of the field evolution. batch_phi/batch_s must be in
    // increasing-s order; for every batch after the first, index 0 is the
    // halo slice (the previous batch's last NEW slice, needed for the i_s-1
    // finite-difference derivative in precompute_z_kernel) — i.e. batch_phi
    // has size (new_slices_in_this_batch + (is_first_batch ? 0 : 1)).
    // Call in order: is_first_batch true exactly once (first call),
    // is_last_batch true exactly once (last call); a single call may have
    // both true (very small gamma_ij, one batch total).
    void ProcessBatch(const std::vector<std::vector<double>> &batch_phi,
                       const std::vector<double> &batch_s,
                       bool is_first_batch, bool is_last_batch);

    // Cheap CPU-only readouts of the accumulated result for one cutoff index.
    // No kernel launch; callable in any order, any number of times, only
    // valid after the last ProcessBatch() call (is_last_batch == true) has
    // returned.
    std::vector<double> Finalize(int i_t) const;
    AmplitudeResult      FinalizeAmplitude(int i_t) const;

    const std::vector<double> &GetW() const { return wlist_; }

private:
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    // Fills phi2_host (size n_z_ * batch_s.size(), [iz*n_batch+is] layout)
    // using the persistent phi0_interp_ built once in the constructor.
    void build_phi2_batch(const std::vector<double> &batch_s,
                          std::vector<double> &phi2_host) const;

    // Shared finalize implementation for Finalize()/FinalizeAmplitude().
    std::vector<double> FinalizeImpl(int i_t,
                                     std::vector<double> *out_amp_re,
                                     std::vector<double> *out_amp_im) const;

    std::size_t n_w_, n_k_, n_z_;
    std::size_t n_t_;
    double ds_, dz_;
    double t_cut_base_, t_m_base_, t_max_base_, t_0_, d_;
    int cutoff_type_;

    std::vector<double> wlist_, z_, times_, klist_;

    int param_floor_ = -1;  // from constructor param; -1 -> adaptive
    int n_floor_     = 0;   // resolved each ProcessBatch()
    int panels_per_oscillation_ = 64;

    int s_batch_size_;   // requested new-slices-per-batch
    int max_alloc_;       // = s_batch_size_ + 1 (room for the halo slice)

    // Fixed (t=0) two-bubble reference-field interpolant, built once in the
    // constructor directly from setup.phi0/setup.z (build_phi2's original
    // logic only ever reads input_phi[0], i.e. the t=0 profile -- it has no
    // dependence on the evolved history). Interpolator stores REFERENCES to
    // its inputs, so z0_new_/phi0_new_ must outlive it as persistent members,
    // declared before phi0_interp_ so they're constructed first.
    std::vector<double> z0_new_, phi0_new_;
    std::optional<Interpolator> phi0_interp_;

    // Device arrays: allocated ONCE in the constructor at max_alloc_
    // (batch-sized, not full-s-grid-sized) and reused for every
    // ProcessBatch() call -- no per-batch cudaMalloc/cudaFree.
    // Layout: [iz * max_alloc_ + is_local] (transposed relative to the
    // natural [is_local][iz] input), so consecutive threads (consecutive
    // i_s_local -- the fastest-varying index in precompute_z_kernel) read
    // consecutive memory for a fixed iz.
    double *d_phi_    = nullptr;   // [n_z_ * max_alloc_]
    double *d_phi2_   = nullptr;   // [n_z_ * max_alloc_]
    double *d_z_      = nullptr;   // [n_z_]
    double *d_w_      = nullptr;   // [n_w_]
    double *d_k_      = nullptr;   // [n_k_]
    double *d_s_      = nullptr;   // [max_alloc_]
    double *d_zbuf_   = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 6], recomputed once per batch
    double *d_intbuf_ = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 6]  -- reused each cutoff within a batch

#ifdef GW_KERNEL_TIMING
    // Diagnostic-build-only: per-thread clock64() cycle counts for
    // [u_integral, zz, xa, xz, tail], reused each ProcessBatch()/cutoff.
    // Not present in the normal production binary -- see Makefile's
    // gpu_profile target and gw_kernel's timebuf parameter.
    long long *d_timebuf_ = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 5]
#endif

    // Persisted cumulative plateau state for the u-integral, one 16-double
    // slot per (i_w,i_k,i_s_local) thread [zz1,xx1,yy1,xz1,zz2,xx2,yy2,xz2] x
    // (re,im); layout is [idx*16 + slot]. Allocated once at max_alloc_, but
    // RESET (memset to 0) at the start of every ProcessBatch() call -- unlike
    // the old full-grid design, this state does not persist across batches,
    // only across the cutoff-time loop WITHIN one batch (each s belongs to
    // exactly one batch).
    double *d_cumbuf_ = nullptr;

    // t_cut from the previous cutoff-time step, WITHIN the current batch;
    // reset to the sentinel at the start of every ProcessBatch() call.
    static constexpr double kNegInfSentinel = -1e30;
    double t_cut_prev_ = kNegInfSentinel;

    // Persistent, batch-independent linear accumulator: for every cutoff
    // index i_t and (i_w,i_k), the running sum (over ALL batches processed
    // so far) of gw_kernel's 6 linear per-s contributions
    // [zz_re,zz_im,xa_re,xa_im,xz_re,xz_im]. Sized n_t_*n_w_*n_k_*6 --
    // independent of n_s/n_z, unlike everything above. This is what
    // Finalize()/FinalizeAmplitude() read from; the nonlinear square +
    // k-integration happens only there, once all batches have contributed.
    std::vector<double> accum_;

    bool finalized_ = false;   // true once is_last_batch has been processed
};
