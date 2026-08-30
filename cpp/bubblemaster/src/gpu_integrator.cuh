#pragma once

#include <functional>
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
// well before gamma_ij=32/64. main.cpp drives this via RunEvolution(), which
// runs the field evolution itself on-device and feeds it straight into the
// same per-batch processing ProcessBatch() exposes for callers that already
// have their own (CPU-computed) batches; once every batch has been
// processed, Finalize(i_t)/FinalizeAmplitude(i_t) read out the accumulated
// result per cutoff time.
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
    // Checkpoint/resume: the state needed to extend a row to new, later
    // cutoff times without re-integrating the ones already computed.
    //   seed_time  : the physical time of the row that produced this
    //                checkpoint (its last requested cutoff).
    //   plateau_re/plateau_im : the "pure plateau" amplitude (Re/Im A(w,k),
    //                same layout as AmplitudeResult::amp_re/amp_im) up to
    //                seed_time's OWN t_cut -- NOT the final windowed
    //                amplitude (which would double-count seed_time's own
    //                transition-window contribution when reused for a
    //                later time -- see the derivation in ProcessBatch/
    //                gw_kernel's comments and FinalizeImpl below).
    struct Checkpoint {
        double seed_time = 0.;
        std::vector<double> plateau_re, plateau_im;   // [n_w * n_k] each
    };

    // Mid-stream progress checkpoint: lets a run interrupted (e.g. by
    // walltime) partway through ProcessBatch()-ing THIS SAME requested times
    // range resume without redoing the expensive GPU integration for batches
    // already folded into accum_/accum_plateau_. Field evolution itself is
    // assumed cheap and always restarts from s=0 (see ProcessBatch's skip
    // logic) -- only the batch COUNT needs to be remembered, not any
    // per-batch field/cum state (which is reset every batch anyway).
    // Only ever valid for resuming the EXACT same (setup, times-range) that
    // produced it -- callers must check the fingerprint before constructing
    // (see checkpoint_io.h's stream_checkpoint_matches).
    struct StreamCheckpoint {
        int batches_done = 0;
        std::vector<double> accum;            // same layout as accum_
        std::vector<double> accum_plateau;    // same layout as accum_plateau_
    };

    // param > 0: use param directly as n_floor (same as --param N for CPU Filon)
    // param <= 0: default n_floor = 8192
    // s_batch_size: number of NEW s-slices processed per ProcessBatch() call
    //   (device buffers are sized for s_batch_size+1 to hold the halo slice).
    // seed: optional checkpoint from a prior run of the SAME (gamma_ij, d)
    //   row; every setup.times[] entry must be strictly greater than
    //   seed->seed_time (this class does not validate that setup itself
    //   matches the checkpoint's origin -- callers must check the fingerprint
    //   before constructing).
    // resume: optional mid-stream progress from an interrupted attempt at
    //   this EXACT (setup, times) request -- resumes accum_/accum_plateau_
    //   from it and skips GPU work for the batches it already covers.
    //   Independent of `seed`: a run can be both seeded (extending an
    //   earlier-completed prefix of this row) and resumed (this particular
    //   extension attempt was itself interrupted) at the same time.
    GpuIntegrator(const Setup &setup, int param = -1,
                  int panels_per_oscillation = 64,
                  int s_batch_size = 32,
                  const Checkpoint *seed = nullptr,
                  const StreamCheckpoint *resume = nullptr);
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

    // Runs the ENTIRE field evolution (s=0..smax) on-device and feeds it
    // straight into the same batch-processing pipeline ProcessBatch() uses
    // -- replaces main.cpp's external Evolution+SBatchWindow construction
    // for the GPU binary. phi never leaves device memory (no per-batch
    // host->device copy), and the per-z stencil update runs as a GPU kernel
    // instead of OpenMP over 32 CPU cores. Mirrors Evolution::Evolve()'s
    // exact step/snapshot cadence and SBatchWindow's exact batching/halo
    // logic (see gpu_integrator.cu) -- must match bit-for-bit in structure,
    // only the arithmetic backend (CUDA kernels vs CPU loops) differs.
    // Respects seed/resume exactly as ProcessBatch() does (same skip_until_/
    // t_cut_prev_ state) -- only where the batch data comes from changes.
    //
    // on_batch_boundary(is_last_batch), if given, is called after EVERY
    // batch boundary (skipped or processed, matching the old external
    // SBatchWindow callback's unconditional per-batch firing) -- lets the
    // caller do its own periodic bookkeeping (e.g. main.cpp's Format-B
    // progress checkpoint) without RunEvolution needing to know about
    // checkpoint_io.h itself.
    //
    // Prints "Timing phase=field_evolution" (the GPU stepping kernels only)
    // and "Timing phase=gpu_integration" (time spent inside
    // ProcessUploadedBatch calls) separately when done -- field_evolution
    // uses the SAME phase name as the CPU binary's own evolution timing, so
    // the two are directly comparable in a log.
    void RunEvolution(const Setup &setup,
                       std::function<void(bool is_last_batch)> on_batch_boundary = nullptr);

    // Cheap CPU-only readouts of the accumulated result for one cutoff index.
    // No kernel launch; callable in any order, any number of times, only
    // valid after the last ProcessBatch() call (is_last_batch == true) has
    // returned. If this run was seeded (see Checkpoint), the seed's
    // plateau_re/plateau_im are added in before squaring/k-integration, so
    // these already reflect the FULL cumulative result from s=0 of the
    // original row, not just this run's own increment.
    std::vector<double> Finalize(int i_t) const;
    AmplitudeResult      FinalizeAmplitude(int i_t) const;

    // Checkpoint for a FUTURE extension beyond this run's own last requested
    // time (setup.times.back()). Combines this run's own plateau-only
    // accumulator (accum_plateau_, built alongside accum_ but only tracking
    // contributions up through the LAST i_t's own t_cut, excluding its
    // transition-window piece) with any seed this run itself was given, so
    // checkpoints chain correctly across repeated extensions. Only valid
    // after the last ProcessBatch() call has returned.
    Checkpoint MakeCheckpoint() const;

    // Mid-stream progress checkpoint, for a run that gets interrupted before
    // reaching is_last_batch. Callable after ANY ProcessBatch() call (no
    // need to have finalized) -- reflects however many batches have been
    // folded into accum_/accum_plateau_ so far, including any batches
    // skipped because `resume` already covered them.
    StreamCheckpoint MakeStreamCheckpoint() const;

    const std::vector<double> &GetW() const { return wlist_; }

private:
    static std::vector<double> linspace (double a, double b, int n);
    static std::vector<double> geomspace(double a, double b, std::size_t n);

    // Fills phi2_host (size n_z_ * batch_s.size(), [iz*n_batch+is] layout)
    // using the persistent phi0_interp_ built once in the constructor.
    void build_phi2_batch(const std::vector<double> &batch_s,
                          std::vector<double> &phi2_host) const;

    // Format-B skip bookkeeping, shared by ProcessBatch() (host-driven) and
    // RunEvolution() (device-driven) so both respect skip_until_ identically
    // -- see next_batch_index_/skip_until_'s doc comment below. Returns true
    // if this batch's GPU work should be skipped; the caller must NOT skip
    // field evolution itself either way (it always regenerates from s=0).
    bool AdvanceAndCheckSkip(bool is_last_batch);

    // Shared tail of ProcessBatch()/RunEvolution(): assumes d_phi_'s first
    // n_z_*n_s_local doubles are ALREADY correctly populated (host-uploaded
    // by ProcessBatch(), or device-repacked by RunEvolution()) -- builds/
    // uploads phi2 and s, runs precompute_z_kernel + the per-cutoff-time
    // gw_kernel loop, and folds the result into accum_/accum_plateau_.
    void ProcessUploadedBatch(int n_s_local, const std::vector<double> &batch_s,
                              bool is_first_batch, bool is_last_batch);

    // Shared finalize implementation for Finalize()/FinalizeAmplitude().
    // Adds seed_re_/seed_im_ (if has_seed_) before squaring/k-integration.
    std::vector<double> FinalizeImpl(int i_t,
                                     std::vector<double> *out_amp_re,
                                     std::vector<double> *out_amp_im) const;

    // Combines accum_plateau_'s 6 raw components into plateau_re/plateau_im
    // via the SAME linear (Onemkk/TwokSq) combination FinalizeImpl uses for
    // accum_ -- no squaring, no k-integration. This is this run's OWN
    // plateau contribution only; MakeCheckpoint() adds any seed on top.
    void CombinePlateau(std::vector<double> &plateau_re,
                        std::vector<double> &plateau_im) const;

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

    // --- RunEvolution()'s on-device field state (unused by the host-driven
    // ProcessBatch() path) ---
    // Current evolution state, one value per z -- NOT batch-sized, since
    // EvolvePi's stencil only ever reads spatial neighbors at the CURRENT
    // step (never previous-s values); the s-history only matters later, to
    // precompute_z_kernel's own s-derivatives.
    double *d_phi_cur_ = nullptr;  // [n_z_]
    double *d_pi_cur_  = nullptr;  // [n_z_]
    // Rolling snapshot staging buffer for the CURRENT batch, FIXED stride
    // max_alloc_ (unlike d_phi_'s stride, which is n_s_local -- only known
    // once a batch closes) -- RunEvolution() writes one new column at a time
    // as evolution proceeds, then repacks the batch's first n_s_local
    // columns into d_phi_ (matching precompute_z_kernel's expected tight
    // layout) once the batch is complete. Halo carry-forward is then just a
    // same-buffer column copy (see carry_halo_kernel).
    double *d_phi_evolve_ = nullptr;   // [n_z_ * max_alloc_]
    double *d_z_      = nullptr;   // [n_z_]
    double *d_w_      = nullptr;   // [n_w_]
    double *d_k_      = nullptr;   // [n_k_]
    double *d_s_      = nullptr;   // [max_alloc_]
    double *d_zbuf_   = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 6], recomputed once per batch
    double *d_intbuf_ = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 6]  -- reused each cutoff within a batch
    // Mirrors d_intbuf_ exactly, but built from the PRE-transition-window
    // cum state (cumbuf, as it stands right after this launch's plateau
    // update) instead of the full zz_r/xx/yy/xz values -- i.e. this cutoff's
    // contribution with its OWN transition-window piece excluded. Written by
    // gw_kernel on every launch (cheap: reuses values already in registers);
    // only copied back and reduced on the host for i_t == n_t_-1, since
    // that is the only cutoff a future checkpoint can ever be taken from.
    double *d_plateau_intbuf_ = nullptr;   // [n_w_ * n_k_ * max_alloc_ * 6]

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
    // reset at the start of every ProcessBatch() call -- to kNegInfSentinel
    // normally (nothing precedes the first i_t), or to seed_t_cut_ if this
    // run was seeded from a checkpoint (skips re-integrating the u-range the
    // seed's row already covered).
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

    // Same accumulation as accum_, but only for the LAST cutoff index
    // (n_t_-1) and only its plateau (pre-transition-window) contribution --
    // i.e. sized n_w_*n_k_*6, not n_t_*n_w_*n_k_*6. Reduced from
    // d_plateau_intbuf_ every batch, same as accum_'s own reduction.
    // CombinePlateau()/MakeCheckpoint() turn this into the amplitude a
    // future run can seed itself from.
    std::vector<double> accum_plateau_;

    // Optional seed from a prior checkpoint (see constructor's `seed`
    // parameter). has_seed_ gates both the per-batch t_cut_prev_ reset and
    // FinalizeImpl's post-accum_ addition.
    bool has_seed_ = false;
    double seed_t_cut_ = kNegInfSentinel;
    std::vector<double> seed_re_, seed_im_;   // [n_w_ * n_k_] each, if has_seed_

    bool finalized_ = false;   // true once is_last_batch has been processed

    // Batch-skip bookkeeping for Format-B stream-resume. next_batch_index_
    // counts every ProcessBatch() call (skipped or not); a call with
    // next_batch_index_ < skip_until_ does zero GPU work (no upload, no
    // kernel launch) since its contribution is already folded into the
    // accum_/accum_plateau_ this run was constructed with -- field evolution
    // still regenerates that batch (cheap, restarted from s=0 by main.cpp),
    // it's just never handed to the GPU. skip_until_ is 0 (nothing skipped)
    // unless this run was constructed with a StreamCheckpoint.
    int next_batch_index_ = 0;
    int skip_until_       = 0;
};
