#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "evolution.h"
#include "io.h"
#include "setup.h"
#include "wall_energy.h"
#include "wall_radius.h"

#ifdef USE_GPU
#  include "checkpoint_io.h"
#  include "gpu_integrator.cuh"
#elif defined(USE_FILON)
#  include "filon_integrator.h"
   using Integrator = FilonIntegrator;
#else
#  include "integrator.h"
#endif

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static double elapsed(Clock::time_point t0) {
    return Sec(Clock::now() - t0).count();
}

int main(int argc, char *argv[]) {
    // Auto-flush std::cout after every '<<' — without this, output redirected
    // to a SLURM log file (not a TTY) is fully buffered, so progress prints
    // (per-time-step timing, etc.) don't actually reach the log until the
    // buffer fills or the process exits, making `tail -f` on the log useless
    // for monitoring an in-progress run.
    std::cout.setf(std::ios::unitbuf);

    auto t_start = Clock::now();
    if (argc < 3) {
        std::cerr << "Usage: bubblemaster <setup.h5> <output_dir/>"
                     " [--save-fields] [--save-amplitude] [--param N]"
                     " [--filon-panels-per-osc N] [--s-batch-size N]"
                     " [--overwrite] [--checkpoint-interval-minutes N]"
                     " [--wall-energy] [--wall-radius-scan] [--wall-radius-every N]\n"
                     "  --param N         Filon: N_min panels; GSL: subinterval limit\n"
                     "  --filon-panels-per-osc N  GPU Filon panels per Bessel oscillation\n"
                     "  --s-batch-size N  GPU only: s-slices streamed per batch (default 128)\n"
                     "  --save-amplitude  Also write Re/Im A(w,cos_theta) to result files\n"
                     "  --overwrite       GPU only: ignore/discard any existing results or\n"
                     "                    checkpoint in output_dir/ and start this row fresh\n"
                     "                    (default: auto-detect and resume/extend instead --\n"
                     "                    see GpuIntegrator's Checkpoint/StreamCheckpoint)\n"
                     "  --checkpoint-interval-minutes N  GPU only: how often to write a\n"
                     "                    mid-stream progress checkpoint (default 10)\n"
                     "  --wall-energy     CPU only: instead of the GW spectral integral,\n"
                     "                    write wall_energy.h5 (colliding-side vs\n"
                     "                    undisturbed-side wall energy over s). Combine with\n"
                     "                    --save-fields to also get the tracked z-window\n"
                     "                    boundaries, for overlaying on the field plot.\n"
                     "  --wall-radius-scan  CPU only: instead of the GW spectral integral,\n"
                     "                    write wall_radius_scan.h5 (R_mid/R_in/R_out over s,\n"
                     "                    measured directly from the undisturbed side's actual\n"
                     "                    gradient-energy peak/half-max points). Streams:\n"
                     "                    never holds the field history in memory.\n"
                     "  --wall-radius-every N  with --wall-radius-scan: measure only every\n"
                     "                    Nth saved snapshot (plus the last; default 1)\n"
                     "  --patch-moments --patch-sc S --patch-deltac D  CPU only: instead of the\n"
                     "                    GW integral, write patch_moments.h5 (wake-window\n"
                     "                    moments A(s), B(s) for the patch-fraction calibration,\n"
                     "                    window |z| < 4 D + max(s - S, 0)). Streams.\n";
        return 1;
    }

    const std::string setup_path = argv[1];
    const std::string output_dir = argv[2];
    bool save_fields    = false;
    bool save_amplitude = false;
    int  qual_param     = -1;   // -1 → use compiled default
    int  filon_panels_per_osc = 64;
    int  s_batch_size   = 128;  // locked production value
    bool overwrite      = false;
    double checkpoint_interval_minutes = 10.0;
    bool wall_energy    = false;
    bool wall_radius_scan = false;
    int  wall_radius_every = 1;
    bool patch_moments = false;
    double patch_sc = -1., patch_deltac = -1.;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--save-fields")
            save_fields = true;
        else if (a == "--save-amplitude")
            save_amplitude = true;
        else if (a == "--param" && i + 1 < argc)
            qual_param = std::atoi(argv[++i]);
        else if (a == "--filon-panels-per-osc" && i + 1 < argc)
            filon_panels_per_osc = std::atoi(argv[++i]);
        else if (a == "--s-batch-size" && i + 1 < argc)
            s_batch_size = std::atoi(argv[++i]);
        else if (a == "--overwrite")
            overwrite = true;
        else if (a == "--checkpoint-interval-minutes" && i + 1 < argc)
            checkpoint_interval_minutes = std::atof(argv[++i]);
        else if (a == "--wall-energy")
            wall_energy = true;
        else if (a == "--wall-radius-scan")
            wall_radius_scan = true;
        else if (a == "--wall-radius-every" && i + 1 < argc)
            wall_radius_every = std::atoi(argv[++i]);
        else if (a == "--patch-moments")
            patch_moments = true;
        else if (a == "--patch-sc" && i + 1 < argc)
            patch_sc = std::atof(argv[++i]);
        else if (a == "--patch-deltac" && i + 1 < argc)
            patch_deltac = std::atof(argv[++i]);
    }
#ifdef USE_GPU
    if (wall_energy) {
        std::cerr << "--wall-energy is CPU only (it uses the CPU "
                     "Evolution's per-snapshot phi, not produced by the GPU "
                     "on-device evolution path); use the CPU binary.\n";
        return 1;
    }
    if (wall_radius_scan) {
        std::cerr << "--wall-radius-scan is CPU only (it uses the CPU "
                     "Evolution's per-snapshot phi, not produced by the GPU "
                     "on-device evolution path); use the CPU binary.\n";
        return 1;
    }
#endif
    if (wall_radius_scan && save_fields) {
        std::cerr << "--wall-radius-scan streams the evolution and never holds "
                     "the field history, so it can't be combined with "
                     "--save-fields; run them separately.\n";
        return 1;
    }
    if (patch_moments && (patch_sc < 0. || patch_deltac <= 0.)) {
        std::cerr << "--patch-moments needs --patch-sc >= 0 and --patch-deltac > 0\n";
        return 1;
    }
    if (wall_radius_every < 1) {
        std::cerr << "--wall-radius-every must be at least 1\n";
        return 1;
    }

    std::cout << "Setup:  " << setup_path << "\n";
    std::cout << "Output: " << output_dir << "\n";
    if (save_fields)
        std::cout << "Saving fields\n";
    if (save_amplitude)
        std::cout << "Saving complex amplitude A(w, cos_theta)\n";

    // --- 1. Load setup ---
    auto t_setup_load = Clock::now();
    Setup setup(setup_path);
    std::cout << "Timing phase=setup_load seconds="
              << elapsed(t_setup_load) << "\n";

#ifdef USE_GPU
    // The streaming GPU integrator keeps device (and evolution) memory
    // bounded by processing the field history in batches of s-slices rather
    // than holding the complete history at once. --save-fields dumps the
    // complete history to disk, which is exactly what this design avoids
    // ever materializing -- so it's a hard error here rather than silently
    // reintroducing O(n_s*n_z) memory. Use the CPU binary for field dumps.
    if (save_fields) {
        std::cerr << "--save-fields is not supported by the GPU binary "
                     "(bounded-memory streaming never materializes the "
                     "complete field history); use the CPU binary instead, "
                     "or omit --save-fields.\n";
        return 1;
    }
    if (filon_panels_per_osc < 2) {
        std::cerr << "--filon-panels-per-osc must be at least 2\n";
        return 1;
    }
    if (s_batch_size < 1) {
        std::cerr << "--s-batch-size must be at least 1\n";
        return 1;
    }

    namespace fs = std::filesystem;
    H5::Exception::dontPrint();   // has_plateau_checkpoint/try_read_stream_checkpoint
                                   // deliberately probe for missing/corrupt files via
                                   // try/catch -- don't spam stderr with HDF5's default
                                   // auto-printed error trace for the expected-miss case.

    auto result_path_for = [&](int i_t) {
        std::ostringstream oss;
        oss << output_dir << "result_" << std::setfill('0') << std::setw(4) << i_t << ".h5";
        return oss.str();
    };

    const int n_t_total = setup.n_t;

    // --- Auto-detect what's already in output_dir/ ---
    // --overwrite discards it all and starts this row completely fresh.
    // Otherwise: k_done = how many of this row's cutoff times already have a
    // result file, counted contiguously from index 0 (result files are only
    // ever written as a complete, ordered set by the finalize loop below, or
    // not written at all if interrupted first -- see the Format-B carve-out
    // just below for the case where finalize itself was interrupted).
    int k_done = 0;
    if (overwrite) {
        if (fs::exists(output_dir)) {
            for (const auto &entry : fs::directory_iterator(output_dir)) {
                const std::string fname = entry.path().filename().string();
                if (fname.rfind("result_", 0) == 0 || fname == "checkpoint.h5")
                    fs::remove(entry.path());
            }
        }
    } else {
        for (int i = 0; i < n_t_total; ++i) {
            if (fs::exists(result_path_for(i))) k_done = i + 1;
            else break;
        }
    }

    if (k_done == n_t_total) {
        std::cout << "All " << n_t_total << " cutoff times already have a result in "
                  << output_dir << " -- nothing to do (pass --overwrite to redo).\n";
        return 0;
    }

    // times_local is what THIS run needs to compute -- computed from the
    // full setup.times BEFORE truncating it below.
    const std::vector<double> times_local(setup.times.begin() + k_done, setup.times.end());
    const std::string checkpoint_path = output_dir + "checkpoint.h5";

    // --- Format A seed: extend an already-completed prefix of this row ---
    std::optional<GpuIntegrator::Checkpoint> seed;
    if (!overwrite && k_done > 0) {
        const std::string last_result_path = result_path_for(k_done - 1);
        if (!has_plateau_checkpoint(last_result_path))
            throw std::runtime_error(
                "main: " + last_result_path + " has no plateau checkpoint to extend "
                "from -- was it produced by an older build, or with the CPU binary?");

        // Hard-require the OLD frequency grid: extending a row must reuse
        // its original wlist unchanged, not one freshly recomputed for a
        // possibly-different GAMMA_STAR_MAX. GpuIntegrator's own seed check
        // only compares vector SIZE (n_w*n_k), which a same-size but
        // shifted grid would pass silently -- this compares actual values.
        const std::vector<double> prior_wlist = read_result_wlist(last_result_path);
        bool wlist_matches = prior_wlist.size() == setup.wlist.size();
        for (std::size_t i = 0; wlist_matches && i < prior_wlist.size(); ++i)
            wlist_matches = std::abs(prior_wlist[i] - setup.wlist[i])
                          <= 1e-9 * std::max(1.0, std::abs(prior_wlist[i]));
        if (!wlist_matches)
            throw std::runtime_error(
                "main: setup.h5's wlist does not match " + last_result_path +
                "'s own wlist -- extending a row must reuse its ORIGINAL frequency "
                "grid unchanged, not a freshly recomputed one.");

        seed = read_plateau_checkpoint(last_result_path);
        std::cout << "Seeding from " << last_result_path
                  << "  (seed_time=" << seed->seed_time << ")\n";
    }

    // --- Format B resume: was a previous attempt at THIS SAME times_local
    // range interrupted mid-stream? ---
    std::optional<GpuIntegrator::StreamCheckpoint> stream_resume;
    if (!overwrite) {
        if (auto loaded = try_read_stream_checkpoint(checkpoint_path)) {
            if (stream_checkpoint_matches(*loaded, setup, k_done, n_t_total, times_local)) {
                stream_resume = loaded->checkpoint;
                std::cout << "Resuming stream from " << checkpoint_path
                          << "  (batches_done=" << stream_resume->batches_done << ")\n";
            } else {
                std::cout << "Ignoring stale " << checkpoint_path
                          << " (setup/progress has changed since it was written) -- "
                             "starting this range's streaming from scratch.\n";
                fs::remove(checkpoint_path);
            }
        }
    }

    // --- Truncate setup.times/n_t to just the remaining range. Evolution
    // never reads setup.times/n_t (only field-evolution parameters), so
    // mutating them in place here -- before either GpuIntegrator or
    // Evolution is constructed -- is safe. ---
    setup.times = times_local;
    setup.n_t   = static_cast<int>(times_local.size());

    auto t_integrator_setup = Clock::now();
    GpuIntegrator integrator(setup, qual_param, filon_panels_per_osc, s_batch_size,
                             seed ? &*seed : nullptr,
                             stream_resume ? &*stream_resume : nullptr);
    std::cout << "Timing phase=integrator_setup_total seconds="
              << elapsed(t_integrator_setup) << "\n";

    // --- 2+3. Run the 2D Milne field evolution entirely on-device (see
    // GpuIntegrator::RunEvolution) and feed it straight into the GW
    // integration (all cutoff times, for each batch) as each batch becomes
    // available -- phi never leaves device memory, unlike the old
    // Evolution(CPU)+SBatchWindow(host)->ProcessBatch(device) path. The
    // per-batch-boundary callback preserves the old periodic (time-based)
    // Format-B progress checkpoint, so an interruption here still doesn't
    // lose all GPU integration work done so far -- see checkpoint_io.h. ---
    auto t_stream = Clock::now();
    auto t_last_checkpoint = Clock::now();
    const double checkpoint_interval_seconds = checkpoint_interval_minutes * 60.0;
    integrator.RunEvolution(setup,
        [&](bool is_last_batch) {
            if (!is_last_batch &&
                elapsed(t_last_checkpoint) >= checkpoint_interval_seconds) {
                write_stream_checkpoint(checkpoint_path, integrator.MakeStreamCheckpoint(),
                                        setup, k_done, n_t_total, times_local);
                t_last_checkpoint = Clock::now();
                std::cout << "Wrote stream checkpoint to " << checkpoint_path << "\n";
            }
        });
    std::cout << "Timing phase=stream_all seconds=" << elapsed(t_stream) << "\n";

    // --- 4. Cheap finalize pass: square + k-integrate the accumulated
    // per-cutoff amplitude, then write output. No further kernel launches.
    // Writes at the GLOBAL result index (k_done + local), so this range's
    // files land alongside any earlier-completed prefix without collision;
    // safe to re-run even over partially-written files from an earlier
    // interrupted finalize pass (SaveStepResult truncates each file fresh). ---
    const auto &wlist = integrator.GetW();
    auto t_finalize = Clock::now();
    for (int local_i_t = 0; local_i_t < setup.n_t; ++local_i_t) {
        const int global_i_t = k_done + local_i_t;
        const double t = times_local[local_i_t];
        auto t_it = Clock::now();
        if (save_amplitude) {
            AmplitudeResult res = integrator.FinalizeAmplitude(local_i_t);
            SaveAmplitudeResult(output_dir, global_i_t, t, setup.gamma_ij, res);
        } else {
            std::vector<double> spectrum = integrator.Finalize(local_i_t);
            SaveStepResult(output_dir, global_i_t, t, setup.gamma_ij, wlist, spectrum);
        }
        std::cout << "Timing phase=finalize index=" << global_i_t
                  << " seconds=" << elapsed(t_it) << "\n";
    }
    std::cout << "Timing phase=finalize_all count=" << setup.n_t
              << " seconds=" << elapsed(t_finalize) << "\n";

    // --- 5. Format A: embed a plateau checkpoint in this row's true last
    // result file, so a FUTURE run can extend it to even later cutoff times
    // without re-integrating these. Only written once every requested time
    // has a result on disk (the loop above completed), matching the
    // "Format A only exists for a row that reached the end" design. ---
    auto t_checkpoint = Clock::now();
    write_plateau_checkpoint(result_path_for(n_t_total - 1), integrator.MakeCheckpoint(),
                             setup.n_w, setup.n_k);
    std::cout << "Timing phase=checkpoint seconds=" << elapsed(t_checkpoint) << "\n";

    // --- This row is now fully complete -- the Format B progress checkpoint
    // (if any) is obsolete. ---
    if (fs::exists(checkpoint_path)) fs::remove(checkpoint_path);

    std::cout << "Total: " << elapsed(t_start) << " s\n";
    return 0;

#else
    // --- Patch-fraction calibration moments: streamed, one snapshot at a time. ---
    if (patch_moments) {
        auto t_pm = Clock::now();
        PatchMoments pm(setup, patch_sc, patch_deltac);
        Evolution evo(setup, [&](const std::vector<double> &phi, double s, bool) {
            pm.add(phi, s);
        });
        SavePatchMoments(output_dir, pm.res, setup.gamma_ij, patch_sc, patch_deltac, setup.z.back());
        std::cout << "Timing phase=patch_moments seconds=" << elapsed(t_pm) << "\n";
        std::cout << "Total: " << elapsed(t_start) << " s\n";
        return 0;
    }

    // --- Wall-energy diagnostic without --save-fields: stream it the same
    // way, one snapshot at a time (with --save-fields it runs below on the
    // stored history instead, since that has to be held anyway). ---
    if (wall_energy && !save_fields) {
        auto t_we = Clock::now();
        WallEnergyResult res;
        Evolution evo(setup, [&](const std::vector<double> &phi, double s, bool) {
            append_wall_energy(res, phi, s, setup);
        });
        SaveWallEnergy(output_dir, res, setup.gamma_ij);
        std::cout << "Timing phase=wall_energy seconds=" << elapsed(t_we) << "\n";
        std::cout << "Total: " << elapsed(t_start) << " s\n";
        return 0;
    }

    // --- Wall-radius diagnostic: skip the GW integral entirely and measure
    // each snapshot as the evolution produces it, via the streaming
    // Evolution constructor -- memory stays O(n_z) instead of holding the
    // whole O(n_s*n_z) field history. ---
    if (wall_radius_scan) {
        auto t_wr = Clock::now();
        WallRadiusResult res;
        long i_snap = 0;
        Evolution evo(setup, [&](const std::vector<double> &phi, double s, bool is_last) {
            if (i_snap++ % wall_radius_every == 0 || is_last)
                append_wall_radius(res, phi, s, setup);
        });
        SaveWallRadiusScan(output_dir, res, setup.gamma_ij);
        std::cout << "Timing phase=wall_radius_scan seconds=" << elapsed(t_wr) << "\n";
        std::cout << "Total: " << elapsed(t_start) << " s\n";
        return 0;
    }

    // --- 2. Run 2D Milne evolution (CPU GSL/Filon binaries: unchanged,
    // full-history behavior — they don't have the GPU memory problem) ---
    auto t_evo = Clock::now();
    Evolution evo(setup);
    const auto &phi_snaps = evo.GetPhi();
    const double evolution_seconds = elapsed(t_evo);
    std::cout << "Evolution: " << evolution_seconds << " s\n";
    std::cout << "Timing phase=field_evolution seconds="
              << evolution_seconds << "\n";

    if (save_fields)
        SaveFields(output_dir, phi_snaps, evo.GetSlist(), setup.z);

    // --- Wall-energy diagnostic: skip the GW integral entirely, just track
    // colliding-vs-undisturbed wall energy over s and write it out. ---
    if (wall_energy) {
        auto t_we = Clock::now();
        WallEnergyResult res = compute_wall_energy(evo, setup);
        SaveWallEnergy(output_dir, res, setup.gamma_ij);
        std::cout << "Timing phase=wall_energy seconds=" << elapsed(t_we) << "\n";
        std::cout << "Total: " << elapsed(t_start) << " s\n";
        return 0;
    }

    // --- 3. Run GW integration for each time index ---
    auto t_integrator_setup = Clock::now();
    Integrator integrator(phi_snaps, setup, qual_param);
    std::cout << "Timing phase=integrator_setup_total seconds="
              << elapsed(t_integrator_setup) << "\n";
    const auto &wlist = integrator.GetW();

    const int n_t = setup.n_t;
    auto t_integ = Clock::now();
    double compute_total = 0.;
    double output_total = 0.;
    for (int i_t = 0; i_t < n_t; ++i_t) {
        auto t_it = Clock::now();
        auto t_compute = Clock::now();
        double compute_seconds = 0.;
        double output_seconds = 0.;
        if (save_amplitude) {
            AmplitudeResult res = integrator.ComputeAmplitude(i_t);
            compute_seconds = elapsed(t_compute);
            auto t_output = Clock::now();
            SaveAmplitudeResult(output_dir, i_t, setup.times[i_t], setup.gamma_ij, res);
            output_seconds = elapsed(t_output);
        } else {
            std::vector<double> spectrum = integrator.Compute(i_t);
            compute_seconds = elapsed(t_compute);
            auto t_output = Clock::now();
            SaveStepResult(output_dir, i_t, setup.times[i_t], setup.gamma_ij, wlist, spectrum);
            output_seconds = elapsed(t_output);
        }
        compute_total += compute_seconds;
        output_total += output_seconds;
        std::cout << "Time index " << i_t << " / " << n_t - 1
                  << ": " << elapsed(t_it) << " s\n";
        std::cout << "Timing phase=cutoff index=" << i_t
                  << " compute_seconds=" << compute_seconds
                  << " output_seconds=" << output_seconds
                  << " total_seconds=" << elapsed(t_it) << "\n";
    }
    std::cout << "Integration total: " << elapsed(t_integ) << " s\n";
    std::cout << "Timing phase=cutoff_all count=" << n_t
              << " compute_seconds=" << compute_total
              << " output_seconds=" << output_total
              << " total_seconds=" << elapsed(t_integ) << "\n";

    std::cout << "Total: " << elapsed(t_start) << " s\n";
    return 0;
#endif
}
