#pragma once

#include <H5Cpp.h>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "gpu_integrator.cuh"
#include "setup.h"

// Two independent checkpoint formats -- see GpuIntegrator::Checkpoint and
// GpuIntegrator::StreamCheckpoint's doc comments for what each holds and why
// they're different sizes/lifetimes.
//
// Format A ("plateau"): lets a row be extended to LATER cutoff times without
// re-integrating the ones already computed. Only ever produced once a run
// has fully completed (see GpuIntegrator::MakeCheckpoint), so it's embedded
// directly in that run's own LAST result file (result_{n_t-1:04d}.h5)
// instead of a separate file -- there's nothing to keep once the row is
// extended past it (the new run's own last result file gets its own).
//
// Format B ("stream"): lets a run resume mid-STREAM if interrupted (e.g. by
// walltime) before ever reaching its own last batch -- at that point no
// result files exist yet for this attempt, so it can't live in one. Written
// periodically (time-based) to a separate checkpoint.h5, always overwriting
// the same file (no history), and deleted once the run it belongs to
// finishes writing its result files and Format A checkpoint.

// ---------------------------------------------------------------------------
// Format A
// ---------------------------------------------------------------------------

inline void write_plateau_checkpoint(const std::string &result_path,
                                     const GpuIntegrator::Checkpoint &cp,
                                     int n_w, int n_k)
{
    // result_path already exists (SaveStepResult/SaveAmplitudeResult wrote
    // it earlier in this same run, in H5F_ACC_TRUNC mode) -- open for
    // read/write to add datasets alongside its existing t/w/spectrum.
    H5::H5File file(result_path, H5F_ACC_RDWR);
    write_attr_double(file, "plateau_seed_time", cp.seed_time);
    write_2d_flat(file, "plateau_re", cp.plateau_re,
                  static_cast<hsize_t>(n_w), static_cast<hsize_t>(n_k));
    write_2d_flat(file, "plateau_im", cp.plateau_im,
                  static_cast<hsize_t>(n_w), static_cast<hsize_t>(n_k));
}

inline bool has_plateau_checkpoint(const std::string &result_path)
{
    if (!std::filesystem::exists(result_path)) return false;
    try {
        H5::H5File file(result_path, H5F_ACC_RDONLY);
        file.openDataSet("plateau_re");
        return true;
    } catch (const H5::Exception &) {
        return false;
    }
}

inline GpuIntegrator::Checkpoint read_plateau_checkpoint(const std::string &result_path)
{
    H5::H5File file(result_path, H5F_ACC_RDONLY);
    GpuIntegrator::Checkpoint cp;
    cp.seed_time = read_attr_double(file, "plateau_seed_time");
    std::vector<hsize_t> dims;
    cp.plateau_re = read_ndarray(file, "plateau_re", dims);
    cp.plateau_im = read_ndarray(file, "plateau_im", dims);
    return cp;
}

// The row's own frequency grid as actually recorded in one of its result
// files -- used by main.cpp to hard-require that a setup.h5 extending this
// row still uses the SAME wlist, not a freshly recomputed one (only the
// vector SIZE is checked inside GpuIntegrator's constructor; a same-size but
// shifted grid would silently misinterpret the seeded plateau otherwise).
inline std::vector<double> read_result_wlist(const std::string &result_path)
{
    H5::H5File file(result_path, H5F_ACC_RDONLY);
    return read_vector(file, "w");
}

// ---------------------------------------------------------------------------
// Format B
// ---------------------------------------------------------------------------

struct LoadedStreamCheckpoint {
    GpuIntegrator::StreamCheckpoint checkpoint;

    // --- Fingerprint: everything that must match for accum_/accum_plateau_
    // to mean what they claim to. fp_k_done/fp_n_t_total pin down exactly
    // WHICH times range this checkpoint belongs to (the count of already-
    // completed earlier results, and the setup's total requested times at
    // the time this checkpoint was written) -- deliberately NOT re-derived
    // from counting result files on a resume, since this attempt's own
    // result files don't exist yet by definition. fp_times is the row's own
    // times[fp_k_done:] slice (not just t_cut/t_m/t_max/d, which fix the
    // cutoff SHAPE but not which physical times were actually requested). ---
    int fp_k_done = 0, fp_n_t_total = 0;
    double fp_d, fp_ds, fp_t_0, fp_t_cut, fp_t_m, fp_t_max;
    int fp_how_often_ds = 0, fp_cutoff_type = 0, fp_n_w = 0, fp_n_k = 0;
    std::vector<double> fp_wlist;
    std::vector<double> fp_times;
};

inline void write_stream_checkpoint(const std::string &path,
                                    const GpuIntegrator::StreamCheckpoint &sc,
                                    const Setup &setup, int k_done, int n_t_total,
                                    const std::vector<double> &times_local)
{
    // Atomic: write to a temp file in the same directory, then rename over
    // the real path -- a crash mid-write leaves the old checkpoint (or none)
    // intact instead of a torn file. Always overwrites; no history kept.
    const std::string tmp_path = path + ".tmp";
    {
        H5::H5File file(tmp_path, H5F_ACC_TRUNC);

        write_attr_int(file, "batches_done", sc.batches_done);
        write_vector(file, "accum",          sc.accum);
        write_vector(file, "accum_plateau",  sc.accum_plateau);

        write_attr_int(file,    "fp_k_done",     k_done);
        write_attr_int(file,    "fp_n_t_total",  n_t_total);
        write_attr_double(file, "fp_d",          setup.d);
        write_attr_double(file, "fp_ds",         setup.ds);
        write_attr_int(file,    "fp_how_often_ds", setup.how_often_ds);
        write_attr_int(file,    "fp_cutoff_type",  setup.cutoff_type);
        write_attr_double(file, "fp_t_0",        setup.t_0);
        write_attr_double(file, "fp_t_cut",      setup.t_cut);
        write_attr_double(file, "fp_t_m",        setup.t_m);
        write_attr_double(file, "fp_t_max",      setup.t_max);
        write_attr_int(file,    "fp_n_w",        setup.n_w);
        write_attr_int(file,    "fp_n_k",        setup.n_k);
        write_vector(file, "fp_wlist", setup.wlist);
        write_vector(file, "fp_times", times_local);
    }   // file closed/flushed here (H5::H5File destructor)

    if (std::rename(tmp_path.c_str(), path.c_str()) != 0)
        throw std::runtime_error("write_stream_checkpoint: rename failed for " + path);
}

// Returns std::nullopt if no checkpoint exists at `path`, OR if it exists
// but fails to parse (e.g. a torn write despite the rename above, or a
// leftover from an incompatible version) -- either way, treated as "nothing
// usable to resume from" rather than a fatal error, since Format B is purely
// an optimization: discarding it just means redoing the streaming work, it
// never risks a wrong result the way a bad Format A seed would.
inline std::optional<LoadedStreamCheckpoint> try_read_stream_checkpoint(const std::string &path)
{
    if (!std::filesystem::exists(path)) return std::nullopt;
    try {
        H5::H5File file(path, H5F_ACC_RDONLY);
        LoadedStreamCheckpoint lc;

        lc.checkpoint.batches_done  = read_attr_int(file, "batches_done");
        lc.checkpoint.accum         = read_vector(file, "accum");
        lc.checkpoint.accum_plateau = read_vector(file, "accum_plateau");

        lc.fp_k_done       = read_attr_int(file, "fp_k_done");
        lc.fp_n_t_total    = read_attr_int(file, "fp_n_t_total");
        lc.fp_d            = read_attr_double(file, "fp_d");
        lc.fp_ds           = read_attr_double(file, "fp_ds");
        lc.fp_how_often_ds = read_attr_int(file, "fp_how_often_ds");
        lc.fp_cutoff_type  = read_attr_int(file, "fp_cutoff_type");
        lc.fp_t_0          = read_attr_double(file, "fp_t_0");
        lc.fp_t_cut        = read_attr_double(file, "fp_t_cut");
        lc.fp_t_m          = read_attr_double(file, "fp_t_m");
        lc.fp_t_max        = read_attr_double(file, "fp_t_max");
        lc.fp_n_w          = read_attr_int(file, "fp_n_w");
        lc.fp_n_k          = read_attr_int(file, "fp_n_k");
        lc.fp_wlist        = read_vector(file, "fp_wlist");
        lc.fp_times        = read_vector(file, "fp_times");

        return lc;
    } catch (...) {
        return std::nullopt;
    }
}

// True iff `lc` was written for the EXACT same (setup, k_done, n_t_total,
// times_local) as the caller is about to run -- i.e. it's safe to resume
// accum_/accum_plateau_ from it. False for anything else (a stale checkpoint
// from before the row was extended further, a different resolution, ...);
// the caller should discard it and start this range's streaming from batch 0.
inline bool stream_checkpoint_matches(const LoadedStreamCheckpoint &lc, const Setup &setup,
                                      int k_done, int n_t_total,
                                      const std::vector<double> &times_local)
{
    auto close_enough = [](double a, double b) {
        return std::abs(a - b) <= 1e-9 * std::max(1.0, std::abs(a));
    };

    if (lc.fp_k_done != k_done)          return false;
    if (lc.fp_n_t_total != n_t_total)    return false;
    if (!close_enough(lc.fp_d, setup.d))         return false;
    if (!close_enough(lc.fp_ds, setup.ds))       return false;
    if (lc.fp_how_often_ds != setup.how_often_ds) return false;
    if (lc.fp_cutoff_type != setup.cutoff_type)   return false;
    if (!close_enough(lc.fp_t_0, setup.t_0))     return false;
    if (!close_enough(lc.fp_t_cut, setup.t_cut)) return false;
    if (!close_enough(lc.fp_t_m, setup.t_m))     return false;
    if (!close_enough(lc.fp_t_max, setup.t_max)) return false;
    if (lc.fp_n_w != setup.n_w) return false;
    if (lc.fp_n_k != setup.n_k) return false;

    if (lc.fp_wlist.size() != setup.wlist.size()) return false;
    for (std::size_t i = 0; i < lc.fp_wlist.size(); ++i)
        if (!close_enough(lc.fp_wlist[i], setup.wlist[i])) return false;

    if (lc.fp_times.size() != times_local.size()) return false;
    for (std::size_t i = 0; i < lc.fp_times.size(); ++i)
        if (!close_enough(lc.fp_times[i], times_local[i])) return false;

    return true;
}
