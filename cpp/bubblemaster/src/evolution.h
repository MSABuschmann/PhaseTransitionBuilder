#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "setup.h"

class Evolution {
public:
    explicit Evolution(const Setup &setup);

    // Called once per saved snapshot, in increasing-s order (including the
    // initial s=0 snapshot). `is_last` is true exactly once, on the final
    // snapshot that will ever be saved for this run — computed up front from
    // n_steps/how_often_ds, so the caller doesn't need foreknowledge of the
    // total snapshot count.
    //
    // Streaming construction: does NOT populate phicomplete/slist (they stay
    // empty) — GetPhi()/GetSlist() must not be called on an instance built
    // this way. Used by the GPU binary's bounded-memory batching path and
    // the CPU binary's --wall-radius-scan;
    // the physics recurrence itself is unchanged (Evolve() only ever reads
    // the *current* phi/pi arrays, never past snapshots).
    using SnapshotSink =
        std::function<void(const std::vector<double> &phi, double s, bool is_last)>;
    Evolution(const Setup &setup, SnapshotSink sink);

    // Returns the saved field snapshots: phicomplete[i_s][i_z]
    const std::vector<std::vector<double>> &GetPhi()   const { return phicomplete; }
    // pi = dphi/ds at the same snapshots as GetPhi() -- only populated by
    // the default (whole-history) constructor; empty for the streaming
    // constructor (unused by any current caller -- see its own comment).
    const std::vector<std::vector<double>> &GetPi()    const { return picomplete; }
    const std::vector<double>              &GetSlist()  const { return slist; }
    double                                  GetDS()     const { return ds_out; }

private:
    Evolution(const Setup &setup, SnapshotSink sink, bool);  // delegate target

    void Evolve();
    void EvolvepiFirstHalfStep(int n_baby);
    double EvolvePi(int i_z, double s, double step) const;

    int    n_z, how_often_ds, baby_steps, n_steps;
    double ds, smax, dz, d;

    const Potential *potential;   // non-owning; Setup owns it

    std::vector<double> z;
    std::vector<double> phi;
    std::vector<double> pi;
    std::vector<double> slist;    // s values for saved snapshots

    // snapshots: phicomplete[i_snapshot][i_z], picomplete[i_snapshot][i_z]
    std::vector<std::vector<double>> phicomplete;
    std::vector<std::vector<double>> picomplete;

    double ds_out; // effective ds between saved snapshots = ds * how_often_ds

    SnapshotSink sink_;
    int last_saved_i_ = 0;   // = (n_steps/how_often_ds)*how_often_ds
};
