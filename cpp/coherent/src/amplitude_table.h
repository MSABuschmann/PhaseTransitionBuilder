#pragma once

#include <string>
#include <vector>

// (gamma, time) -> (amp_re, amp_im) lookup table, read from scan_cache.h5
// (built by ptbuilder.scan.run_scan(..., save_amplitude=True), see
// ptbuilder/scan.py's ScanResult.to_hdf5/from_hdf5).
struct AmplitudeTable {
    int Ng, Nt, n_w, n_k;
    std::vector<double> gamma_grid;  // [Ng] sorted ascending
    std::vector<double> t_grid;      // [Nt] scan times (shared across gamma groups)
    std::vector<double> w;           // [n_w]
    std::vector<double> k;           // [n_k] cos(theta), expected linspace(0,1,n_k)
    std::vector<double> amp_re;      // flat [Ng*Nt*n_w*n_k], index (((ig*Nt+it)*n_w+iw)*n_k+ik)
    std::vector<double> amp_im;      // same layout

    inline size_t idx(int ig, int it, int iw, int ik) const {
        return ((static_cast<size_t>(ig) * Nt + it) * n_w + iw) * n_k + ik;
    }
};

AmplitudeTable read_amplitude_table(const std::string &scan_cache_path);
