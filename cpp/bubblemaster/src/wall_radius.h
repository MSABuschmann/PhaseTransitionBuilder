#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "setup.h"

// ---------------------------------------------------------------------------
// Numerical wall-radius diagnostic.
//
// Locates the wall from the ACTUAL simulated field's gradient-energy density
// (dphi/dz)^2, not from the static bounce profile's tanh(phi0)-fraction
// contours (rin_0/rmid_0/rout_0) -- those are pinned to the field's value at
// nucleation and, for a bounce whose core is far from true vacuum, drift
// into flat, energetically-irrelevant territory once the field has relaxed
// (see the wall-energy pilot investigation). The steepest part of the
// profile can even sit inside the light cone, in the dynamically-relaxing
// interior, which no closed-form static-profile formula captures -- hence
// measuring it directly from the simulated field here instead.
//
// Only the UNDISTURBED side (z > z_center) is measured: it never
// experiences collision effects, so a single run spans the full range of
// proper time s (and hence local wall gamma) needed, with no separate scan
// over gamma required.
// ---------------------------------------------------------------------------

struct WallRadiusResult {
    std::vector<double> s;
    std::vector<double> R_mid, R_in, R_out;
    // 1.0/0.0: false if a half-max crossing ran off the domain edge instead
    // of being found within it (e.g. the wall has moved past the padded
    // region, or the domain is too small for this snapshot).
    std::vector<double> valid;
};

// Measure one snapshot phi(z) at proper time s and append it to res. Only
// ever reads this one snapshot, so it can be fed straight from Evolution's
// streaming SnapshotSink -- the full field history (O(n_s*n_z), hundreds of
// GB at high gamma) is never needed.
inline void append_wall_radius(WallRadiusResult &res, const std::vector<double> &phi,
                               double s, const Setup &setup) {
    const double z_center = setup.d / 2.0;
    const double dz       = std::abs(setup.z[1] - setup.z[0]);
    const int n_z         = static_cast<int>(setup.z.size());

    // Restrict the search to z > z_center (the undisturbed side); need
    // iz-1/iz+1 for the gradient stencil.
    int iz_lo = static_cast<int>(std::ceil((z_center - setup.z[0]) / dz)) + 1;
    iz_lo = std::max(iz_lo, 1);
    int iz_hi = n_z - 2;

    if (iz_hi < iz_lo) {
        res.s.push_back(s);
        res.R_mid.push_back(0.); res.R_in.push_back(0.); res.R_out.push_back(0.);
        res.valid.push_back(0.);
        return;
    }

    std::vector<double> g2(iz_hi - iz_lo + 1);
    int i_peak = iz_lo;
    double peak_val = -1.;
    for (int iz = iz_lo; iz <= iz_hi; ++iz) {
        const double grad = (phi[iz + 1] - phi[iz - 1]) / (2. * dz);
        const double v = grad * grad;
        g2[iz - iz_lo] = v;
        if (v > peak_val) { peak_val = v; i_peak = iz; }
    }

    const double half = peak_val / 2.;
    bool valid = true;

    int i_out = -1;
    for (int iz = i_peak; iz <= iz_hi; ++iz)
        if (g2[iz - iz_lo] < half) { i_out = iz; break; }
    int i_in = -1;
    for (int iz = i_peak; iz >= iz_lo; --iz)
        if (g2[iz - iz_lo] < half) { i_in = iz; break; }

    if (i_out < 0) { i_out = iz_hi; valid = false; }
    if (i_in  < 0) { i_in  = iz_lo; valid = false; }

    res.s.push_back(s);
    res.R_mid.push_back(setup.z[i_peak] - z_center);
    res.R_in.push_back(setup.z[i_in]    - z_center);
    res.R_out.push_back(setup.z[i_out]  - z_center);
    res.valid.push_back(valid ? 1. : 0.);
}
