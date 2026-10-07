#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "setup.h"
#include "wall_radius.h"

// ---------------------------------------------------------------------------
// Patch-fraction calibration moments.
//
// For each native snapshot, the zero-spatial-frequency column moments of the
// collision wake window |z| < z_cut(s) = 4*delta_c + max(s - s_c, 0):
//   A(s)        = int (dphi/dz)^2 dz, from edge differences at s,
//   B(s - ds/2) = int (dphi/ds)^2 dz, from the difference of consecutive
//                 native snapshots (so B lives half a snapshot earlier).
// z = 0 is the collision plane (reflecting boundary), so both are integrated
// over [0, z_cut] and doubled to cover both halves of the pair. s_c and
// delta_c are inputs: the textbook or the measured-wall convention is chosen
// by the caller. Guards (window reaching the outer wall or the domain edge)
// are applied afterwards, from the recorded z_cut, R_in and z_max.
// ---------------------------------------------------------------------------

struct PatchMomentsResult {
    std::vector<double> s_A, A, zcut_A, R_in_A;   // R_in_A: measured undisturbed inner wall radius
    std::vector<double> s_B, B, zcut_B;
};

// Integral over [0, zc] of the piecewise-linear interpolant through
// (x[k], y[k]) (x increasing, x[0] <= 0 handled by clamping), including the
// partial cell at zc.
inline double integrate_to(const std::vector<double> &x, const std::vector<double> &y, double zc) {
    double sum = 0.;
    for (std::size_t k = 0; k + 1 < x.size(); ++k) {
        const double a = std::max(x[k], 0.), b = std::min(x[k+1], zc);
        if (b <= a) { if (x[k] >= zc) break; continue; }
        auto lerp = [&](double z) { return y[k] + (y[k+1] - y[k]) * (z - x[k]) / (x[k+1] - x[k]); };
        sum += 0.5 * (lerp(a) + lerp(b)) * (b - a);
    }
    return sum;
}

class PatchMoments {
public:
    PatchMoments(const Setup &setup, double s_c, double delta_c)
        : setup_(setup), s_c_(s_c), delta_c_(delta_c),
          dz_(std::abs(setup.z[1] - setup.z[0])) {}

    double zcut(double s) const { return 4. * delta_c_ + std::max(s - s_c_, 0.); }

    void add(const std::vector<double> &phi, double s) {
        const std::vector<double> &z = setup_.z;
        const int n = static_cast<int>(z.size());

        // A at s: (dphi/dz)^2 on edges z_{i+1/2}; mirror edge at -dz/2 (same
        // value by the reflection symmetry) makes the interpolant flat on [0, dz/2].
        std::vector<double> xe(n), ye(n);
        for (int i = 0; i + 1 < n; ++i) {
            const double g = (phi[i+1] - phi[i]) / dz_;
            xe[i+1] = 0.5 * (z[i] + z[i+1]);
            ye[i+1] = g * g;
        }
        xe[0] = -xe[1]; ye[0] = ye[1];
        const double zc = zcut(s);
        WallRadiusResult wr;
        append_wall_radius(wr, phi, s, setup_);
        res.s_A.push_back(s);
        res.A.push_back(2. * integrate_to(xe, ye, zc));
        res.zcut_A.push_back(zc);
        res.R_in_A.push_back(wr.R_in[0]);

        // B at s - ds/2 from consecutive native snapshots, on the nodes.
        if (!prev_.empty()) {
            const double ds = s - s_prev_, sb = s - 0.5 * ds;
            std::vector<double> yb(n);
            for (int i = 0; i < n; ++i) {
                const double v = (phi[i] - prev_[i]) / ds;
                yb[i] = v * v;
            }
            res.s_B.push_back(sb);
            res.B.push_back(2. * integrate_to(z, yb, zcut(sb)));
            res.zcut_B.push_back(zcut(sb));
        }
        prev_ = phi;
        s_prev_ = s;
    }

    PatchMomentsResult res;

private:
    const Setup &setup_;
    double s_c_, delta_c_, dz_;
    std::vector<double> prev_;
    double s_prev_ = 0.;
};
