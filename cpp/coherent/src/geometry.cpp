#include "geometry.h"

#include <cmath>

namespace {

// Python-style modulo (result in [0, L)), unlike std::fmod which can be
// negative for negative inputs.
inline double mod_L(double x, double L) {
    return x - L * std::floor(x / L);
}

} // namespace

PairGeometry compute_pair_geometry(const BubbleSetup &bubbles, const WeightsData &wd, double L) {
    PairGeometry geo;
    geo.axis.resize(wd.n_pairs);
    geo.center.resize(wd.n_pairs);

    for (int p = 0; p < wd.n_pairs; ++p) {
        const Eigen::Vector3d &pi = bubbles.pos[wd.pair_i[p]];
        const Eigen::Vector3d &pj = bubbles.pos[wd.pair_j[p]];

        Eigen::Vector3d dp = pi - pj;
        for (int c = 0; c < 3; ++c)
            dp[c] -= L * std::round(dp[c] / L);
        geo.axis[p] = dp.normalized();

        Eigen::Vector3d mid = 0.5 * (pi + pj);
        for (int c = 0; c < 3; ++c)
            mid[c] = mod_L(mid[c], L);
        geo.center[p] = mid;
    }
    return geo;
}
