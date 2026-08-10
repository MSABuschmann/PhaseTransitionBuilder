#pragma once

#include <vector>

#include "amplitude_table.h"
#include "geometry.h"
#include "io.h"
#include "quadrature.h"

// Pairs whose weight exceeds `threshold` anywhere in [t=1 .. t_max] (matches
// notebook 06's active_pairs_for). Returns indices into wd's pair arrays.
std::vector<int> active_pairs_for(const WeightsData &wd, double t_max, double threshold = 0.01);

// Coherent GW power spectrum reconstruction for a single t_max, matching
// notebook 06's coherent_sum_n64 formula exactly. Returns P_coh[n_w],
// already normalized by w^3 * 2*pi.
std::vector<double> compute_coherent_spectrum(
    double t_max, const WeightsData &wd, const AmplitudeTable &amp,
    const PairGeometry &geo, const SphereQuadrature &quad,
    int t_chunk, double threshold = 0.01);
