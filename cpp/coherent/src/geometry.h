#pragma once

#include <Eigen/Dense>
#include <vector>

#include "io.h"

// Per-pair collision axis and center-of-mass position, using minimum-image
// PBC -- same convention as cpp/weights/src/main.cpp's gamma computation.
struct PairGeometry {
    std::vector<Eigen::Vector3d> axis;    // [n_pairs] unit vector, min-image (i-j)
    std::vector<Eigen::Vector3d> center;  // [n_pairs] 0.5*(pos_i+pos_j) mod L
};

PairGeometry compute_pair_geometry(const BubbleSetup &bubbles, const WeightsData &wd, double L);
