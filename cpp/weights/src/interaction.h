#pragma once

#include <Eigen/Dense>
#include <vector>

struct WeightsSetup {
    double L;
    double rout_0, rin_0;             // outer/inner wall radii at t=0
    int    n_t, n_b;
    std::vector<double>          t;    // time points  [n_t]
    std::vector<double>          R;    // outer wall radius R_out(t) [n_t]
    std::vector<Eigen::Vector3d> pos;  // positions    [n_b]
};

// Compute arc-length weight for a colliding pair (test, other) at each time.
// Pair geometry and occlusion use periodic images in the box of side L.
// weight[i_t] = fraction of collision-circle arc NOT blocked by other bubbles.
void ComputePairWeight(size_t test, size_t other,
                       const WeightsSetup &setup,
                       std::vector<double> &weight);

// Compute all pairwise weights and store in flat vector (n_pairs * n_t).
// Also fills collision_pairs with (i, j) indices of colliding pairs.
void FindAllCollisionWeights(const WeightsSetup &setup,
                              std::vector<double> &flat_weights,
                              std::vector<std::pair<int,int>> &collision_pairs);
