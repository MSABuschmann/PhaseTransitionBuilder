#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

struct WeightsSetup {
    double L;
    double rout_0, rin_0;             // outer/inner wall radii at t=0
    double collision_r0;              // selected mid/outer surface at t=0
    std::string collision_radius;      // "mid" or "out"
    int    n_t, n_b;
    std::vector<double>          t;    // time points  [n_t]
    std::vector<double>          R;    // selected collision radius R(t) [n_t]
    std::vector<Eigen::Vector3d> pos;  // positions    [n_b]
};

// Compute the collision-arc weight for one candidate periodic image of a
// bubble pair, given the pair's own already-shifted absolute positions c0,
// c1 (c0 is always the real bubble's own, unshifted position; c1 may be a
// periodic ghost copy of the other bubble).  real_i, real_j are the
// underlying real bubble indices -- used to exclude this pair from its own
// third-body occlusion check (occlusion still searches all periodic images
// of every other real bubble).
// weight[i_t] = fraction of collision-circle arc NOT blocked by other bubbles.
// Returns false (weight left all-zero) if this image never collides.
bool ComputePairWeight(const Eigen::Vector3d &c0, const Eigen::Vector3d &c1,
                       int real_i, int real_j,
                       const WeightsSetup &setup,
                       std::vector<double> &weight);

// Search every periodic image of every bubble pair (including a bubble
// against its own periodic image) for collisions.  Only the "other" bubble
// is ever shifted (c0 = pos[i] always, c1 = pos[j] + one of the 27 periodic
// offsets) -- a bijection onto the 27 physically distinct relative
// configurations of a pair, so nothing is ever double-counted and no
// box-membership filtering is needed.  A real bubble pair that collides
// more than once -- once directly, once again through the periodic
// boundary -- produces more than one entry here, each with its own
// distance (pair_d) and weight curve.
void FindAllCollisionWeights(const WeightsSetup &setup,
                              std::vector<double> &flat_weights,
                              std::vector<std::pair<int,int>> &collision_pairs,
                              std::vector<double> &pair_d);
