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
    // Measured wall width R_out(t) - R_in(t) [n_t], set together with R from a
    // --wall-radius file; empty otherwise (then the persistence model uses the
    // textbook dt_wall).
    std::vector<double>          wall_width;
    std::vector<Eigen::Vector3d> pos;  // positions    [n_b]

    // Back-of-envelope test of "a third wall doesn't cut off the source
    // instantly on geometric contact": when computing whether a THIRD
    // bubble occludes part of a pair's own collision ring, that third
    // bubble's radius is evaluated occlusion_delay time units in the past
    // (R(t - occlusion_delay) instead of R(t)) -- i.e. the third wall is
    // treated as if it arrived occlusion_delay later than it geometrically
    // did. Zero reproduces the original instantaneous-cutoff behaviour
    // exactly. Only affects the OCCLUDING bubble's own radius, never the
    // two primary colliding bubbles' own R0/R1.
    double occlusion_delay = 0.;

    // Per-segment exponential persistence: instead of instantly zeroing an
    // angular patch of the collision ring the moment a third bubble's wall
    // geometrically reaches it, the patch's contribution decays as
    // exp(-(t - t_touch)/tau), with tau = persistence_alpha * dt_wall(R2 at
    // t_touch) -- dt_wall being that occluding bubble's own wall-crossing
    // time (sqrt(R2^2-rin_0^2) - sqrt(R2^2-rout_0^2)) at the moment of
    // contact. 0 disables this and reproduces the original instantaneous
    // hard-cutoff behaviour exactly (cheap path). Nonzero enables a
    // substantially more expensive per-angular-bin path -- see
    // persistence_n_bins.
    double persistence_alpha = 0.;
    int    persistence_n_bins = 360;

    // Alternative persistence timescale: if > 0, tau = persistence_alpha *
    // max(dt_wall(R2 at touch), gamma_local(R2 at touch)/persistence_m_true)
    // instead of persistence_alpha * dt_wall(R2 at touch) alone. Physical
    // motivation: the source persists until whichever of two processes
    // takes longer -- the wall's own geometric, Lorentz-CONTRACTED crossing
    // time (dt_wall, shrinks with gamma, dominates at low gamma) or the
    // field's time-DILATED intrinsic relaxation (proper timescale set by
    // the true-vacuum curvature mass, stretched by the occluding bubble's
    // own local Lorentz factor at contact; grows with gamma, dominates at
    // high gamma). persistence_m_true=0 (default) keeps the original
    // dt_wall-only model.
    double persistence_m_true = 0.;

    // Optional post-collision ringing term (rough model from the pair
    // wall-energy runs): a covered patch then contributes
    //   (1 - ring_amp) exp(-dt/tau) + ring_amp exp(-dt/tau_ring),
    //   tau_ring = ring_beta * gamma_local(R2 at t_touch) / persistence_m_true.
    // ring_amp = 0 (default) keeps the single-exponential model.
    double ring_amp  = 0.;
    double ring_beta = 0.;
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
