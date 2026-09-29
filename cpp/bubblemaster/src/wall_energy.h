#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "evolution.h"
#include "setup.h"
#include "wall_radius.h"

// ---------------------------------------------------------------------------
// Wall-energy diagnostic.
//
// Only ONE bubble is explicitly simulated, centred at z=d/2; z=0 is a
// reflecting boundary standing in for its mirror-image collision partner
// (see evolve_pi_kernel/EvolvePi's iz==0 clamp) -- so the wall "collides
// with itself" at z=0 rather than with a second, separately-tracked bubble.
//
// Track the energy in two fixed physical windows as the simulation evolves
// past that collision:
//   - the COLLIDING side: the window that geometrically sweeps toward the
//     reflecting boundary (decreasing z, toward z=0), i.e. where the naive
//     isolated-bubble picture is expected to break down once the wall
//     reaches its own reflection.
//   - the UNDISTURBED side: the window sweeping away from the boundary
//     (increasing z), which never reflects and so serves as a same-run,
//     same-numerics control for what "no collision happened" energy looks
//     like at the same s.
//
// Both windows span [R_in(s), R_out(s)]: the numerically measured wall on
// the UNDISTURBED side at the same snapshot (the half-max points of the
// gradient energy (dphi/dz)^2 around its peak -- see wall_radius.h), not the
// textbook sqrt(r_0^2 + s^2) contours. The colliding window is that same
// measured wall mirrored toward z=0 -- i.e. "where the wall would be had no
// collision happened"; only the field values used to compute the energy
// density are the true, simulated ones there.
//
// Energy density: only the gradient energy that sources gravitational waves
// (the anisotropic stress d_i phi d_j phi, which on the collision axis is
// just (dphi/dz)^2):
//   epsilon(s,z) = s^2 * 0.5*(dphi/dz)^2
// The kinetic term 0.5*pi^2 and the potential V(phi) are deliberately left
// out: V < 0 in the true vacuum would count released vacuum energy
// (negative, and dominant at small lambda_bar) inside the window. The s^2
// Milne volume factor is common to both windows at the same s, so it
// cancels exactly in any same-s colliding/undisturbed ratio.
//
// NOTE on NaN: this file deliberately never uses NaN as a sentinel. This
// binary is built with -ffast-math (-ffinite-math-only), which licenses the
// compiler to assume no NaN/Inf ever appears anywhere in the program; a
// std::numeric_limits<double>::quiet_NaN() sentinel violates that
// assumption and was observed to trigger a hardware FP trap (SIGTRAP) on
// this platform when consumed downstream. "No valid window" is instead
// signalled via an explicit bool, with the energy value left at a safe 0.0.
// ---------------------------------------------------------------------------

struct WallWindow {
    double z_lo, z_hi;
};

// Wall window for radii r_in_s <= r_out_s at proper time s. colliding=false: the
// undisturbed outer window (moves toward increasing z, away from the
// boundary). colliding=true: the window moving toward the reflecting
// boundary at z=0 -- and, once its free (unfolded) trajectory would put it
// at z<0, folded back into the simulated domain z>=0. The fold is exact,
// not an approximation: the reflecting boundary condition IS the mirror
// image collision partner (phi(-z,s) == phi(z,s) for all s, since the two
// colliding bubbles are identical and the PDE preserves that symmetry), so
// "the wall's free trajectory at z<0" and "its mirror image at |z|" are the
// same physical field values, and the simulated domain only stores the
// latter.
inline WallWindow wall_window(double z_center, double r_in_s, double r_out_s,
                              bool colliding) {
    if (!colliding) {
        return {z_center + r_in_s, z_center + r_out_s};
    }
    const double a = z_center - r_in_s;   // nearer edge (smaller radius)
    const double b = z_center - r_out_s;  // farther edge (larger radius), b <= a
    if (b >= 0.) return {b, a};                       // still fully on the physical side
    if (a <= 0.) return {-a, -b};                      // fully folded past the boundary
    return {0., std::max(a, -b)};                      // straddles z=0: union of both folded pieces
}
// NOTE: the straddling case above is only the window's extent, for plotting.
// Its energy must NOT be integrated over that union (the overlap would be
// counted once, shrinking the window) -- see colliding_energy below.

// Trapezoidal integral of epsilon(s,z) = s^2*0.5*phi_z^2
// over the grid points falling inside [z_lo, z_hi]. Sets *valid=false (and
// returns 0.0) if the window doesn't overlap at least two grid points (e.g.
// it has run off the domain, or collapsed below grid resolution). z[0] = 0
// is the reflecting boundary, where phi_z = 0 by the mirror symmetry
// phi(-z) = phi(z).
inline double window_energy(const std::vector<double> &phi,
                            const std::vector<double> &z, double dz, double s,
                            double z_lo, double z_hi, bool *valid) {
    *valid = false;
    const int n_z = static_cast<int>(z.size());
    if (n_z < 3) return 0.;

    // Window edges typically sit exactly on grid points (the measured wall
    // radii are grid-point positions); the tolerance keeps ceil/floor from
    // flipping on round-off, which would shift a mirrored window by one
    // grid point relative to its partner.
    const double tol = 1e-6;
    int i_lo = static_cast<int>(std::ceil((z_lo - z[0]) / dz - tol));
    int i_hi = static_cast<int>(std::floor((z_hi - z[0]) / dz + tol));
    i_lo = std::max(i_lo, 0);
    i_hi = std::min(i_hi, n_z - 2);        // need iz+1
    if (i_hi - i_lo < 1) return 0.;

    *valid = true;
    const double s2 = s * s;
    double integral = 0.;
    double eps_prev = 0.;
    for (int iz = i_lo; iz <= i_hi; ++iz) {
        const double phi_z = (iz == 0) ? 0. : (phi[iz+1] - phi[iz-1]) / (2. * dz);
        const double eps = s2 * 0.5*phi_z*phi_z;
        if (iz > i_lo) integral += 0.5 * (eps + eps_prev) * dz;
        eps_prev = eps;
    }
    return integral;
}

// Energy in the colliding window, i.e. the measured wall [r_in_s, r_out_s]
// mirrored toward z=0: the UNFOLDED window [b, a] = [z_center - r_out_s,
// z_center - r_in_s], always the same width as the undisturbed one. The part
// at z<0 lies beyond the reflecting boundary; by phi(-z) = phi(z) its energy
// equals that of its mirror image [-a, -b] inside the domain, so
//   E[b, a] = E[max(b,0), a] + E[max(-a,0), -b]
// covers not-yet-reached, straddling and fully-past windows alike.
inline double colliding_energy(const std::vector<double> &phi,
                               const std::vector<double> &z, double dz, double s,
                               double z_center, double r_in_s, double r_out_s,
                               bool *valid) {
    const double a = z_center - r_in_s;
    const double b = z_center - r_out_s;
    bool v1 = false, v2 = false;
    double e = 0.;
    if (a > 0.) e += window_energy(phi, z, dz, s, std::max(b, 0.), a, &v1);
    if (b < 0.) e += window_energy(phi, z, dz, s, std::max(-a, 0.), -b, &v2);
    *valid = v1 || v2;
    return e;
}

struct WallEnergyResult {
    std::vector<double> s;
    std::vector<double> e_colliding, e_undisturbed;
    // 1.0/0.0 (not bool -- written straight to HDF5 as a plain double
    // dataset) marking which entries are physically meaningful, i.e. where
    // the window actually overlapped the domain.
    std::vector<double> valid_colliding, valid_undisturbed;
    // Window boundaries at each s, for overlaying on a z-vs-s field plot.
    std::vector<double> z_lo_colliding, z_hi_colliding;
    std::vector<double> z_lo_undisturbed, z_hi_undisturbed;

    // Collision = first snapshot where the measured R_mid reaches the
    // midplane (d/2) -- the same criterion the weights binary uses. From then
    // on the colliding window's width is frozen at its value then (see
    // append_wall_energy). s_collision < 0 until it happens.
    double s_collision = -1.;
    double d_in_collision = 0., d_out_collision = 0.;   // R_mid-R_in, R_out-R_mid at collision
};

// Measure both windows' energy for one snapshot phi(z) at proper time s and
// append it to res. Only reads this one snapshot, so it can be fed straight
// from Evolution's streaming SnapshotSink.
inline void append_wall_energy(WallEnergyResult &res, const std::vector<double> &phi,
                               double s, const Setup &setup) {
    const double z_center = setup.d / 2.0;   // the one explicitly simulated bubble
    const double dz = std::abs(setup.z[1] - setup.z[0]);

    WallRadiusResult wr;
    append_wall_radius(wr, phi, s, setup);
    const double R_mid = wr.R_mid[0], R_in = wr.R_in[0], R_out = wr.R_out[0];

    // After the collision there is no wall left on the colliding side to
    // Lorentz-contract: keep the colliding window's offsets from R_mid fixed
    // at their collision values, and only move it along with the mirrored
    // centre R_mid(s) of the undisturbed wall.
    if (res.s_collision < 0. && R_mid >= z_center) {
        res.s_collision     = s;
        res.d_in_collision  = R_mid - R_in;
        res.d_out_collision = R_out - R_mid;
    }
    const bool collided = res.s_collision >= 0.;
    const double rc_in  = collided ? R_mid - res.d_in_collision  : R_in;
    const double rc_out = collided ? R_mid + res.d_out_collision : R_out;

    WallWindow w_col = wall_window(z_center, rc_in, rc_out, /*colliding=*/true);
    WallWindow w_und = wall_window(z_center, R_in, R_out, /*colliding=*/false);

    bool valid_col = false, valid_und = false;
    double e_col = colliding_energy(phi, setup.z, dz, s, z_center,
                                    rc_in, rc_out, &valid_col);
    double e_und = window_energy(phi, setup.z, dz, s,
                                 w_und.z_lo, w_und.z_hi, &valid_und);

    res.s.push_back(s);
    res.e_colliding.push_back(e_col);
    res.e_undisturbed.push_back(e_und);
    res.valid_colliding.push_back(valid_col ? 1.0 : 0.0);
    res.valid_undisturbed.push_back(valid_und ? 1.0 : 0.0);
    res.z_lo_colliding.push_back(w_col.z_lo);
    res.z_hi_colliding.push_back(w_col.z_hi);
    res.z_lo_undisturbed.push_back(w_und.z_lo);
    res.z_hi_undisturbed.push_back(w_und.z_hi);
}

// Same, over every snapshot already held by `evo` (built with Evolution's
// default, whole-history constructor -- only used together with --save-fields).
inline WallEnergyResult compute_wall_energy(const Evolution &evo, const Setup &setup) {
    WallEnergyResult res;
    const auto &phi_snaps = evo.GetPhi();
    const auto &slist     = evo.GetSlist();
    for (std::size_t i = 0; i < slist.size(); ++i)
        append_wall_energy(res, phi_snaps[i], slist[i], setup);
    return res;
}
