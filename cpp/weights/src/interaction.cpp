#include "interaction.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Geometric helpers (carried and adapted from reference interaction.cpp)
// ---------------------------------------------------------------------------

static double plane_dist(const Eigen::Vector3d &nxhat,
                          const Eigen::Vector3d &cx,
                          const Eigen::Vector3d &c) {
    return nxhat.dot(c - cx);
}

static void other_circle(const Eigen::Vector3d &c2, double R2,
                          const Eigen::Vector3d &nxhat,
                          const Eigen::Vector3d &cx,
                          Eigen::Vector3d &cs, double &Rs) {
    double rho = plane_dist(nxhat, cx, c2);
    cs = c2 - rho * nxhat;
    Rs = (std::abs(rho) < R2) ? std::sqrt(R2*R2 - rho*rho) : 0.;
}

static void get_cut_range(const Eigen::Vector3d &cs, double Rs,
                           const Eigen::Vector3d &cx, double Rx,
                           const Eigen::Vector3d &u, const Eigen::Vector3d &v,
                           double &theta0, double &theta1) {
    Eigen::Vector3d P  = cs - cx;
    double px   = P.dot(u), py = P.dot(v);
    double d    = std::sqrt(px*px + py*py);
    double tmid = std::atan2(py, px);
    double theta;
    if (Rs + Rx < d) {
        theta = 0.;
    } else if (Rs - Rx > d) {
        theta = M_PI;
    } else {
        double tmp = d*d - Rs*Rs + Rx*Rx;
        double rad = 4.*d*d*Rx*Rx - tmp*tmp;
        if (rad < 0.) { theta0 = theta1 = 0.; return; }
        double a = std::sqrt(rad) / d;
        theta = std::asin(a / (2.*Rx));
        if (Rs >= std::sqrt(d*d + Rx*Rx)) theta = M_PI - theta;
    }
    theta0 = tmid - theta;
    theta1 = tmid + theta;
}

static void normalize_intervals(
        std::vector<std::pair<double,double>> &out,
        const std::vector<std::pair<double,double>> &inp,
        double period, double eps = 1e-10) {
    auto mod_pos = [&](double x) -> double {
        double m = std::fmod(x, period);
        if (m < 0.) m += period;
        if (m >= period) m = 0.;
        return m;
    };
    for (const auto &[s, e] : inp) {
        double sm = mod_pos(s), w = e - s;
        if (std::abs(w - period) < eps) { out.emplace_back(0., period); continue; }
        double em = sm + w;
        if (em > period) { out.emplace_back(sm, period); out.emplace_back(0., em-period); }
        else              { out.emplace_back(sm, em); }
    }
}

static double remaining_arc(
        const std::vector<std::pair<double,double>> &remove_sections,
        double period, double eps = 1e-10) {
    std::vector<std::pair<double,double>> rem, merged;
    normalize_intervals(rem, remove_sections, period, eps);
    std::sort(rem.begin(), rem.end(),
              [](const auto &a, const auto &b) {
                  return a.first != b.first ? a.first < b.first
                                            : a.second < b.second;
              });
    for (const auto &[s, e] : rem) {
        if (merged.empty() || s > merged.back().second)
            merged.emplace_back(s, e);
        else if (e > merged.back().second)
            merged.back().second = e;
    }
    double arc = 0., cur = 0.;
    for (const auto &[rs, re] : merged) {
        if (cur < rs) arc += rs - cur;
        cur = re;
        if (cur >= period) break;
    }
    if (cur < period) arc += period - cur;
    return arc;
}

// ---------------------------------------------------------------------------
// Main pair-weight computation
// ---------------------------------------------------------------------------

bool ComputePairWeight(const Eigen::Vector3d &c0, const Eigen::Vector3d &c1,
                       int real_i, int real_j,
                       const WeightsSetup &setup,
                       std::vector<double> &weight) {
    const int n_t = setup.n_t;
    weight.assign(n_t, 0.);

    Eigen::Vector3d nxhat = (c1 - c0).normalized();
    double d = (c1 - c0).norm();
    if (d < 1e-9) return false;   // coincident images, not a real pair

    // Build a frame (u, v) perpendicular to nxhat
    Eigen::Vector3d ax = {0.12345, 0.42134625, 0.14542456};
    Eigen::Vector3d u  = ax.cross(nxhat).normalized();
    Eigen::Vector3d v  = u.cross(nxhat);

    // Find first collision time index
    int i_tcol  = -1;
    std::vector<Eigen::Vector3d> cx(n_t);
    std::vector<double>          Rx(n_t, 0.);

    for (int i_t = 0; i_t < n_t; ++i_t) {
        double R0 = setup.R[i_t], R1 = setup.R[i_t];
        double x  = (d*d - R1*R1 + R0*R0) / (2.*d);
        cx[i_t]   = c0 + x * nxhat;

        if (R0 + R1 >= d && i_tcol < 0) {
            i_tcol = i_t;
        }
        if (R0 + R1 >= d) {
            double tmp = d*d - R1*R1 + R0*R0;
            Rx[i_t] = 1./(2.*d) * std::sqrt(std::max(0., 4.*d*d*R0*R0 - tmp*tmp));
        }
    }

    if (i_tcol < 0) return false;

    for (int i_t = i_tcol; i_t < n_t; ++i_t) {
        std::vector<std::pair<double,double>> sections;
        double R2 = setup.R[i_t];

        for (int b = 0; b < setup.n_b; ++b) {
            if (b == real_i || b == real_j)
                continue;
            // Include the same 3x3x3 periodic images used by the reference
            // GetGhosts implementation.  Start around the image nearest to
            // the collision centre so this remains correct when cx itself is
            // in an unwrapped neighbouring cell.
            Eigen::Vector3d c_near = setup.pos[b];
            for (int k = 0; k < 3; ++k)
                c_near[k] += setup.L * std::round(
                    (cx[i_t][k] - c_near[k]) / setup.L);

            for (int sx = -1; sx <= 1; ++sx) {
                for (int sy = -1; sy <= 1; ++sy) {
                    for (int sz = -1; sz <= 1; ++sz) {
                        Eigen::Vector3d c2 = c_near + setup.L *
                            Eigen::Vector3d(sx, sy, sz);
                        Eigen::Vector3d dist = c2 - cx[i_t];
                        if (dist.squaredNorm() > R2*R2*4) continue;

                        Eigen::Vector3d cs; double Rs;
                        other_circle(c2, R2, nxhat, cx[i_t], cs, Rs);
                        double t0, t1;
                        get_cut_range(cs, Rs, cx[i_t], Rx[i_t], u, v,
                                      t0, t1);
                        if (!std::isnan(t0) && !std::isnan(t1) && t0 != t1)
                            sections.push_back({t0, t1});
                    }
                }
            }
        }

        if (sections.empty()) {
            weight[i_t] = 1.;
        } else {
            double arc = remaining_arc(sections, 2.*M_PI);
            weight[i_t] = arc / (2.*M_PI);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// All pairs, including collisions through the periodic boundary
// ---------------------------------------------------------------------------

void FindAllCollisionWeights(const WeightsSetup &setup,
                              std::vector<double> &flat_weights,
                              std::vector<std::pair<int,int>> &collision_pairs,
                              std::vector<double> &pair_d) {
    const int n_b = setup.n_b;
    const double L = setup.L;

    // The 27 periodic offsets (including zero, the direct/native position).
    // Every candidate pair below shifts only the "j" side by one of these --
    // a bijection onto the 27 physically distinct relative configurations of
    // a pair, so (unlike shifting both sides) no configuration is ever
    // reachable two different ways and nothing can be double-counted.
    std::vector<Eigen::Vector3d> offsets;
    offsets.reserve(27);
    for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
            for (int sz = -1; sz <= 1; ++sz)
                offsets.push_back(L * Eigen::Vector3d(sx, sy, sz));

    std::vector<std::pair<int,int>> real_pairs;
    for (int i = 0; i < n_b; ++i)
        for (int j = i; j < n_b; ++j)
            real_pairs.push_back({i, j});
    const int n_p = static_cast<int>(real_pairs.size());

    struct Entry { int i, j; double d; bool is_direct; std::vector<double> w; };
    std::vector<Entry> entries;

#pragma omp parallel
    {
        std::vector<Entry> local;

#pragma omp for schedule(dynamic) nowait
        for (int p = 0; p < n_p; ++p) {
            int i = real_pairs[p].first, j = real_pairs[p].second;
            const Eigen::Vector3d &c0 = setup.pos[i];

            for (const Eigen::Vector3d &off : offsets) {
                if (i == j && off.isZero(0)) continue;   // no self-collision
                Eigen::Vector3d c1 = setup.pos[j] + off;

                std::vector<double> w;
                if (!ComputePairWeight(c0, c1, i, j, setup, w))
                    continue;
                bool any = false;
                for (double wi : w) if (wi > 0.) { any = true; break; }
                if (any) local.push_back({i, j, (c1 - c0).norm(), off.isZero(0),
                                          std::move(w)});
            }
        }

#pragma omp critical
        {
            for (auto &e : local)
                entries.push_back(std::move(e));
        }
    }

    // OMP scheduling makes the merge order above nondeterministic -- fix a
    // canonical order so callers that only look at the first entry per (i,j)
    // (e.g. an isolated two-bubble comparison) reliably get the direct,
    // non-periodic collision, with any periodic images following sorted by
    // distance.
    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry &a, const Entry &b) {
                         if (a.i != b.i) return a.i < b.i;
                         if (a.j != b.j) return a.j < b.j;
                         if (a.is_direct != b.is_direct) return a.is_direct;
                         return a.d < b.d;
                     });

    // Flatten: flat_weights[pair * n_t + i_t]
    flat_weights.clear();
    collision_pairs.clear();
    pair_d.clear();
    for (const auto &e : entries) {
        collision_pairs.push_back({e.i, e.j});
        pair_d.push_back(e.d);
        flat_weights.insert(flat_weights.end(), e.w.begin(), e.w.end());
    }
}
