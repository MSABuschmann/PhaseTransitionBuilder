#include <H5Cpp.h>
#include <algorithm>
#include <hdf5.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "interaction.h"

// Read the weights input HDF5.
//
// Expected format (written by ptbuilder.analysis or a Python helper):
//   attrs:    L (double), n_t (int), n_b (int)
//   datasets: t [n_t], R [n_t], xlocs [n_b], ylocs [n_b], zlocs [n_b]
//
// Output HDF5:
//   attrs:    n_pairs (int), n_t (int)
//   dataset:  weights  [n_pairs * n_t] (flattened, row = pair)
//   dataset:  pair_i   [n_pairs]
//   dataset:  pair_j   [n_pairs]

static WeightsSetup read_setup(const std::string &path) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    WeightsSetup s;
    s.L      = read_attr_double(file, "L");
    s.rout_0 = read_attr_double(file, "rout_0");
    s.rin_0  = read_attr_double(file, "rin_0");
    if (!file.attrExists("collision_r0") ||
        !file.attrExists("collision_radius"))
        throw std::runtime_error(
            "Weights input lacks collision-radius metadata; regenerate or "
            "relabel it explicitly before use");
    s.collision_r0 = read_attr_double(file, "collision_r0");
    s.collision_radius = read_attr_string(file, "collision_radius");
    if (s.collision_radius != "mid" && s.collision_radius != "out" &&
        s.collision_radius != "in")
        throw std::runtime_error(
            "collision_radius must be 'mid', 'out', or 'in', got '" +
            s.collision_radius + "'");
    s.n_t    = read_attr_int   (file, "n_t");
    s.n_b    = read_attr_int   (file, "n_b");
    s.t      = read_vector(file, "t");
    s.R      = read_vector(file, "R");
    auto xs = read_vector(file, "xlocs");
    auto ys = read_vector(file, "ylocs");
    auto zs = read_vector(file, "zlocs");
    s.pos.resize(s.n_b);
    for (int i = 0; i < s.n_b; ++i)
        s.pos[i] = Eigen::Vector3d(xs[i], ys[i], zs[i]);
    return s;
}

// Replace setup.R with the numerically measured wall radius from a
// bubblemaster --wall-radius-scan run (wall_radius_scan.h5: s, R_mid), linearly
// interpolated onto setup.t. Along the collision axis s = t. No fit or
// extrapolation: every requested t must lie inside the measured range.
static void apply_wall_radius(WeightsSetup &setup, const std::string &path) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    const std::vector<double> s     = read_vector(file, "s");
    const std::vector<double> R_mid = read_vector(file, "R_mid");
    const std::vector<double> R_in  = read_vector(file, "R_in");
    const std::vector<double> R_out = read_vector(file, "R_out");
    const std::vector<double> valid = read_vector(file, "valid");
    for (double v : valid)
        if (v != 1.)
            throw std::runtime_error(path + " contains invalid snapshots");
    if (setup.t.front() < s.front() || setup.t.back() > s.back())
        throw std::runtime_error(
            "weights times [" + std::to_string(setup.t.front()) + ", " +
            std::to_string(setup.t.back()) + "] exceed the wall-radius scan range [" +
            std::to_string(s.front()) + ", " + std::to_string(s.back()) + "] of " + path);
    setup.wall_width.resize(setup.n_t);
    for (int i = 0; i < setup.n_t; ++i) {
        const double t = setup.t[i];
        const std::size_t i1 = std::upper_bound(s.begin(), s.end(), t) - s.begin();
        if (i1 == s.size()) {
            setup.R[i] = R_mid.back();
            setup.wall_width[i] = R_out.back() - R_in.back();
            continue;
        }
        const std::size_t i0 = i1 - 1;
        const double frac = (t - s[i0]) / (s[i1] - s[i0]);
        auto lerp = [&](const std::vector<double> &v) { return v[i0] + frac * (v[i1] - v[i0]); };
        setup.R[i] = lerp(R_mid);
        setup.wall_width[i] = lerp(R_out) - lerp(R_in);
    }
}

int main(int argc, char *argv[]) {
    // --wall-radius <file> may appear anywhere; the rest stay positional.
    std::string wall_radius_path;
    double ring_amp = 0., ring_beta = 0.;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--wall-radius" && i + 1 < argc) wall_radius_path = argv[++i];
        else if (a == "--ring-amp" && i + 1 < argc) ring_amp = std::stod(argv[++i]);
        else if (a == "--ring-beta" && i + 1 < argc) ring_beta = std::stod(argv[++i]);
        else args.push_back(a);
    }
    if (args.size() < 2 || args.size() > 5) {
        std::cerr << "Usage: weights <input.h5> <output.h5> "
                     "[occlusion_delay] [persistence_alpha] [persistence_m_true] "
                     "[--wall-radius <wall_radius_scan.h5>]\n"
                     "  --wall-radius  use the numerically measured R_mid(t) from a\n"
                     "                 bubblemaster --wall-radius-scan run instead of\n"
                     "                 the input file's analytic R(t); with\n"
                     "                 persistence_alpha > 0, a covered ring patch then\n"
                     "                 decays as exp(-dt/tau), tau = persistence_alpha *\n"
                     "                 (measured wall width R_out - R_in when covered)\n"
                     "  --ring-amp A --ring-beta B  add a ringing term: a covered patch\n"
                     "                 decays as (1-A) exp(-dt/tau) + A exp(-dt/tau_ring),\n"
                     "                 tau_ring = B * gamma_local / persistence_m_true\n";
        return 1;
    }
    const std::string in_path  = args[0];
    const std::string out_path = args[1];
    const double occlusion_delay    = (args.size() >= 3) ? std::stod(args[2]) : 0.;
    const double persistence_alpha  = (args.size() >= 4) ? std::stod(args[3]) : 0.;
    const double persistence_m_true = (args.size() >= 5) ? std::stod(args[4]) : 0.;

    std::cout << "Input:  " << in_path  << "\n";
    std::cout << "Output: " << out_path << "\n";
    if (occlusion_delay > 0.)
        std::cout << "Occlusion delay: " << occlusion_delay << "\n";
    if (persistence_alpha > 0.)
        std::cout << "Persistence alpha: " << persistence_alpha
                   << " (expensive per-bin path)\n";
    if (persistence_m_true > 0.)
        std::cout << "Persistence m_true: " << persistence_m_true
                   << " (time-dilated relaxation model)\n";

    WeightsSetup setup = read_setup(in_path);
    setup.occlusion_delay    = occlusion_delay;
    setup.persistence_alpha  = persistence_alpha;
    setup.persistence_m_true = persistence_m_true;
    setup.ring_amp  = ring_amp;
    setup.ring_beta = ring_beta;
    if (ring_amp > 0. && (persistence_alpha <= 0. || persistence_m_true <= 0. || ring_beta <= 0.)) {
        std::cerr << "--ring-amp needs persistence_alpha > 0, persistence_m_true > 0 and --ring-beta > 0\n";
        return 1;
    }
    if (!wall_radius_path.empty()) {
        apply_wall_radius(setup, wall_radius_path);
        std::cout << "Wall radius: numerical R_mid(t) from " << wall_radius_path << "\n";
    }
    std::cout << "Bubbles: " << setup.n_b
              << "  Time steps: " << setup.n_t
              << "  Collision radius: " << setup.collision_radius << "\n";

    std::vector<double>            flat_weights;
    std::vector<std::pair<int,int>> collision_pairs;
    std::vector<double>            pair_dist;
    FindAllCollisionWeights(setup, flat_weights, collision_pairs, pair_dist);

    int n_pairs = static_cast<int>(collision_pairs.size());
    std::cout << "Collisions found: " << n_pairs << "\n";

    // Compute gamma per pair analytically.
    // Collision: the selected surfaces touch, d = 2*R_collision(t_m).
    // Gamma = (rout_0 - rin_0) / (R_out(t_m) - R_in(t_m)), where
    // t_m^2 = d^2/4 - collision_r0^2.  Only for the "out" convention is
    // R_out(t_m)=d/2; retaining both evolved radii also handles "mid".
    // d is the specific periodic image's separation found by
    // FindAllCollisionWeights -- the same real bubble pair can appear more
    // than once here (once directly, once through the periodic boundary),
    // each with its own d and gamma.
    const double rout_0 = setup.rout_0;
    const double rin_0  = setup.rin_0;
    const double collision_r0 = setup.collision_r0;
    const double w0     = rout_0 - rin_0;

    std::vector<double> pair_i(n_pairs), pair_j(n_pairs), gammas(n_pairs);
    for (int p = 0; p < n_pairs; ++p) {
        pair_i[p] = collision_pairs[p].first;
        pair_j[p] = collision_pairs[p].second;

        double d      = pair_dist[p];
        double tm_sq  = std::max(0., d*d/4. - collision_r0*collision_r0);
        double R_out_m = std::sqrt(rout_0*rout_0 + tm_sq);
        double R_in_m = std::sqrt(rin_0*rin_0 + tm_sq);
        gammas[p]     = w0 / (R_out_m - R_in_m);
    }

    // Write output
    H5::H5File out(out_path, H5F_ACC_TRUNC);

    write_attr_int(out, "n_pairs", n_pairs);
    write_attr_int(out, "n_t",     setup.n_t);
    write_attr_double(out, "collision_r0", setup.collision_r0);
    write_attr_string(out, "collision_radius", setup.collision_radius);
    write_attr_string(out, "wall_radius_file", wall_radius_path);   // "" = analytic R(t)

    write_vector(out, "weights", flat_weights);
    write_vector(out, "pair_i",  pair_i);
    write_vector(out, "pair_j",  pair_j);
    write_vector(out, "gamma",   gammas);

    std::cout << "Done.\n";
    return 0;
}
