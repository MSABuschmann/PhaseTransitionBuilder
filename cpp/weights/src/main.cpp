#include <H5Cpp.h>
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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: weights <input.h5> <output.h5>\n";
        return 1;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];

    std::cout << "Input:  " << in_path  << "\n";
    std::cout << "Output: " << out_path << "\n";

    WeightsSetup setup = read_setup(in_path);
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

    write_vector(out, "weights", flat_weights);
    write_vector(out, "pair_i",  pair_i);
    write_vector(out, "pair_j",  pair_j);
    write_vector(out, "gamma",   gammas);

    std::cout << "Done.\n";
    return 0;
}
