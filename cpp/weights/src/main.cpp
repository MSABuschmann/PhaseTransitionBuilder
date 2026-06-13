#include <H5Cpp.h>
#include <hdf5.h>
#include <iostream>
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
    s.L   = read_attr_double(file, "L");
    s.n_t = read_attr_int   (file, "n_t");
    s.n_b = read_attr_int   (file, "n_b");
    s.t   = read_vector(file, "t");
    s.R   = read_vector(file, "R");
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
              << "  Time steps: " << setup.n_t << "\n";

    std::vector<double>            flat_weights;
    std::vector<std::pair<int,int>> collision_pairs;
    FindAllCollisionWeights(setup, flat_weights, collision_pairs);

    int n_pairs = static_cast<int>(collision_pairs.size());
    std::cout << "Collisions found: " << n_pairs << "\n";

    // Write output
    H5::H5File out(out_path, H5F_ACC_TRUNC);

    write_attr_int(out, "n_pairs", n_pairs);
    write_attr_int(out, "n_t",     setup.n_t);

    write_vector(out, "weights", flat_weights);

    std::vector<double> pair_i(n_pairs), pair_j(n_pairs);
    for (int p = 0; p < n_pairs; ++p) {
        pair_i[p] = collision_pairs[p].first;
        pair_j[p] = collision_pairs[p].second;
    }
    write_vector(out, "pair_i", pair_i);
    write_vector(out, "pair_j", pair_j);

    std::cout << "Done.\n";
    return 0;
}
