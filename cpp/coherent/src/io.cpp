#include "io.h"

#include <H5Cpp.h>
#include <stdexcept>

#include "../../common/hdf5_utils.h"

BubbleSetup read_bubble_setup(const std::string &weights_in_path) {
    H5::H5File file(weights_in_path, H5F_ACC_RDONLY);
    BubbleSetup s;
    s.L   = read_attr_double(file, "L");
    s.n_b = read_attr_int(file, "n_b");

    auto xs = read_vector(file, "xlocs");
    auto ys = read_vector(file, "ylocs");
    auto zs = read_vector(file, "zlocs");
    s.pos.resize(s.n_b);
    for (int i = 0; i < s.n_b; ++i)
        s.pos[i] = Eigen::Vector3d(xs[i], ys[i], zs[i]);
    return s;
}

WeightsData read_weights_data(const std::string &weights_in_path,
                               const std::string &weights_out_path) {
    WeightsData wd;

    H5::H5File in_file(weights_in_path, H5F_ACC_RDONLY);
    wd.t = read_vector(in_file, "t");

    H5::H5File out_file(weights_out_path, H5F_ACC_RDONLY);
    wd.n_pairs = read_attr_int(out_file, "n_pairs");
    wd.n_t     = read_attr_int(out_file, "n_t");
    if (wd.n_t != static_cast<int>(wd.t.size()))
        throw std::runtime_error(
            "read_weights_data: n_t mismatch between weights_in.h5 (" +
            std::to_string(wd.t.size()) + ") and weights_out.h5 (" +
            std::to_string(wd.n_t) + ")");

    wd.weights = read_vector(out_file, "weights");
    auto pi    = read_vector(out_file, "pair_i");
    auto pj    = read_vector(out_file, "pair_j");
    wd.gamma   = read_vector(out_file, "gamma");

    wd.pair_i.resize(wd.n_pairs);
    wd.pair_j.resize(wd.n_pairs);
    for (int p = 0; p < wd.n_pairs; ++p) {
        wd.pair_i[p] = static_cast<int>(pi[p]);
        wd.pair_j[p] = static_cast<int>(pj[p]);
    }
    return wd;
}

void write_output(const std::string &out_path,
                   const std::vector<double> &w_grid,
                   const std::vector<double> &t_max_values,
                   const std::vector<int>    &n_active_per_tmax,
                   const std::vector<std::vector<double>> &P_coh_per_tmax) {
    H5::H5File out(out_path, H5F_ACC_TRUNC);

    int n_w     = static_cast<int>(w_grid.size());
    int n_t_max = static_cast<int>(t_max_values.size());

    write_attr_int(out, "n_w",     n_w);
    write_attr_int(out, "n_t_max", n_t_max);

    write_vector(out, "w",     w_grid);
    write_vector(out, "t_max", t_max_values);

    std::vector<double> n_active_d(n_active_per_tmax.begin(), n_active_per_tmax.end());
    write_vector(out, "n_active", n_active_d);

    std::vector<double> flat;
    flat.reserve(static_cast<size_t>(n_t_max) * n_w);
    for (const auto &row : P_coh_per_tmax) {
        if (static_cast<int>(row.size()) != n_w)
            throw std::runtime_error("write_output: P_coh row size mismatch");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    write_2d_flat(out, "P_coh", flat, static_cast<hsize_t>(n_t_max), static_cast<hsize_t>(n_w));
}
