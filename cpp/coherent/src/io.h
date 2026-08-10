#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

// Bubble positions and box size, read from weights_in.h5 (same file
// cpp/weights reads -- see cpp/weights/src/main.cpp's read_setup).
struct BubbleSetup {
    double L;
    int    n_b;
    std::vector<Eigen::Vector3d> pos;   // [n_b]
};

// Collision weights, pairs, and gammas, read from weights_in.h5 (for the
// fine time grid) + weights_out.h5 (for pairs/weights/gamma -- see
// cpp/weights/src/main.cpp's output format).
struct WeightsData {
    int n_pairs, n_t;
    std::vector<double> t;              // [n_t]        fine time grid ("weights_times")
    std::vector<double> weights;        // [n_pairs*n_t] flattened, row = pair
    std::vector<int>    pair_i, pair_j; // [n_pairs]
    std::vector<double> gamma;          // [n_pairs]
};

BubbleSetup read_bubble_setup(const std::string &weights_in_path);

WeightsData read_weights_data(const std::string &weights_in_path,
                               const std::string &weights_out_path);

// Writes the output HDF5:
//   attrs:    n_w (int), n_t_max (int)
//   datasets: w [n_w], t_max [n_t_max], n_active [n_t_max],
//             P_coh [n_t_max * n_w] flattened row-major, row = t_max index
void write_output(const std::string &out_path,
                   const std::vector<double> &w_grid,
                   const std::vector<double> &t_max_values,
                   const std::vector<int>    &n_active_per_tmax,
                   const std::vector<std::vector<double>> &P_coh_per_tmax);
