#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "amplitude_table.h"
#include "coherent_sum.h"
#include "geometry.h"
#include "io.h"
#include "quadrature.h"

namespace {

void print_usage(const char *prog) {
    std::cerr
        << "Usage: " << prog
        << " <weights_in.h5> <weights_out.h5> <scan_cache.h5> <output.h5>\n"
        << "       --t-max VALUE [--t-max VALUE ...]\n"
        << "       [--threshold X]     (default 0.01)\n"
        << "       [--n-theta N]       (default 64)\n"
        << "       [--n-phi N]         (default 128)\n"
        << "       [--t-chunk N]       (default 32)\n";
}

} // namespace

int main(int argc, char *argv[]) {
    std::cout.setf(std::ios::unitbuf);

    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string weights_in_path  = argv[1];
    const std::string weights_out_path = argv[2];
    const std::string scan_cache_path  = argv[3];
    const std::string output_path      = argv[4];

    std::vector<double> t_max_values;
    double threshold = 0.01;
    int n_theta = 64, n_phi = 128, t_chunk = 32;

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_double = [&](const char *flag) -> double {
            if (i + 1 >= argc) {
                std::cerr << flag << " requires a value\n";
                std::exit(1);
            }
            return std::stod(argv[++i]);
        };
        if (arg == "--t-max") {
            t_max_values.push_back(next_double("--t-max"));
        } else if (arg == "--threshold") {
            threshold = next_double("--threshold");
        } else if (arg == "--n-theta") {
            n_theta = static_cast<int>(next_double("--n-theta"));
        } else if (arg == "--n-phi") {
            n_phi = static_cast<int>(next_double("--n-phi"));
        } else if (arg == "--t-chunk") {
            t_chunk = static_cast<int>(next_double("--t-chunk"));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (t_max_values.empty()) {
        std::cerr << "At least one --t-max is required\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "weights_in:  " << weights_in_path  << "\n";
    std::cout << "weights_out: " << weights_out_path << "\n";
    std::cout << "scan_cache:  " << scan_cache_path  << "\n";
    std::cout << "output:      " << output_path      << "\n";

    BubbleSetup bubbles = read_bubble_setup(weights_in_path);
    WeightsData wd      = read_weights_data(weights_in_path, weights_out_path);
    std::cout << "Bubbles: " << bubbles.n_b << "  Pairs: " << wd.n_pairs
              << "  Fine time steps: " << wd.n_t << "\n";

    AmplitudeTable amp = read_amplitude_table(scan_cache_path);
    std::cout << "Amplitude table: " << amp.Ng << " gammas x " << amp.Nt
              << " times  n_w=" << amp.n_w << " n_k=" << amp.n_k << "\n";

    PairGeometry geo = compute_pair_geometry(bubbles, wd, bubbles.L);

    SphereQuadrature quad = build_sphere_quadrature(n_theta, n_phi);
    std::cout << "Sphere quadrature: " << quad.n_sph << " points ("
              << n_theta << " x " << n_phi << ")\n";

    std::vector<int> n_active_per_tmax;
    std::vector<std::vector<double>> P_coh_per_tmax;

    for (double t_max : t_max_values) {
        auto active = active_pairs_for(wd, t_max, threshold);
        std::cout << "t_max=" << t_max << ": " << active.size() << "/" << wd.n_pairs
                  << " pairs active\n";

        auto t_start = std::chrono::steady_clock::now();
        auto P_coh = compute_coherent_spectrum(t_max, wd, amp, geo, quad, t_chunk, threshold);
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::cout << "  done in " << elapsed << " s\n";

        n_active_per_tmax.push_back(static_cast<int>(active.size()));
        P_coh_per_tmax.push_back(std::move(P_coh));
    }

    write_output(output_path, amp.w, t_max_values, n_active_per_tmax, P_coh_per_tmax);
    std::cout << "Wrote " << output_path << "\n";

    return 0;
}
