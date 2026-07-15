#include <chrono>
#include <iostream>
#include <string>

#include "evolution.h"
#include "io.h"
#include "setup.h"

#ifdef USE_GPU
#  include "gpu_integrator.cuh"
   using Integrator = GpuIntegrator;
#elif defined(USE_FILON)
#  include "filon_integrator.h"
   using Integrator = FilonIntegrator;
#else
#  include "integrator.h"
#endif

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static double elapsed(Clock::time_point t0) {
    return Sec(Clock::now() - t0).count();
}

int main(int argc, char *argv[]) {
    auto t_start = Clock::now();
    if (argc < 3) {
        std::cerr << "Usage: bubblemaster <setup.h5> <output_dir/>"
                     " [--save-fields] [--param N]\n"
                     "  --param N  Filon: N_min panels; GSL: subinterval limit\n";
        return 1;
    }

    const std::string setup_path = argv[1];
    const std::string output_dir = argv[2];
    bool save_fields = false;
    int  qual_param  = -1;   // -1 → use compiled default
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--save-fields")
            save_fields = true;
        else if (a == "--param" && i + 1 < argc)
            qual_param = std::atoi(argv[++i]);
    }

    std::cout << "Setup:  " << setup_path << "\n";
    std::cout << "Output: " << output_dir << "\n";
    if (save_fields)
        std::cout << "Saving fields\n";

    // --- 1. Load setup ---
    Setup setup(setup_path);

    // --- 2. Run 2D Milne evolution ---
    auto t_evo = Clock::now();
    Evolution evo(setup);
    const auto &phi_snaps = evo.GetPhi();
    std::cout << "Evolution: " << elapsed(t_evo) << " s\n";

    if (save_fields)
        SaveFields(output_dir, phi_snaps, evo.GetSlist(), setup.z);

    // --- 3. Run GW integration for each time index ---
    Integrator integrator(phi_snaps, setup, qual_param);
    const auto &wlist = integrator.GetW();

    const int n_t = setup.n_t;
    auto t_integ = Clock::now();
    for (int i_t = 0; i_t < n_t; ++i_t) {
        auto t_it = Clock::now();
        std::vector<double> spectrum = integrator.Compute(i_t);
        SaveStepResult(output_dir, i_t, wlist, spectrum);
        std::cout << "Time index " << i_t << " / " << n_t - 1
                  << ": " << elapsed(t_it) << " s\n";
    }
    std::cout << "Integration total: " << elapsed(t_integ) << " s\n";

    std::cout << "Total: " << elapsed(t_start) << " s\n";
    return 0;
}
