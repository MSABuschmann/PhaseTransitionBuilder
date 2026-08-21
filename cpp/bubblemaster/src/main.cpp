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
    // Auto-flush std::cout after every '<<' — without this, output redirected
    // to a SLURM log file (not a TTY) is fully buffered, so progress prints
    // (per-time-step timing, etc.) don't actually reach the log until the
    // buffer fills or the process exits, making `tail -f` on the log useless
    // for monitoring an in-progress run.
    std::cout.setf(std::ios::unitbuf);

    auto t_start = Clock::now();
    if (argc < 3) {
        std::cerr << "Usage: bubblemaster <setup.h5> <output_dir/>"
                     " [--save-fields] [--save-amplitude] [--param N]"
                     " [--filon-panels-per-osc N]\n"
                     "  --param N         Filon: N_min panels; GSL: subinterval limit\n"
                     "  --filon-panels-per-osc N  GPU Filon panels per Bessel oscillation\n"
                     "  --save-amplitude  Also write Re/Im A(w,cos_theta) to result files\n";
        return 1;
    }

    const std::string setup_path = argv[1];
    const std::string output_dir = argv[2];
    bool save_fields    = false;
    bool save_amplitude = false;
    int  qual_param     = -1;   // -1 → use compiled default
    int  filon_panels_per_osc = 64;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--save-fields")
            save_fields = true;
        else if (a == "--save-amplitude")
            save_amplitude = true;
        else if (a == "--param" && i + 1 < argc)
            qual_param = std::atoi(argv[++i]);
        else if (a == "--filon-panels-per-osc" && i + 1 < argc)
            filon_panels_per_osc = std::atoi(argv[++i]);
    }

    std::cout << "Setup:  " << setup_path << "\n";
    std::cout << "Output: " << output_dir << "\n";
    if (save_fields)
        std::cout << "Saving fields\n";
    if (save_amplitude)
        std::cout << "Saving complex amplitude A(w, cos_theta)\n";

    // --- 1. Load setup ---
    auto t_setup_load = Clock::now();
    Setup setup(setup_path);
    std::cout << "Timing phase=setup_load seconds="
              << elapsed(t_setup_load) << "\n";

    // --- 2. Run 2D Milne evolution ---
    auto t_evo = Clock::now();
    Evolution evo(setup);
    const auto &phi_snaps = evo.GetPhi();
    const double evolution_seconds = elapsed(t_evo);
    std::cout << "Evolution: " << evolution_seconds << " s\n";
    std::cout << "Timing phase=field_evolution seconds="
              << evolution_seconds << "\n";

    if (save_fields)
        SaveFields(output_dir, phi_snaps, evo.GetSlist(), setup.z);

    // --- 3. Run GW integration for each time index ---
    auto t_integrator_setup = Clock::now();
#ifdef USE_GPU
    if (filon_panels_per_osc < 2) {
        std::cerr << "--filon-panels-per-osc must be at least 2\n";
        return 1;
    }
    Integrator integrator(phi_snaps, setup, qual_param,
                          filon_panels_per_osc);
#else
    Integrator integrator(phi_snaps, setup, qual_param);
#endif
    std::cout << "Timing phase=integrator_setup_total seconds="
              << elapsed(t_integrator_setup) << "\n";
    const auto &wlist = integrator.GetW();

    const int n_t = setup.n_t;
    auto t_integ = Clock::now();
    double compute_total = 0.;
    double output_total = 0.;
    for (int i_t = 0; i_t < n_t; ++i_t) {
        auto t_it = Clock::now();
        auto t_compute = Clock::now();
        double compute_seconds = 0.;
        double output_seconds = 0.;
        if (save_amplitude) {
            AmplitudeResult res = integrator.ComputeAmplitude(i_t);
            compute_seconds = elapsed(t_compute);
            auto t_output = Clock::now();
            SaveAmplitudeResult(output_dir, i_t, res);
            output_seconds = elapsed(t_output);
        } else {
            std::vector<double> spectrum = integrator.Compute(i_t);
            compute_seconds = elapsed(t_compute);
            auto t_output = Clock::now();
            SaveStepResult(output_dir, i_t, wlist, spectrum);
            output_seconds = elapsed(t_output);
        }
        compute_total += compute_seconds;
        output_total += output_seconds;
        std::cout << "Time index " << i_t << " / " << n_t - 1
                  << ": " << elapsed(t_it) << " s\n";
        std::cout << "Timing phase=cutoff index=" << i_t
                  << " compute_seconds=" << compute_seconds
                  << " output_seconds=" << output_seconds
                  << " total_seconds=" << elapsed(t_it) << "\n";
    }
    std::cout << "Integration total: " << elapsed(t_integ) << " s\n";
    std::cout << "Timing phase=cutoff_all count=" << n_t
              << " compute_seconds=" << compute_total
              << " output_seconds=" << output_total
              << " total_seconds=" << elapsed(t_integ) << "\n";

    std::cout << "Total: " << elapsed(t_start) << " s\n";
    return 0;
}
