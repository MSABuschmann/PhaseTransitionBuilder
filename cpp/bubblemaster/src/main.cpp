#include <iostream>
#include <string>

#include "evolution.h"
#include "io.h"
#include "setup.h"

#ifdef USE_FILON
#  include "filon_integrator.h"
   using Integrator = FilonIntegrator;
#else
#  include "integrator.h"
#endif

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: bubblemaster <setup.h5> <output_dir/> [--save-fields]\n";
        return 1;
    }

    const std::string setup_path = argv[1];
    const std::string output_dir = argv[2];
    const bool save_fields = (argc == 4 &&
                               std::string(argv[3]) == "--save-fields");

    std::cout << "Setup:  " << setup_path << "\n";
    std::cout << "Output: " << output_dir << "\n";
    if (save_fields)
        std::cout << "Saving fields\n";

    // --- 1. Load setup ---
    Setup setup(setup_path);

    // --- 2. Run 2D Milne evolution ---
    Evolution evo(setup);
    const auto &phi_snaps = evo.GetPhi();

    if (save_fields)
        SaveFields(output_dir, phi_snaps, evo.GetSlist(), setup.z);

    // --- 3. Run GW integration for each time index ---
    Integrator integrator(phi_snaps, setup);
    const auto &wlist = integrator.GetW();

    const int n_t = setup.n_t;
    for (int i_t = 0; i_t < n_t; ++i_t) {
        std::cout << "\n=== Time index " << i_t << " / " << n_t - 1 << " ===\n";
        std::vector<double> spectrum = integrator.Compute(i_t);
        SaveStepResult(output_dir, i_t, wlist, spectrum);
    }

    std::cout << "\nDone.\n";
    return 0;
}
