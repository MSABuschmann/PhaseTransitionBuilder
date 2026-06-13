#include <iostream>
#include <string>

#include "io.h"
#include "solver.h"

// Validation tool: evolve a single bubble radially and write output HDF5.
// Usage: ./solver_1d <setup.h5> <output.h5>
//
// setup.h5 must have:
//   attrs    : t_end (double), dr (double), n_r (int)
//   group    : "potential"  (readable by Potential::from_hdf5)
//   datasets : "R", "Phi"   (instanton profile at top level,
//               as written by PhysicsModel._save_instanton in physics.py)

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: solver_1d <setup.h5> <output.h5>\n";
        return 1;
    }
    const std::string setup_path  = argv[1];
    const std::string output_path = argv[2];

    std::cout << "Setup:  " << setup_path  << "\n";
    std::cout << "Output: " << output_path << "\n";

    Solver1D solver(setup_path);
    solver.Run();
    solver.Save(output_path);

    std::cout << "Done.\n";
    return 0;
}
