#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../common/potential_io.h"

// 1D spherical leapfrog solver for a single expanding bubble.
// Used for validation against the Python-based field evolution in ic.py.
class Solver1D {
public:
    explicit Solver1D(const std::string &setup_path);

    void Run();
    void Save(const std::string &output_path) const;

    const std::vector<double> &GetR()    const { return r_; }
    const std::vector<double> &GetPhi()  const { return phi_; }
    const std::vector<double> &GetPi()   const { return pi_; }
    double                     GetTime() const { return t_; }

private:
    void EvolveHalfStep();
    double Laplacian(int i) const;

    int    n_r_;
    double dr_, dt_, t_end_, t_;

    std::unique_ptr<Potential> potential_;

    std::vector<double> r_;
    std::vector<double> phi_;
    std::vector<double> pi_;
};
