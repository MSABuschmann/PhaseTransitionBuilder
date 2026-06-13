#include "solver.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "../../common/hdf5_utils.h"

Solver1D::Solver1D(const std::string &setup_path) {
    H5::H5File file(setup_path, H5F_ACC_RDONLY);

    t_end_ = read_attr_double(file, "t_end");
    dr_    = read_attr_double(file, "dr");
    n_r_   = read_attr_int   (file, "n_r");
    dt_    = dr_ * 0.5;
    t_     = 0.;

    potential_ = Potential::from_hdf5(file.openGroup("potential"));

    // Read instanton profile and interpolate onto our r grid
    // (written by Python physics.py _save_instanton: R and Phi at top level)
    auto R_in   = read_vector(file, "R");
    auto Phi_in = read_vector(file, "Phi");

    r_.resize(n_r_);
    phi_.resize(n_r_, 0.);
    pi_.resize(n_r_, 0.);

    for (int i = 0; i < n_r_; ++i) {
        r_[i] = i * dr_;
        // Linear interpolation from instanton profile
        phi_[i] = std::max(0., [&]() -> double {
            double ri = r_[i];
            if (ri <= R_in.front()) return Phi_in.front();
            if (ri >= R_in.back())  return 0.;
            auto it  = std::lower_bound(R_in.begin(), R_in.end(), ri);
            int  idx = static_cast<int>(it - R_in.begin()) - 1;
            if (idx < 0) idx = 0;
            double x0 = R_in[idx], x1 = R_in[idx + 1];
            double y0 = Phi_in[idx], y1 = Phi_in[idx + 1];
            return y0 + (ri - x0) * (y1 - y0) / (x1 - x0);
        }());
    }
}

double Solver1D::Laplacian(int i) const {
    if (i == 0) {
        // Neumann BC at r=0: d²φ/dr² + (2/r)(dφ/dr) → 3 d²φ/dr²|_{r=0}
        return 6. * (phi_[1] - phi_[0]) / (dr_ * dr_);
    }
    if (i == n_r_ - 1) {
        return (phi_[n_r_ - 2] - phi_[n_r_ - 1]) / (dr_ * dr_)
             - phi_[n_r_ - 1] / (r_[n_r_ - 1] * dr_);
    }
    // Standard 3-point spherical Laplacian
    return (phi_[i + 1] - 2. * phi_[i] + phi_[i - 1]) / (dr_ * dr_)
         + (phi_[i + 1] - phi_[i - 1]) / (2. * r_[i] * dr_);
}

void Solver1D::EvolveHalfStep() {
    for (int i = 0; i < n_r_; ++i)
        pi_[i] += 0.5 * dt_ * (Laplacian(i) - potential_->dV(phi_[i]));
}

void Solver1D::Run() {
    int n_steps = static_cast<int>(std::round(t_end_ / dt_));
    dt_         = t_end_ / n_steps;

    EvolveHalfStep(); // initial half-step for π

    int n_print = std::max(n_steps / 20, 1);
    for (int step = 0; step < n_steps; ++step) {
        if (step % n_print == 0)
            std::cout << "Step " << step << " / " << n_steps << "\n";

        for (int i = 0; i < n_r_; ++i)
            phi_[i] += dt_ * pi_[i];
        t_ += dt_;

        for (int i = 0; i < n_r_; ++i)
            pi_[i] += dt_ * (Laplacian(i) - potential_->dV(phi_[i]));
    }
    // Complete the last π half-step
    for (int i = 0; i < n_r_; ++i)
        pi_[i] -= 0.5 * dt_ * (Laplacian(i) - potential_->dV(phi_[i]));
}

void Solver1D::Save(const std::string &path) const {
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_attr_double(file, "t", t_);
    write_vector(file, "r",   r_);
    write_vector(file, "phi", phi_);
    write_vector(file, "pi",  pi_);
}
