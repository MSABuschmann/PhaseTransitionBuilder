#include "evolution.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Evolution::Evolution(const Setup &setup)
    : n_z(setup.n_z), how_often_ds(setup.how_often_ds),
      baby_steps(setup.baby_steps), ds(setup.ds), smax(setup.smax),
      dz(std::abs(setup.z[1] - setup.z[0])), d(setup.d),
      potential(setup.potential.get()), z(setup.z), phi(setup.phi0)
{
    n_steps = static_cast<int>(std::round(smax / ds));
    if (n_steps < 1) n_steps = 1;

    pi.resize(n_z, 0.);
    phicomplete.push_back(phi);
    slist.push_back(0.);
    ds_out = ds * how_often_ds;

    Evolve();
}

// ---------------------------------------------------------------------------
// Milne-coordinate pi update  (Evolvepi equivalent from reference)
//
// Discretized Milne equation in leapfrog:
//   π_{n+1} = π_n * (1 - 2·baby_ds/(s+baby_ds))
//           + s·baby_ds/(s+baby_ds) * (d²φ/dz² - dV/dφ)
//
// At s≈0 this formula degenerates gracefully; the baby-step first half-step
// approaches this limit smoothly with many small substeps.
// ---------------------------------------------------------------------------

double Evolution::EvolvePi(int i_z, double s, double step) const {
    double phiprev = (i_z != 0)      ? phi[i_z - 1] : phi[1];
    double phinext = (i_z != n_z-1)  ? phi[i_z + 1] : phi[n_z - 2];

    double lap = (phiprev - 2.*phi[i_z] + phinext) / (dz * dz);
    double dv  = potential->dV(phi[i_z]);

    double fac_pi  = 1. - 2.*step / (s + step);
    double fac_src =  s * step / (s + step);

    return pi[i_z] * fac_pi + fac_src * (lap - dv);
}

void Evolution::EvolvepiFirstHalfStep(int n_baby) {
    // Sub-step ds/2 in n_baby-1 increments from s=0 to s≈ds/2
    double baby_ds = (0.5 * ds) / static_cast<double>(n_baby - 1);

    for (int i = 1; i < n_baby; ++i) {
        double s = (i - 1) * baby_ds;
#pragma omp parallel for
        for (int i_z = 0; i_z < n_z; ++i_z)
            pi[i_z] = EvolvePi(i_z, s, baby_ds);

#pragma omp parallel for
        for (int i_z = 0; i_z < n_z; ++i_z)
            phi[i_z] += baby_ds * pi[i_z];
    }
}

void Evolution::Evolve() {
    EvolvepiFirstHalfStep(baby_steps);

    int n_print = std::max(n_steps / 20, 1);

    for (int i = 1; i <= n_steps; ++i) {
        if (i % n_print == 0)
            std::cout << "Evolution: step " << i << " / " << n_steps << "\n";

        // Full pi step (skip first — pi was already advanced by baby steps)
        if (i > 1) {
            double s = (i - 1) * ds;
#pragma omp parallel for
            for (int i_z = 0; i_z < n_z; ++i_z)
                pi[i_z] = EvolvePi(i_z, s, ds);
        }

        // phi step
#pragma omp parallel for
        for (int i_z = 0; i_z < n_z; ++i_z)
            phi[i_z] += ds * pi[i_z];

        // Save snapshot every how_often_ds steps
        if (i % how_often_ds == 0) {
            phicomplete.push_back(phi);
            slist.push_back(i * ds);
        }
    }
}
