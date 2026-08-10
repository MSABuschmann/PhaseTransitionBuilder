#pragma once

#include <Eigen/Dense>
#include <vector>

// Sphere quadrature: Gauss-Legendre in cos(theta) x uniform in phi. Matches
// notebook 06's leggauss(N_THETA) + linspace(0,2pi,N_PHI,endpoint=False)
// construction; node/weight ordering doesn't need to match numpy's since the
// final result is a plain order-invariant sum over directions.
struct SphereQuadrature {
    int n_theta, n_phi, n_sph;
    std::vector<Eigen::Vector3d> khat;   // [n_sph]
    std::vector<double> domega;          // [n_sph]
};

SphereQuadrature build_sphere_quadrature(int n_theta, int n_phi);
