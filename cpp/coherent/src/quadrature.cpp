#include "quadrature.h"

#include <algorithm>
#include <cmath>
#include <gsl/gsl_integration.h>

SphereQuadrature build_sphere_quadrature(int n_theta, int n_phi) {
    SphereQuadrature q;
    q.n_theta = n_theta;
    q.n_phi   = n_phi;
    q.n_sph   = n_theta * n_phi;
    q.khat.resize(q.n_sph);
    q.domega.resize(q.n_sph);

    gsl_integration_glfixed_table *tbl = gsl_integration_glfixed_table_alloc(n_theta);
    double d_phi = 2.0 * M_PI / n_phi;

    for (int it = 0; it < n_theta; ++it) {
        double ct, gw;
        gsl_integration_glfixed_point(-1.0, 1.0, it, &ct, &gw, tbl);
        double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
        for (int ip = 0; ip < n_phi; ++ip) {
            double phi = ip * d_phi;
            int idx = it * n_phi + ip;
            q.khat[idx]   = Eigen::Vector3d(st * std::cos(phi), st * std::sin(phi), ct);
            q.domega[idx] = gw * d_phi;
        }
    }
    gsl_integration_glfixed_table_free(tbl);
    return q;
}
