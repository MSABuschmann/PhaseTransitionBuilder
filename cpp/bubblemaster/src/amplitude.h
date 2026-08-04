#pragma once

#include <vector>

// Holds both the direction-integrated power spectrum and the pre-integration
// complex amplitude A(w, cos_theta) from a single bubble-collision run.
// amp_re and amp_im are stored row-major: index as [i_w * n_k + i_k].
// Squaring and direction-integrating amp gives spectrum exactly:
//   spectrum[i_w] = sum_k dk * fk * (amp_re^2 + amp_im^2) * w^3 * 2*pi
struct AmplitudeResult {
    std::vector<double> w;        // [n_w]  frequency grid
    std::vector<double> klist;    // [n_k]  cos(theta) grid, linspace(0,1,n_k)
    std::vector<double> spectrum; // [n_w]  direction-integrated power spectrum
    std::vector<double> amp_re;   // [n_w * n_k]  Re A(w, cos_theta)
    std::vector<double> amp_im;   // [n_w * n_k]  Im A(w, cos_theta)
};
