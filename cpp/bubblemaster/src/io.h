#pragma once

#include <H5Cpp.h>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "amplitude.h"
#include "wall_energy.h"
#include "wall_radius.h"
#include "patch_moments.h"

// Write one time-step result: datasets "w" [n_w] and "spectrum" [n_w], plus
// the "t" and "gamma_ij" attributes (the physical cutoff time and the row's
// own Lorentz wall-thinning factor this result was computed at) -- with
// both embedded, a single result_NNNN.h5 file is fully self-contained and
// interpretable without its setup.h5 or any manifest.
void SaveStepResult(const std::string &output_dir, int i_t, double t, double gamma_ij,
                    const std::vector<double> &wlist,
                    const std::vector<double> &spectrum);

// Write amplitude result: "w", "k", "spectrum", "amp_re"[n_w,n_k], "amp_im"[n_w,n_k],
// plus the "t"/"gamma_ij" attributes (see SaveStepResult).
void SaveAmplitudeResult(const std::string &output_dir, int i_t, double t, double gamma_ij,
                          const AmplitudeResult &res);

// Write the full field snapshot array (optional --save-fields flag)
void SaveFields(const std::string &output_dir,
                const std::vector<std::vector<double>> &phicomplete,
                const std::vector<double> &slist,
                const std::vector<double> &z);

// Write the wall-energy diagnostic (--wall-energy flag): s, e_colliding,
// e_undisturbed, and each window's [z_lo,z_hi] boundary trajectory (for
// overlaying on a z-vs-s field plot from --save-fields' own fields.h5).
void SaveWallEnergy(const std::string &output_dir, const WallEnergyResult &res,
                    double gamma_ij);

// Write the numerical wall-radius scan (--wall-radius-scan flag): s, and the
// gradient-energy-based R_mid/R_in/R_out (measured from the undisturbed
// side's actual (dphi/dz)^2 peak and half-max points, not the static
// bounce's tanh(phi0)-fraction contours) plus a validity flag per snapshot.
// Write the patch-fraction calibration moments (--patch-moments flag).
void SavePatchMoments(const std::string &output_dir, const PatchMomentsResult &res,
                      double gamma_ij, double s_c, double delta_c, double z_max);

void SaveWallRadiusScan(const std::string &output_dir, const WallRadiusResult &res,
                        double gamma_ij);
