#pragma once

#include <H5Cpp.h>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "amplitude.h"

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
