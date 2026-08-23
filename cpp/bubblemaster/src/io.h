#pragma once

#include <H5Cpp.h>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "amplitude.h"

// Write one time-step result: datasets "w" [n_w] and "spectrum" [n_w], plus
// the "t" attribute (the physical cutoff time this result was computed at) --
// without it, a result_NNNN.h5 file is only interpretable together with the
// exact setup.h5 that produced it (its "times" array at index i_t).
void SaveStepResult(const std::string &output_dir, int i_t, double t,
                    const std::vector<double> &wlist,
                    const std::vector<double> &spectrum);

// Write amplitude result: "w", "k", "spectrum", "amp_re"[n_w,n_k], "amp_im"[n_w,n_k],
// plus the "t" attribute (see SaveStepResult).
void SaveAmplitudeResult(const std::string &output_dir, int i_t, double t,
                          const AmplitudeResult &res);

// Write the full field snapshot array (optional --save-fields flag)
void SaveFields(const std::string &output_dir,
                const std::vector<std::vector<double>> &phicomplete,
                const std::vector<double> &slist,
                const std::vector<double> &z);
