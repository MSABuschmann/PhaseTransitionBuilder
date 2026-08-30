#pragma once

#include <H5Cpp.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../common/hdf5_utils.h"
#include "../../common/potential_io.h"

class Setup {
public:
    explicit Setup(const std::string &path) {
        H5::H5File file(path, H5F_ACC_RDONLY);

        // --- scalar attributes (replaces the old flat header[16]) ---
        n_z          = read_attr_int(file, "n_z");
        n_w          = read_attr_int(file, "n_w");
        n_k          = read_attr_int(file, "n_k");
        n_t          = read_attr_int(file, "n_t");
        how_often_ds = read_attr_int(file, "how_often_ds");
        baby_steps   = read_attr_int(file, "baby_steps");
        cutoff_type  = read_attr_int(file, "cutoff_type");

        d            = read_attr_double(file, "d");
        ds           = read_attr_double(file, "ds");
        t_0          = read_attr_double(file, "t_0");
        t_cut        = read_attr_double(file, "t_cut");
        t_m          = read_attr_double(file, "t_m");
        t_max        = read_attr_double(file, "t_max");
        smax         = read_attr_double(file, "smax");
        gamma_ij     = read_attr_double(file, "gamma_ij");

        // --- potential ---
        potential = potential_from_hdf5(file.openGroup("potential"));

        // --- datasets ---
        z     = read_vector(file, "z");
        phi0  = read_vector(file, "phi0");
        wlist = read_vector(file, "wlist");
        times = read_vector(file, "times");
    }

    // integers
    int n_z, n_w, n_k, n_t, how_often_ds, baby_steps, cutoff_type;

    // doubles
    double d, ds, t_0, t_cut, t_m, t_max, smax, gamma_ij;

    // potential (owned)
    std::unique_ptr<Potential> potential;

    // arrays
    std::vector<double> z, phi0, wlist, times;
};
