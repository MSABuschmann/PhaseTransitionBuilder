#include "io.h"

#include <iomanip>
#include <sstream>

void SaveStepResult(const std::string &output_dir, int i_t, double t, double gamma_ij,
                    const std::vector<double> &wlist,
                    const std::vector<double> &spectrum) {
    std::ostringstream oss;
    oss << output_dir << "result_"
        << std::setfill('0') << std::setw(4) << i_t << ".h5";

    H5::H5File file(oss.str(), H5F_ACC_TRUNC);
    write_attr_double(file, "t", t);
    write_attr_double(file, "gamma_ij", gamma_ij);
    write_vector(file, "w",        wlist);
    write_vector(file, "spectrum", spectrum);
}

void SaveAmplitudeResult(const std::string &output_dir, int i_t, double t, double gamma_ij,
                          const AmplitudeResult &res) {
    std::ostringstream oss;
    oss << output_dir << "result_"
        << std::setfill('0') << std::setw(4) << i_t << ".h5";

    H5::H5File file(oss.str(), H5F_ACC_TRUNC);
    write_attr_double(file, "t", t);
    write_attr_double(file, "gamma_ij", gamma_ij);
    write_vector(file, "w",        res.w);
    write_vector(file, "k",        res.klist);
    write_vector(file, "spectrum", res.spectrum);
    hsize_t nw = static_cast<hsize_t>(res.w.size());
    hsize_t nk = static_cast<hsize_t>(res.klist.size());
    write_2d_flat(file, "amp_re", res.amp_re, nw, nk);
    write_2d_flat(file, "amp_im", res.amp_im, nw, nk);
}

void SaveFields(const std::string &output_dir,
                const std::vector<std::vector<double>> &phicomplete,
                const std::vector<double> &slist,
                const std::vector<double> &z) {
    std::string path = output_dir + "fields.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_2d(file, "phi", phicomplete);
    write_vector(file, "s", slist);
    write_vector(file, "z", z);
}

void SaveWallEnergy(const std::string &output_dir, const WallEnergyResult &res,
                    double gamma_ij) {
    std::string path = output_dir + "wall_energy.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_attr_double(file, "gamma_ij", gamma_ij);
    write_vector(file, "s",                res.s);
    write_vector(file, "e_colliding",      res.e_colliding);
    write_vector(file, "e_undisturbed",    res.e_undisturbed);
    write_vector(file, "valid_colliding",  res.valid_colliding);
    write_vector(file, "valid_undisturbed", res.valid_undisturbed);
    write_vector(file, "z_lo_colliding",   res.z_lo_colliding);
    write_vector(file, "z_hi_colliding",   res.z_hi_colliding);
    write_vector(file, "z_lo_undisturbed", res.z_lo_undisturbed);
    write_vector(file, "z_hi_undisturbed", res.z_hi_undisturbed);
    write_attr_double(file, "s_collision", res.s_collision);
}

void SavePatchMoments(const std::string &output_dir, const PatchMomentsResult &res,
                      double gamma_ij, double s_c, double delta_c, double z_max) {
    std::string path = output_dir + "patch_moments.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_attr_double(file, "gamma_ij", gamma_ij);
    write_attr_double(file, "s_c", s_c);
    write_attr_double(file, "delta_c", delta_c);
    write_attr_double(file, "z_max", z_max);
    write_vector(file, "s_A",    res.s_A);
    write_vector(file, "A",      res.A);
    write_vector(file, "zcut_A", res.zcut_A);
    write_vector(file, "R_in_A", res.R_in_A);
    write_vector(file, "s_B",    res.s_B);
    write_vector(file, "B",      res.B);
    write_vector(file, "zcut_B", res.zcut_B);
}

void SaveWallRadiusScan(const std::string &output_dir, const WallRadiusResult &res,
                        double gamma_ij) {
    std::string path = output_dir + "wall_radius_scan.h5";
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_attr_double(file, "gamma_ij", gamma_ij);
    write_vector(file, "s",     res.s);
    write_vector(file, "R_mid", res.R_mid);
    write_vector(file, "R_in",  res.R_in);
    write_vector(file, "R_out", res.R_out);
    write_vector(file, "valid", res.valid);
}
