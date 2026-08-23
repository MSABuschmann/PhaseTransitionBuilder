#include "io.h"

#include <iomanip>
#include <sstream>

void SaveStepResult(const std::string &output_dir, int i_t, double t,
                    const std::vector<double> &wlist,
                    const std::vector<double> &spectrum) {
    std::ostringstream oss;
    oss << output_dir << "result_"
        << std::setfill('0') << std::setw(4) << i_t << ".h5";

    H5::H5File file(oss.str(), H5F_ACC_TRUNC);
    write_attr_double(file, "t", t);
    write_vector(file, "w",        wlist);
    write_vector(file, "spectrum", spectrum);
}

void SaveAmplitudeResult(const std::string &output_dir, int i_t, double t,
                          const AmplitudeResult &res) {
    std::ostringstream oss;
    oss << output_dir << "result_"
        << std::setfill('0') << std::setw(4) << i_t << ".h5";

    H5::H5File file(oss.str(), H5F_ACC_TRUNC);
    write_attr_double(file, "t", t);
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
