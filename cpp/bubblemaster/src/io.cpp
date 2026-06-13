#include "io.h"

#include <iomanip>
#include <sstream>

void SaveStepResult(const std::string &output_dir, int i_t,
                    const std::vector<double> &wlist,
                    const std::vector<double> &spectrum) {
    std::ostringstream oss;
    oss << output_dir << "result_"
        << std::setfill('0') << std::setw(4) << i_t << ".h5";

    H5::H5File file(oss.str(), H5F_ACC_TRUNC);
    write_vector(file, "w",        wlist);
    write_vector(file, "spectrum", spectrum);
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
