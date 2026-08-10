#include "amplitude_table.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "../../common/hdf5_utils.h"

namespace {

bool vectors_close(const std::vector<double> &a, const std::vector<double> &b,
                    double rtol = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
        if (std::abs(a[i] - b[i]) > rtol * scale) return false;
    }
    return true;
}

} // namespace

AmplitudeTable read_amplitude_table(const std::string &scan_cache_path) {
    H5::H5File file(scan_cache_path, H5F_ACC_RDONLY);
    auto group_names = list_group_names(file);
    if (group_names.empty())
        throw std::runtime_error("read_amplitude_table: " + scan_cache_path + " has no groups");

    // Read gamma for each group, then process in ascending-gamma order.
    std::vector<std::pair<double, std::string>> gamma_names;
    gamma_names.reserve(group_names.size());
    for (const auto &name : group_names) {
        H5::Group grp = file.openGroup(name);
        gamma_names.emplace_back(read_attr_double(grp, "gamma"), name);
    }
    std::sort(gamma_names.begin(), gamma_names.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    AmplitudeTable amp;
    amp.Ng = static_cast<int>(gamma_names.size());

    for (int ig = 0; ig < amp.Ng; ++ig) {
        H5::Group grp = file.openGroup(gamma_names[ig].second);

        bool has_amp;
        try {
            grp.openDataSet("amp_re");
            has_amp = true;
        } catch (const H5::Exception &) {
            has_amp = false;
        }
        if (!has_amp)
            throw std::runtime_error(
                "read_amplitude_table: group '" + gamma_names[ig].second +
                "' (gamma=" + std::to_string(gamma_names[ig].first) +
                ") has no amplitude data -- was scan_cache.h5 built with "
                "--save-amplitude?");

        auto times = read_vector(grp, "times");
        auto w     = read_vector(grp, "w");
        auto k     = read_vector(grp, "k");

        std::vector<hsize_t> re_dims, im_dims;
        auto amp_re = read_ndarray(grp, "amp_re", re_dims);
        auto amp_im = read_ndarray(grp, "amp_im", im_dims);
        if (re_dims.size() != 3 || re_dims != im_dims)
            throw std::runtime_error("read_amplitude_table: unexpected amp_re/amp_im shape");

        int Nt = static_cast<int>(re_dims[0]);
        int n_w = static_cast<int>(re_dims[1]);
        int n_k = static_cast<int>(re_dims[2]);

        if (ig == 0) {
            amp.Nt = Nt; amp.n_w = n_w; amp.n_k = n_k;
            amp.t_grid = times; amp.w = w; amp.k = k;
            amp.amp_re.assign(static_cast<size_t>(amp.Ng) * Nt * n_w * n_k, 0.0);
            amp.amp_im.assign(amp.amp_re.size(), 0.0);
        } else {
            if (Nt != amp.Nt || n_w != amp.n_w || n_k != amp.n_k)
                throw std::runtime_error(
                    "read_amplitude_table: grid shape mismatch across gamma groups");
            if (!vectors_close(times, amp.t_grid) || !vectors_close(w, amp.w) ||
                !vectors_close(k, amp.k))
                throw std::runtime_error(
                    "read_amplitude_table: times/w/k differ across gamma groups");
        }

        amp.gamma_grid.push_back(gamma_names[ig].first);
        size_t base = amp.idx(ig, 0, 0, 0);
        std::copy(amp_re.begin(), amp_re.end(), amp.amp_re.begin() + base);
        std::copy(amp_im.begin(), amp_im.end(), amp.amp_im.begin() + base);
    }

    return amp;
}
