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

// Linear interpolation matching numpy.interp(..., left=0., right=0.):
// values outside [xp.front(), xp.back()] are zero, not extrapolated or
// clamped. xp must be sorted ascending.
double lerp_zero_outside(double x, const std::vector<double> &xp, const double *fp, size_t stride) {
    if (x <= xp.front() || x >= xp.back()) {
        if (x == xp.front()) return fp[0];
        if (x == xp.back())  return fp[(xp.size() - 1) * stride];
        return 0.0;
    }
    auto it = std::upper_bound(xp.begin(), xp.end(), x);
    size_t hi = std::clamp(static_cast<size_t>(it - xp.begin()), size_t(1), xp.size() - 1);
    size_t lo = hi - 1;
    double frac = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo * stride] + frac * (fp[hi * stride] - fp[lo * stride]);
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
        auto w_g   = read_vector(grp, "w");   // this group's OWN w grid -- see below
        auto k     = read_vector(grp, "k");

        std::vector<hsize_t> re_dims, im_dims;
        auto amp_re_g = read_ndarray(grp, "amp_re", re_dims);
        auto amp_im_g = read_ndarray(grp, "amp_im", im_dims);
        if (re_dims.size() != 3 || re_dims != im_dims)
            throw std::runtime_error("read_amplitude_table: unexpected amp_re/amp_im shape");

        int Nt = static_cast<int>(re_dims[0]);
        int n_w = static_cast<int>(re_dims[1]);
        int n_k = static_cast<int>(re_dims[2]);

        if (ig == 0) {
            amp.Nt = Nt; amp.n_w = n_w; amp.n_k = n_k;
            amp.t_grid = times; amp.w = w_g; amp.k = k;
            amp.amp_re.assign(static_cast<size_t>(amp.Ng) * Nt * n_w * n_k, 0.0);
            amp.amp_im.assign(amp.amp_re.size(), 0.0);
        } else {
            if (Nt != amp.Nt || n_w != amp.n_w || n_k != amp.n_k)
                throw std::runtime_error(
                    "read_amplitude_table: grid shape mismatch across gamma groups");
            if (!vectors_close(times, amp.t_grid) || !vectors_close(k, amp.k))
                throw std::runtime_error(
                    "read_amplitude_table: times/k differ across gamma groups "
                    "(expected identical -- both are gamma-independent by "
                    "construction: times comes from the shared scan grid, k "
                    "is always linspace(0,1,n_k))");
            // NOTE: w is NOT expected to match across gamma groups -- BubbleMaster
            // derives wmin from this pair's own spatial grid (which depends on
            // gamma via the bubble separation d), so each gamma has its own
            // log-spaced w axis (confirmed against real scan data: wmin varies
            // ~5x across the gamma range while wmax stays fixed). Every gamma's
            // amp_re/amp_im must therefore be resampled onto ig=0's w grid
            // (amp.w) before being stacked into a common (gamma,time,w,k)
            // table -- otherwise the same array index `iw` would refer to a
            // different physical frequency depending on gamma, corrupting the
            // (gamma,time) bilinear interpolation used later. This mirrors the
            // resampling already done for the power-only spectrum in
            // notebooks/06_n64_reconstruction.ipynb's build_log_scan_interpolator.
        }

        amp.gamma_grid.push_back(gamma_names[ig].first);
        size_t base = amp.idx(ig, 0, 0, 0);

        if (ig == 0) {
            std::copy(amp_re_g.begin(), amp_re_g.end(), amp.amp_re.begin() + base);
            std::copy(amp_im_g.begin(), amp_im_g.end(), amp.amp_im.begin() + base);
        } else {
            for (int it = 0; it < Nt; ++it) {
                for (int ik = 0; ik < n_k; ++ik) {
                    const double *re_col = &amp_re_g[(static_cast<size_t>(it) * n_w) * n_k + ik];
                    const double *im_col = &amp_im_g[(static_cast<size_t>(it) * n_w) * n_k + ik];
                    for (int iw = 0; iw < amp.n_w; ++iw) {
                        double x = amp.w[iw];
                        size_t out = amp.idx(ig, it, iw, ik);
                        amp.amp_re[out] = lerp_zero_outside(x, w_g, re_col, n_k);
                        amp.amp_im[out] = lerp_zero_outside(x, w_g, im_col, n_k);
                    }
                }
            }
        }
    }

    return amp;
}
