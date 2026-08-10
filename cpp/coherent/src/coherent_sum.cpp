#include "coherent_sum.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

namespace {

// Find lo such that x falls between vec[lo] and vec[lo+1] (vec sorted
// ascending, size >= 2), clamping x to vec's range first. Returns the blend
// fraction frac in [0,1] such that x ~= (1-frac)*vec[lo] + frac*vec[lo+1].
void find_bracket(const std::vector<double> &vec, double x, int &lo, double &frac) {
    int n = static_cast<int>(vec.size());
    x = std::clamp(x, vec.front(), vec.back());
    auto it = std::upper_bound(vec.begin(), vec.end(), x);
    int hi = std::clamp(static_cast<int>(it - vec.begin()), 1, n - 1);
    lo = hi - 1;
    double span = vec[hi] - vec[lo];
    frac = (span > 0.0) ? (x - vec[lo]) / span : 0.0;
}

// Linear interpolation matching numpy.interp's default clamping (values
// outside [xp.front(), xp.back()] are clamped to the nearest endpoint, not
// extrapolated).
double lerp1d(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back())  return fp.back();
    int lo; double frac;
    find_bracket(xp, x, lo, frac);
    return fp[lo] + frac * (fp[lo + 1] - fp[lo]);
}

// Bilinear (gamma,time) interpolation of amp_re/amp_im at one (g_lo,g_frac,
// t_lo,t_frac) grid location, for all (w,k) -- writes into out_re/out_im,
// each sized n_w*n_k, k fastest-varying.
void bilinear_amp_slice(const AmplitudeTable &amp, int g_lo, double g_frac,
                         int t_lo, double t_frac,
                         std::vector<double> &out_re, std::vector<double> &out_im) {
    int n_w = amp.n_w, n_k = amp.n_k;
    double w00 = (1 - g_frac) * (1 - t_frac);
    double w10 = g_frac       * (1 - t_frac);
    double w01 = (1 - g_frac) * t_frac;
    double w11 = g_frac       * t_frac;

    for (int iw = 0; iw < n_w; ++iw) {
        for (int ik = 0; ik < n_k; ++ik) {
            size_t i00 = amp.idx(g_lo,     t_lo,     iw, ik);
            size_t i10 = amp.idx(g_lo + 1, t_lo,     iw, ik);
            size_t i01 = amp.idx(g_lo,     t_lo + 1, iw, ik);
            size_t i11 = amp.idx(g_lo + 1, t_lo + 1, iw, ik);
            int out_idx = iw * n_k + ik;
            out_re[out_idx] = w00 * amp.amp_re[i00] + w10 * amp.amp_re[i10] +
                               w01 * amp.amp_re[i01] + w11 * amp.amp_re[i11];
            out_im[out_idx] = w00 * amp.amp_im[i00] + w10 * amp.amp_im[i10] +
                               w01 * amp.amp_im[i01] + w11 * amp.amp_im[i11];
        }
    }
}

// Computes dA_re/dA_im for one active pair, one time chunk: for each of the
// chunk's n_diff intervals, diff the bilinearly-interpolated amplitude
// between consecutive fine time points and scale by sqrt(weight) evaluated
// at the interval's start -- matches notebook 06's dA_pair_series_batch /
// coherent_sum_n64 chunk step exactly. Writes into the shared
// [n_active][n_diff][n_w][n_k] buffers at slot `p`.
void build_amp_diff_for_pair(const AmplitudeTable &amp, const WeightsData &wd,
                              int pair_idx, double gamma_p,
                              const std::vector<double> &t_fine, int c0, int c1,
                              std::vector<double> &dA_re, std::vector<double> &dA_im,
                              int p, int n_active, int n_diff, int n_w, int n_k) {
    (void)n_active;
    (void)c1;
    int g_lo; double g_frac;
    find_bracket(amp.gamma_grid, gamma_p, g_lo, g_frac);

    std::vector<double> slice_re(n_w * n_k), slice_im(n_w * n_k);
    std::vector<double> prev_re(n_w * n_k), prev_im(n_w * n_k);

    int t_lo; double t_frac;
    find_bracket(amp.t_grid, t_fine[c0], t_lo, t_frac);
    bilinear_amp_slice(amp, g_lo, g_frac, t_lo, t_frac, prev_re, prev_im);

    const double *pair_weights = &wd.weights[static_cast<size_t>(pair_idx) * wd.n_t];
    std::vector<double> weight_row(pair_weights, pair_weights + wd.n_t);

    for (int it = 0; it < n_diff; ++it) {
        find_bracket(amp.t_grid, t_fine[c0 + it + 1], t_lo, t_frac);
        bilinear_amp_slice(amp, g_lo, g_frac, t_lo, t_frac, slice_re, slice_im);

        double wgt = std::max(0.0, lerp1d(t_fine[c0 + it], wd.t, weight_row));
        double sqrt_wgt = std::sqrt(wgt);

        size_t out_base = (static_cast<size_t>(p) * n_diff + it) * n_w * n_k;
        for (int j = 0; j < n_w * n_k; ++j) {
            dA_re[out_base + j] = (slice_re[j] - prev_re[j]) * sqrt_wgt;
            dA_im[out_base + j] = (slice_im[j] - prev_im[j]) * sqrt_wgt;
        }
        std::swap(slice_re, prev_re);
        std::swap(slice_im, prev_im);
    }
}

} // namespace

std::vector<int> active_pairs_for(const WeightsData &wd, double t_max, double threshold) {
    std::vector<int> active;
    for (int p = 0; p < wd.n_pairs; ++p) {
        const double *row = &wd.weights[static_cast<size_t>(p) * wd.n_t];
        bool any_active = false;
        for (int it = 0; it < wd.n_t && wd.t[it] <= t_max; ++it) {
            if (row[it] > threshold) { any_active = true; break; }
        }
        if (any_active) active.push_back(p);
    }
    return active;
}

void debug_dump_pair(const AmplitudeTable &amp, const WeightsData &wd,
                      int pair_idx, double t_max) {
    double gamma_p = std::clamp(wd.gamma[pair_idx], amp.gamma_grid.front(), amp.gamma_grid.back());

    double t_hi = std::min(t_max, amp.t_grid.back());
    std::vector<double> t_fine;
    for (double x : wd.t)
        if (x >= amp.t_grid.front() && x <= t_hi) t_fine.push_back(x);
    if (t_fine.empty() || t_fine.back() < t_hi) t_fine.push_back(t_hi);

    std::cout.precision(17);
    std::cout << "[debug] pair_idx=" << pair_idx << "  raw_gamma=" << wd.gamma[pair_idx]
              << "  clipped_gamma=" << gamma_p << "\n";
    std::cout << "[debug] t_fine: n=" << t_fine.size() << "  first=" << t_fine.front()
              << "  second=" << t_fine[1] << "  last=" << t_fine.back() << "\n";

    int g_lo; double g_frac;
    find_bracket(amp.gamma_grid, gamma_p, g_lo, g_frac);
    std::cout << "[debug] gamma bracket: g_lo=" << g_lo << " (" << amp.gamma_grid[g_lo]
              << ")  g_hi=" << g_lo + 1 << " (" << amp.gamma_grid[g_lo + 1]
              << ")  g_frac=" << g_frac << "\n";

    std::vector<double> re0(amp.n_w * amp.n_k), im0(amp.n_w * amp.n_k);
    std::vector<double> re1(amp.n_w * amp.n_k), im1(amp.n_w * amp.n_k);
    int t_lo; double t_frac;
    find_bracket(amp.t_grid, t_fine.front(), t_lo, t_frac);
    std::cout << "[debug] t bracket @t_fine[0]: t_lo=" << t_lo << " (" << amp.t_grid[t_lo]
              << ")  t_hi=" << t_lo + 1 << " (" << amp.t_grid[t_lo + 1]
              << ")  t_frac=" << t_frac << "\n";
    bilinear_amp_slice(amp, g_lo, g_frac, t_lo, t_frac, re0, im0);

    find_bracket(amp.t_grid, t_fine.back(), t_lo, t_frac);
    std::cout << "[debug] t bracket @t_fine[-1]: t_lo=" << t_lo << " (" << amp.t_grid[t_lo]
              << ")  t_hi=" << t_lo + 1 << " (" << amp.t_grid[t_lo + 1]
              << ")  t_frac=" << t_frac << "\n";
    bilinear_amp_slice(amp, g_lo, g_frac, t_lo, t_frac, re1, im1);

    std::cout << "[debug] amp_re[iw=0, k=0..4] @t_fine[0]: ";
    for (int ik = 0; ik < std::min(5, amp.n_k); ++ik) std::cout << re0[ik] << " ";
    std::cout << "\n[debug] amp_im[iw=0, k=0..4] @t_fine[0]: ";
    for (int ik = 0; ik < std::min(5, amp.n_k); ++ik) std::cout << im0[ik] << " ";
    std::cout << "\n[debug] amp_re[iw=0, k=0..4] @t_fine[-1]: ";
    for (int ik = 0; ik < std::min(5, amp.n_k); ++ik) std::cout << re1[ik] << " ";
    std::cout << "\n[debug] amp_im[iw=0, k=0..4] @t_fine[-1]: ";
    for (int ik = 0; ik < std::min(5, amp.n_k); ++ik) std::cout << im1[ik] << " ";
    std::cout << "\n";

    const double *pair_weights = &wd.weights[static_cast<size_t>(pair_idx) * wd.n_t];
    std::vector<double> weight_row(pair_weights, pair_weights + wd.n_t);
    double wgt0 = lerp1d(t_fine.front(), wd.t, weight_row);
    std::cout << "[debug] weight @t_fine[0]=" << wgt0 << "\n";
}

std::vector<double> compute_coherent_spectrum(
        double t_max, const WeightsData &wd, const AmplitudeTable &amp,
        const PairGeometry &geo, const SphereQuadrature &quad,
        int t_chunk, double threshold) {

    std::vector<int> active = active_pairs_for(wd, t_max, threshold);
    int n_active = static_cast<int>(active.size());
    int n_w = amp.n_w, n_k = amp.n_k, n_sph = quad.n_sph;
    if (n_active == 0) return std::vector<double>(n_w, 0.0);

    std::vector<double> g_arr(n_active);
    for (int p = 0; p < n_active; ++p)
        g_arr[p] = std::clamp(wd.gamma[active[p]], amp.gamma_grid.front(), amp.gamma_grid.back());

    double t_hi = std::min(t_max, amp.t_grid.back());
    std::vector<double> t_fine;
    for (double x : wd.t)
        if (x >= amp.t_grid.front() && x <= t_hi) t_fine.push_back(x);
    if (t_fine.empty() || t_fine.back() < t_hi) t_fine.push_back(t_hi);
    int n_t = static_cast<int>(t_fine.size());

    std::vector<double> P_coh(n_w, 0.0);
    if (n_t < 2) return P_coh;
    double *P_coh_ptr = P_coh.data();

    int n_threads = omp_get_max_threads();
    std::vector<std::vector<double>> phase_re_pool(n_threads, std::vector<double>(static_cast<size_t>(n_active) * n_w));
    std::vector<std::vector<double>> phase_im_pool(n_threads, std::vector<double>(static_cast<size_t>(n_active) * n_w));
    std::vector<std::vector<int>>    kidx_pool(n_threads, std::vector<int>(n_active));
    std::vector<std::vector<double>> kfrac_pool(n_threads, std::vector<double>(n_active));

    for (int c0 = 0; c0 < n_t - 1; c0 += t_chunk) {
        int c1     = std::min(c0 + t_chunk, n_t - 1);
        int n_diff = c1 - c0;

        std::vector<double> dA_re(static_cast<size_t>(n_active) * n_diff * n_w * n_k);
        std::vector<double> dA_im(dA_re.size());

        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n_active; ++p)
            build_amp_diff_for_pair(amp, wd, active[p], g_arr[p], t_fine, c0, c1,
                                     dA_re, dA_im, p, n_active, n_diff, n_w, n_k);

        #pragma omp parallel for schedule(static) reduction(+:P_coh_ptr[:n_w])
        for (int d = 0; d < n_sph; ++d) {
            int tid = omp_get_thread_num();
            auto &phase_re = phase_re_pool[tid];
            auto &phase_im = phase_im_pool[tid];
            auto &kidx     = kidx_pool[tid];
            auto &kfrac    = kfrac_pool[tid];

            for (int p = 0; p < n_active; ++p) {
                double cth = std::clamp(std::abs(quad.khat[d].dot(geo.axis[active[p]])), 0.0, 1.0);
                // Bracket-search rather than assuming amp.k is exactly
                // uniform (linspace(0,1,n_k)) -- matches the gamma/time
                // lookups elsewhere and removes any risk from real k grids
                // not being bit-exactly uniform. Identical result to the
                // O(1) formula for a genuinely uniform grid, just safer.
                int lo; double frac;
                find_bracket(amp.k, cth, lo, frac);
                kidx[p]  = lo;
                kfrac[p] = frac;
            }
            for (int p = 0; p < n_active; ++p) {
                double proj = quad.khat[d].dot(geo.center[active[p]]);
                for (int iw = 0; iw < n_w; ++iw) {
                    double ang = amp.w[iw] * proj;
                    phase_re[static_cast<size_t>(p) * n_w + iw] = std::cos(ang);
                    phase_im[static_cast<size_t>(p) * n_w + iw] = std::sin(ang);
                }
            }
            for (int it = 0; it < n_diff; ++it) {
                for (int iw = 0; iw < n_w; ++iw) {
                    double Are = 0.0, Aim = 0.0;
                    for (int p = 0; p < n_active; ++p) {
                        size_t base = ((static_cast<size_t>(p) * n_diff + it) * n_w + iw) * n_k;
                        int lo = kidx[p]; double f = kfrac[p];
                        double ar = dA_re[base + lo] * (1 - f) + dA_re[base + lo + 1] * f;
                        double ai = dA_im[base + lo] * (1 - f) + dA_im[base + lo + 1] * f;
                        double pr = phase_re[static_cast<size_t>(p) * n_w + iw];
                        double pi = phase_im[static_cast<size_t>(p) * n_w + iw];
                        Are += ar * pr - ai * pi;
                        Aim += ar * pi + ai * pr;
                    }
                    P_coh_ptr[iw] += quad.domega[d] * (Are * Are + Aim * Aim);
                }
            }
        }
    }

    for (int iw = 0; iw < n_w; ++iw)
        P_coh[iw] *= amp.w[iw] * amp.w[iw] * amp.w[iw] * 2.0 * M_PI;
    return P_coh;
}
