#include "gpu_integrator.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../common/potential.h"

// ---------------------------------------------------------------------------
// CUDA error helper
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_e));       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Device copies of math helpers (from integrand.h / filon.h)
// ---------------------------------------------------------------------------

__device__ __forceinline__ double d_j0(double x)
{
    double ax = fabs(x);
    if (ax < 8.0) {
        double y = x * x;
        double num = 57568490574.0 + y*(-13362590354.0 + y*(651619640.7
                   + y*(-11214424.18 + y*(77392.33017 + y*(-184.9052456)))));
        double den = 57568490411.0 + y*(1029532985.0 + y*(9494680.718
                   + y*(59272.64853 + y*(267.8532712 + y))));
        return num / den;
    }
    double z  = 8.0 / ax, y = z*z, xx = ax - 0.785398164;
    return sqrt(0.636619772 / ax)
         * (cos(xx) * (1.0 + y*(-0.1098628627e-2 + y*(0.2734510407e-4
                      + y*(-0.2073370639e-5 + y*0.2093887211e-6))))
          - z * sin(xx) * (0.1562499995e-1 + y*(0.1430488765e-3
                      + y*(-0.6911147651e-5 + y*0.7621095161e-6))));
}

__device__ __forceinline__ double d_j1(double x)
{
    double ax = fabs(x), ans;
    if (ax < 8.0) {
        double y = x * x;
        double num = x*(72362614232.0 + y*(-7895059235.0 + y*(242396853.1
                   + y*(-2972611.439 + y*(15704.48260 + y*(-30.16036606))))));
        double den = 144725228442.0 + y*(2300535178.0 + y*(18583304.74
                   + y*(99447.43394 + y*(376.9991397 + y))));
        ans = num / den;
    } else {
        double z  = 8.0 / ax, y = z*z, xx = ax - 2.356194491;
        ans = sqrt(0.636619772 / ax)
            * (cos(xx) * (1.0 + y*(0.183105e-2 + y*(-0.3516396496e-4
                          + y*(0.2457520174e-5 + y*(-0.240337019e-6)))))
             - z * sin(xx) * (0.04687499995 + y*(-0.2002690873e-3
                          + y*(0.8449199096e-5 + y*(-0.88228987e-6)))));
        if (x < 0.) ans = -ans;
    }
    return ans;
}

__device__ __forceinline__ double d_C1(double t, double t_cut, double t_m,
                                        double t_0, double t_max, int type)
{
    if (t < t_cut)  return 1.;
    if (t >= t_max) return 0.;
    switch (type) {
    case 0: { double a = 2.*(t - t_m) + t_0;
              return exp(-a*a*log(2.)/(t_0*t_0)); }
    case 1: { double s = sin(M_PI*(2.*(t_m-t)+t_0)/(4.*t_0)); return s*s; }
    case 2: { double a = 2.*(t_m-t)+t_0;
              return (t+t_0-t_m)*a*a/(2.*t_0*t_0*t_0); }
    default: return 0.;
    }
}

__device__ __forceinline__ void d_filon_coeffs(double theta,
                                                double &alpha,
                                                double &beta,
                                                double &gamma_f)
{
    if (fabs(theta) < 1e-6) {
        double th2 = theta*theta;
        alpha  = 2.*th2*theta/45.;
        beta   = 2./3. + 2.*th2/15.;
        gamma_f = 4./3. - 2.*th2/15.;
    } else {
        double s = sin(theta), c = cos(theta);
        double th2 = theta*theta, th3 = th2*theta;
        alpha  = (th2 + theta*s*c - 2.*s*s) / th3;
        beta   = 2.*(theta*(1.+c*c) - 2.*s*c) / th3;
        gamma_f = 4.*(s - theta*c) / th3;
    }
}

// ---------------------------------------------------------------------------
// Streaming Filon accumulator
// Avoids storing the N+1 g values; accumulates directly into 6 doubles.
// ---------------------------------------------------------------------------

struct FStream {
    double Se_c, Se_s;   // beta sums (even-index, endpoint weight = 0.5)
    double So_c, So_s;   // gamma sums (odd-index)
    double alpha_c;      //  gN*sin(wuN) - g0*sin(wu0)   -> I_cos alpha term
    double alpha_s;      //  g0*cos(wu0) - gN*cos(wuN)   -> I_sin alpha term
};

__device__ __forceinline__ void fs_update(FStream &fs, int i, int N,
                                           double g, double cw, double sw)
{
    if (i % 2 == 0) {
        double wt = (i == 0 || i == N) ? 0.5 : 1.0;
        fs.Se_c += wt * g * cw;
        fs.Se_s += wt * g * sw;
        if (i == 0)  { fs.alpha_c -= g * sw; fs.alpha_s += g * cw; }
        if (i == N)  { fs.alpha_c += g * sw; fs.alpha_s -= g * cw; }
    } else {
        fs.So_c += g * cw;
        fs.So_s += g * sw;
    }
}

__device__ __forceinline__ void fs_result(const FStream &fs,
                                           double alpha, double beta,
                                           double gamma_f, double h,
                                           double &I_cos, double &I_sin)
{
    I_cos = h * (alpha*fs.alpha_c + beta*fs.Se_c + gamma_f*fs.So_c);
    I_sin = h * (alpha*fs.alpha_s + beta*fs.Se_s + gamma_f*fs.So_s);
}

// ---------------------------------------------------------------------------
// N for one sub-interval: configurable panels per Bessel oscillation,
// floor=n_floor
// (default 8192, matching CPU Filon's FILON_N_MIN -- see filon.h).
// Uses delta-ib = ib(b) - ib(a) so each segment is resolved independently.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int filon_N(double a, double b, double sign,
                                        double w, double Sqrt1mkk, double s,
                                        int n_floor, int panels_per_oscillation)
{
    const double u2s_a = a * a + sign;
    const double ib_a  = (u2s_a > 0.) ? w * Sqrt1mkk * s * sqrt(u2s_a) : 0.;
    const double u2s_b = b * b + sign;
    const double ib_b  = (u2s_b > 0.) ? w * Sqrt1mkk * s * sqrt(u2s_b) : 0.;
    int N = max(n_floor, (int)(panels_per_oscillation *
                (ib_b - ib_a) / (2.0 * M_PI)) + 2);
    return (N + 1) & ~1;
}

// ---------------------------------------------------------------------------
// Streaming Filon u-integral over one sub-interval [a, b] with N panels.
// Templated to compute only needed stress-tensor components.
// ---------------------------------------------------------------------------

template <bool NEED_ZZ, bool NEED_XYZ>
__device__ void filon_segment(
    double s, double Sqrt1mkk, double w, double sign,
    double a, double b, int N,
    double t_cut, double t_m, double t_0, double t_max, int cutoff_type,
    double &I_zz_c, double &I_zz_s,
    double &I_xx_c, double &I_xx_s,
    double &I_yy_c, double &I_yy_s,
    double &I_xz_c, double &I_xz_s)
{
    const double h     = (b - a) / N;
    const double omega = w * s;

    FStream fzz = {}, fxx = {}, fyy = {}, fxz = {};

    for (int i = 0; i <= N; ++i) {
        double u       = a + i * h;
        double u2s     = u * u + sign;
        double u2s_pos = fmax(0., u2s);
        double ib      = w * Sqrt1mkk * s * sqrt(u2s_pos);
        double c1      = d_C1(s * u, t_cut, t_m, t_0, t_max, cutoff_type);

        double bj0 = d_j0(ib), bj1 = d_j1(ib);
        double bj0m2, bj0p2;
        if (ib < 1e-14) {
            bj0m2 = 1.0; bj0p2 = 1.0;
        } else {
            double toi = 2.0 / ib;
            bj0m2 = 2.0 * bj0 - toi * bj1;
            bj0p2 = toi * bj1;
        }

        double wu = omega * u;
        double cw = cos(wu), sw = sin(wu);

        if (NEED_ZZ)  fs_update(fzz, i, N, bj0 * c1,                   cw, sw);
        if (NEED_XYZ) {
            fs_update(fxx, i, N, u2s * bj0m2 * c1,                     cw, sw);
            fs_update(fyy, i, N, u2s * bj0p2 * c1,                     cw, sw);
            fs_update(fxz, i, N, sign * sqrt(u2s_pos) * bj1 * c1,      cw, sw);
        }
    }

    double alpha, beta, gamma_f;
    d_filon_coeffs(omega * h, alpha, beta, gamma_f);

    if (NEED_ZZ)  fs_result(fzz, alpha, beta, gamma_f, h, I_zz_c, I_zz_s);
    if (NEED_XYZ) {
        fs_result(fxx, alpha, beta, gamma_f, h, I_xx_c, I_xx_s);
        fs_result(fyy, alpha, beta, gamma_f, h, I_yy_c, I_yy_s);
        fs_result(fxz, alpha, beta, gamma_f, h, I_xz_c, I_xz_s);
    }
}

// ---------------------------------------------------------------------------
// Incremental streaming Filon u-integral -- split at t_cut/s to eliminate the
// C^2 kink, dead zone removed (upper limit t_max/s, not 1+t_max/s), N from
// delta-ib, same as the original filon_u. The difference: the plateau piece
// [umin, u_split] doesn't depend on i_t (C1==1 there identically), so instead
// of re-running Filon over the whole growing plateau every kernel launch
// (time step), only the NEW slice since the previous launch's u_split
// (t_cut_prev/s) is integrated and added into the persisted cum_re/cum_im
// (references directly into this thread's slot of the device-resident
// d_cumbuf_ -- see GpuIntegrator::d_cumbuf_). The transition window
// [u_split, u_top] has fixed width and just slides in u as t_cut advances,
// so it's still recomputed fresh every launch.
//
// Split into a zz-only and an xyz-only variant (rather than one templated
// function) since each needs a different number/shape of cum_re/cum_im
// arguments -- mirrors the two distinct call shapes already used in
// gw_kernel (one at s for zz, one at s_off for xx/yy/xz).
//
// NOTE: cum_re/cum_im here are per-batch (see GpuIntegrator::d_cumbuf_) --
// they persist across cutoff-time calls WITHIN one ProcessBatch() call, and
// are reset to 0 (along with t_cut_prev) at the start of every batch, since
// each s belongs to exactly one batch.
// ---------------------------------------------------------------------------

__device__ void filon_zz_incremental(
    double s, double Sqrt1mkk, double w,
    double sign, double umin,
    double t_cut, double t_m, double t_0, double t_max,
    int cutoff_type, int n_floor, int panels_per_oscillation,
    double t_cut_prev,
    double &cum_re, double &cum_im,
    double &zz_r, double &zz_i)
{
    const double u_top   = t_max / s;
    const double u_split = t_cut / s;

    if (u_top <= umin) {
        zz_r = cum_re; zz_i = cum_im;
        return;
    }

    const double plateau_now  = fmax(umin, u_split);
    const double plateau_prev = fmax(umin, t_cut_prev / s);

    if (plateau_now > plateau_prev) {
        int N = filon_N(plateau_prev, plateau_now, sign, w, Sqrt1mkk, s,
                        n_floor, panels_per_oscillation);
        double d_r, d_i, dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i;
        filon_segment<true, false>(s, Sqrt1mkk, w, sign,
            plateau_prev, plateau_now, N,
            t_cut, t_m, t_0, t_max, cutoff_type,
            d_r, d_i, dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i);
        cum_re += d_r; cum_im += d_i;
    }

    double win_r = 0., win_i = 0.;
    if (u_top > plateau_now) {
        int N = filon_N(plateau_now, u_top, sign, w, Sqrt1mkk, s,
                        n_floor, panels_per_oscillation);
        double dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i;
        filon_segment<true, false>(s, Sqrt1mkk, w, sign,
            plateau_now, u_top, N,
            t_cut, t_m, t_0, t_max, cutoff_type,
            win_r, win_i, dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i);
    }

    zz_r = cum_re + win_r;
    zz_i = cum_im + win_i;
}

__device__ void filon_xyz_incremental(
    double s, double Sqrt1mkk, double w,
    double sign, double umin,
    double t_cut, double t_m, double t_0, double t_max,
    int cutoff_type, int n_floor, int panels_per_oscillation,
    double t_cut_prev,
    double &cum_xx_re, double &cum_xx_im,
    double &cum_yy_re, double &cum_yy_im,
    double &cum_xz_re, double &cum_xz_im,
    double &xx_r, double &xx_i,
    double &yy_r, double &yy_i,
    double &xz_r, double &xz_i)
{
    const double u_top   = t_max / s;
    const double u_split = t_cut / s;

    if (u_top <= umin) {
        xx_r = cum_xx_re; xx_i = cum_xx_im;
        yy_r = cum_yy_re; yy_i = cum_yy_im;
        xz_r = cum_xz_re; xz_i = cum_xz_im;
        return;
    }

    const double plateau_now  = fmax(umin, u_split);
    const double plateau_prev = fmax(umin, t_cut_prev / s);

    if (plateau_now > plateau_prev) {
        int N = filon_N(plateau_prev, plateau_now, sign, w, Sqrt1mkk, s,
                        n_floor, panels_per_oscillation);
        double dum_r, dum_i, d_xx_r, d_xx_i, d_yy_r, d_yy_i, d_xz_r, d_xz_i;
        filon_segment<false, true>(s, Sqrt1mkk, w, sign,
            plateau_prev, plateau_now, N,
            t_cut, t_m, t_0, t_max, cutoff_type,
            dum_r, dum_i, d_xx_r, d_xx_i, d_yy_r, d_yy_i, d_xz_r, d_xz_i);
        cum_xx_re += d_xx_r; cum_xx_im += d_xx_i;
        cum_yy_re += d_yy_r; cum_yy_im += d_yy_i;
        cum_xz_re += d_xz_r; cum_xz_im += d_xz_i;
    }

    double w_xx_r = 0., w_xx_i = 0., w_yy_r = 0., w_yy_i = 0., w_xz_r = 0., w_xz_i = 0.;
    if (u_top > plateau_now) {
        int N = filon_N(plateau_now, u_top, sign, w, Sqrt1mkk, s,
                        n_floor, panels_per_oscillation);
        double dum_r, dum_i;
        filon_segment<false, true>(s, Sqrt1mkk, w, sign,
            plateau_now, u_top, N,
            t_cut, t_m, t_0, t_max, cutoff_type,
            dum_r, dum_i, w_xx_r, w_xx_i, w_yy_r, w_yy_i, w_xz_r, w_xz_i);
    }

    xx_r = cum_xx_re + w_xx_r; xx_i = cum_xx_im + w_xx_i;
    yy_r = cum_yy_re + w_yy_r; yy_i = cum_yy_im + w_yy_i;
    xz_r = cum_xz_re + w_xz_r; xz_i = cum_xz_im + w_xz_i;
}

// ---------------------------------------------------------------------------
// Main kernels -- one thread per (i_w, i_k, i_s_local) within the CURRENT
// BATCH (i_s_local indexes into the batch's own arrays, size n_s <= max_alloc_,
// not the full run's s-grid).
//
// i_s_start (always 1, passed explicitly for clarity at the call site) skips
// local index 0 in EVERY batch: in batch 0 that's the true global s=0 sample
// (no i_s-1 predecessor exists, matching the original code's unconditional
// skip of the very first snapshot); in every later batch it's the halo slice
// carried forward from the previous batch's last NEW slice, which was already
// computed and folded into the caller's accumulator there. Either way, local
// index 0 exists only to supply the i_s-1 reference for local index 1's
// finite-difference derivative, never as an output point itself.
// ---------------------------------------------------------------------------

__global__ void precompute_z_kernel(
    const double * __restrict__ phi,     // [n_s * n_z]
    const double * __restrict__ phi2,    // [n_s * n_z]
    int n_s, int n_z, int n_w, int n_k, int i_s_start,
    double ds, double dz,
    const double * __restrict__ z_arr,   // [n_z]
    const double * __restrict__ w_arr,   // [n_w]
    const double * __restrict__ k_arr,   // [n_k]
    double * __restrict__ zbuf)          // [n_w * n_k * n_s * 6]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_w * n_k * n_s) return;

    int i_s = idx % n_s;
    int i_k = (idx / n_s) % n_k;
    int i_w = idx / (n_k * n_s);
    if (i_s < i_s_start) return;

    const double w = w_arr[i_w];
    const double k = k_arr[i_k];
    if (k == 1. || k == -1.) return;

    double iz1_zz = 0., iz2_zz = 0.;
    for (int iz = 1; iz < n_z; ++iz) {
        double zm   = z_arr[iz] - dz * 0.5;
        double czm  = cos(w * k * zm);
        double q1   = (phi [iz*n_s + i_s] - phi [(iz-1)*n_s + i_s]) / dz;
        double q2   = (phi2[iz*n_s + i_s] - phi2[(iz-1)*n_s + i_s]) / dz;
        iz1_zz += dz * 2.0 * czm * q1 * q1;
        iz2_zz += dz * 2.0 * czm * q2 * q2;
    }

    double iz1_xa = 0., iz2_xa = 0.;
    for (int iz = 0; iz < n_z; ++iz) {
        double fz  = (iz == 0 || iz == n_z-1) ? 0.5 : 1.0;
        double cz  = cos(w * k * z_arr[iz]);
        double q1  = (phi [iz*n_s + i_s] - phi [iz*n_s + (i_s-1)]) / ds;
        double q2  = (phi2[iz*n_s + i_s] - phi2[iz*n_s + (i_s-1)]) / ds;
        iz1_xa += fz * dz * 2.0 * cz * q1 * q1;
        iz2_xa += fz * dz * 2.0 * cz * q2 * q2;
    }

    double iz1_xz = 0., iz2_xz = 0.;
    for (int iz = 1; iz < n_z; ++iz) {
        double szm  = sin(w * k * (z_arr[iz] - dz * 0.5));
        double ds1  = 0.5/ds * (phi [iz*n_s+i_s]     - phi [iz*n_s+(i_s-1)]
                               + phi [(iz-1)*n_s+i_s] - phi [(iz-1)*n_s+(i_s-1)]);
        double dz1  = 0.5/dz * (phi [iz*n_s+i_s]     - phi [(iz-1)*n_s+i_s]
                               + phi [iz*n_s+(i_s-1)] - phi [(iz-1)*n_s+(i_s-1)]);
        double ds2  = 0.5/ds * (phi2[iz*n_s+i_s]     - phi2[iz*n_s+(i_s-1)]
                               + phi2[(iz-1)*n_s+i_s] - phi2[(iz-1)*n_s+(i_s-1)]);
        double dz2  = 0.5/dz * (phi2[iz*n_s+i_s]     - phi2[(iz-1)*n_s+i_s]
                               + phi2[iz*n_s+(i_s-1)] - phi2[(iz-1)*n_s+(i_s-1)]);
        iz1_xz += dz * 2.0 * szm * ds1 * dz1;
        iz2_xz += dz * 2.0 * szm * ds2 * dz2;
    }

    double *out = zbuf + idx * 6;
    out[0] = iz1_zz; out[1] = iz2_zz;
    out[2] = iz1_xa; out[3] = iz2_xa;
    out[4] = iz1_xz; out[5] = iz2_xz;
}

__global__ void gw_kernel(
    int n_s, int n_w, int n_k, int i_s_start, bool is_last_batch,
    double ds,
    const double * __restrict__ w_arr,   // [n_w]
    const double * __restrict__ k_arr,   // [n_k]
    const double * __restrict__ s_arr,   // [n_s]
    const double * __restrict__ zbuf,    // [n_w * n_k * n_s * 6]
    double t_cut, double t_m, double t_0, double t_max,
    int cutoff_type, int n_floor, int panels_per_oscillation,
    double t_cut_prev,
    double * __restrict__ intbuf,        // [n_w * n_k * n_s * 6]
    double * __restrict__ plateau_intbuf,// [n_w * n_k * n_s * 6] -- same layout as intbuf, pre-transition-window
    double * __restrict__ cumbuf         // [n_w * n_k * n_s * 16] -- persists across cutoffs WITHIN one batch
#ifdef GW_KERNEL_TIMING
    , long long * __restrict__ timebuf   // [n_w*n_k*n_s*5] per-thread cycle counts:
                                          // [u_integral, zz, xa, xz, tail] -- diagnostic
                                          // build only, see Makefile's gpu_profile target.
#endif
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_w * n_k * n_s) return;

    int i_s = idx % n_s;
    int i_k = (idx / n_s) % n_k;
    int i_w = idx / (n_k * n_s);
    if (i_s < i_s_start) return;   // subsumes the old "s == 0" first-sample skip

    double s = s_arr[i_s];
    double w = w_arr[i_w];
    double k = k_arr[i_k];
    if (k == 1. || k == -1.) return;

#ifdef GW_KERNEL_TIMING
    long long t0 = clock64();
#endif

    double k_sq     = k * k;
    double Onemkk   = 1. - k_sq;
    double Sqrt1mkk = sqrt(Onemkk);
    double s_off    = s - 0.5 * ds;

    // This thread's persistent (per-batch) cum slot -- references bind
    // straight to global memory, so += on these writes through immediately
    // (no manual copy-back). Reset to 0 by the caller at the start of every
    // ProcessBatch() call (see GpuIntegrator::ProcessBatch).
    double *c = cumbuf + idx * 16;
    double &cum_zz1_re = c[0],  &cum_zz1_im = c[1],  &cum_zz2_re = c[2],  &cum_zz2_im = c[3];
    double &cum_xx1_re = c[4],  &cum_xx1_im = c[5],  &cum_xx2_re = c[6],  &cum_xx2_im = c[7];
    double &cum_yy1_re = c[8],  &cum_yy1_im = c[9],  &cum_yy2_re = c[10], &cum_yy2_im = c[11];
    double &cum_xz1_re = c[12], &cum_xz1_im = c[13], &cum_xz2_re = c[14], &cum_xz2_im = c[15];

    // --- u-integrals (incremental -- see filon_zz_incremental/filon_xyz_incremental) ---
    double zz_r1, zz_i1, zz_r2, zz_i2;
    filon_zz_incremental(s, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor,
                        panels_per_oscillation, t_cut_prev,
                        cum_zz1_re, cum_zz1_im, zz_r1, zz_i1);
    filon_zz_incremental(s, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor,
                        panels_per_oscillation, t_cut_prev,
                        cum_zz2_re, cum_zz2_im, zz_r2, zz_i2);

    double xx1, xi1, yy1, yi1, xz1, xzi1;
    double xx2, xi2, yy2, yi2, xz2, xzi2;
    filon_xyz_incremental(s_off, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor,
                        panels_per_oscillation, t_cut_prev,
                        cum_xx1_re, cum_xx1_im, cum_yy1_re, cum_yy1_im, cum_xz1_re, cum_xz1_im,
                        xx1, xi1, yy1, yi1, xz1, xzi1);
    filon_xyz_incremental(s_off, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor,
                        panels_per_oscillation, t_cut_prev,
                        cum_xx2_re, cum_xx2_im, cum_yy2_re, cum_yy2_im, cum_xz2_re, cum_xz2_im,
                        xx2, xi2, yy2, yi2, xz2, xzi2);

#ifdef GW_KERNEL_TIMING
    long long t1 = clock64();
#endif

    const double *izvals = zbuf + idx * 6;
    const double iz1_zz = izvals[0], iz2_zz = izvals[1];

#ifdef GW_KERNEL_TIMING
    long long t2 = clock64();
#endif

    const double iz1_xa = izvals[2], iz2_xa = izvals[3];

#ifdef GW_KERNEL_TIMING
    long long t3 = clock64();
#endif

    const double iz1_xz = izvals[4], iz2_xz = izvals[5];

#ifdef GW_KERNEL_TIMING
    long long t4 = clock64();
#endif

    // --- Accumulate linear per-s contributions ---
    // fac's half-weight applies only to the single LAST slice of the WHOLE
    // run (a trapezoidal-rule endpoint), not merely the last slice of this
    // batch -- so it must gate on is_last_batch, not just i_s==n_s-1.
    // pre_zz uses the physical s value (not the batch-local thread index
    // i_s), since i_s is no longer a global index once batched.
    double fac    = (is_last_batch && i_s == n_s-1) ? 0.5 : 1.;
    double pre_zz = fac * s * s * ds;
    double pre_xa = 0.5 * s_off * s_off * ds;
    double pre_xz = -s_off * s_off * ds;

    int base = (i_w * n_k * n_s + i_k * n_s + i_s) * 6;
    intbuf[base + 0] = pre_zz * (zz_r1*iz1_zz + zz_r2*iz2_zz);
    intbuf[base + 1] = pre_zz * (zz_i1*iz1_zz + zz_i2*iz2_zz);
    intbuf[base + 2] = pre_xa * ((xx1*k_sq - yy1)*iz1_xa + (xx2*k_sq - yy2)*iz2_xa);
    intbuf[base + 3] = pre_xa * ((xi1*k_sq - yi1)*iz1_xa + (xi2*k_sq - yi2)*iz2_xa);
    intbuf[base + 4] = pre_xz * (xz1*iz1_xz + xz2*iz2_xz);
    intbuf[base + 5] = pre_xz * (xzi1*iz1_xz + xzi2*iz2_xz);

    // Plateau-only mirror of the six lines above: same combination, but
    // using the cum_* values (this cutoff's plateau contribution, AFTER its
    // own filon_*_incremental update above but BEFORE its transition-window
    // piece) instead of zz_r1/xx1/etc (which include that window). Cheap --
    // reuses iz*_zz/xa/xz and pre_zz/pre_xa/pre_xz already in registers.
    // Written every launch; the host only copies/reduces this for i_t ==
    // n_t_-1 (see GpuIntegrator::ProcessBatch), since that is the only
    // cutoff a future checkpoint can be taken from.
    plateau_intbuf[base + 0] = pre_zz * (cum_zz1_re*iz1_zz + cum_zz2_re*iz2_zz);
    plateau_intbuf[base + 1] = pre_zz * (cum_zz1_im*iz1_zz + cum_zz2_im*iz2_zz);
    plateau_intbuf[base + 2] = pre_xa * ((cum_xx1_re*k_sq - cum_yy1_re)*iz1_xa
                                        + (cum_xx2_re*k_sq - cum_yy2_re)*iz2_xa);
    plateau_intbuf[base + 3] = pre_xa * ((cum_xx1_im*k_sq - cum_yy1_im)*iz1_xa
                                        + (cum_xx2_im*k_sq - cum_yy2_im)*iz2_xa);
    plateau_intbuf[base + 4] = pre_xz * (cum_xz1_re*iz1_xz + cum_xz2_re*iz2_xz);
    plateau_intbuf[base + 5] = pre_xz * (cum_xz1_im*iz1_xz + cum_xz2_im*iz2_xz);

#ifdef GW_KERNEL_TIMING
    long long t5 = clock64();
    long long *tb = timebuf + idx * 5;
    tb[0] = t1 - t0;   // u-integral
    tb[1] = t2 - t1;   // zz z-integral loop
    tb[2] = t3 - t2;   // xa z-integral loop
    tb[3] = t4 - t3;   // xz z-integral loop
    tb[4] = t5 - t4;   // tail (accumulate + intbuf write)
#endif
}

// ---------------------------------------------------------------------------
// On-device field evolution (RunEvolution()'s kernels)
//
// device_dV/evolve_pi_kernel/phi_update_kernel are the GPU equivalent of
// Evolution::EvolvePi and its phi update (evolution.cpp) -- same stencil,
// same boundary clamping, same leapfrog-style pi update. CUDA device code
// can't dispatch through a host-constructed Potential* vtable, so
// device_dV switches on DevicePotentialParams::kind instead (see
// cpp/common/potential.h's to_device_params()).
//
// write_snapshot_kernel/repack_batch_kernel/carry_halo_kernel manage
// d_phi_evolve_, the fixed-max_alloc_-stride staging buffer RunEvolution()
// writes new snapshots into one at a time (its final size, n_s_local, isn't
// known until a batch closes, so it can't be written directly into d_phi_'s
// tightly-packed [iz*n_s_local+is] layout the way ProcessBatch()'s host
// upload can) -- repack_batch_kernel copies a closed batch's first
// n_s_local columns into d_phi_ unchanged, so precompute_z_kernel/gw_kernel
// need no modification at all.
// ---------------------------------------------------------------------------

__device__ __forceinline__ double device_dV(double phi, DevicePotentialParams pot)
{
    switch (pot.kind) {
    case PotentialKind::kPhi4:
        return 2.*pot.c[0]*phi + 3.*pot.c[1]*phi*phi + 4.*pot.c[2]*phi*phi*phi;
    case PotentialKind::kPhi4Piecewise:
        if (phi <= pot.c[3])
            return 2.*pot.c[0]*phi + 3.*pot.c[1]*phi*phi + 4.*pot.c[2]*phi*phi*phi;
        return pot.c[4]*pot.c[4]*(phi - pot.c[5]);
    case PotentialKind::kPolynomial:
        return phi*phi*phi - phi*phi + (2./9.)*pot.c[0]*phi;
    }
    return 0.;   // unreachable
}

// One thread per z. Reads phi (read-only this kernel), writes a new pi in
// place -- matches Evolution::EvolvePi exactly, including the i_z==0/n_z-1
// boundary clamp (reflecting boundary conditions).
__global__ void evolve_pi_kernel(
    const double * __restrict__ phi, double * __restrict__ pi,
    int n_z, double dz, double s, double step, DevicePotentialParams pot)
{
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    if (iz >= n_z) return;

    double phiprev = (iz != 0)     ? phi[iz-1] : phi[1];
    double phinext = (iz != n_z-1) ? phi[iz+1] : phi[n_z-2];
    double lap = (phiprev - 2.*phi[iz] + phinext) / (dz*dz);
    double dv  = device_dV(phi[iz], pot);

    double fac_pi  = 1. - 2.*step/(s+step);
    double fac_src =      s*step/(s+step);
    pi[iz] = pi[iz]*fac_pi + fac_src*(lap - dv);
}

// One thread per z. phi[iz] += step*pi[iz] -- matches Evolution::Evolve()'s
// phi update exactly.
__global__ void phi_update_kernel(
    double * __restrict__ phi, const double * __restrict__ pi,
    int n_z, double step)
{
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    if (iz >= n_z) return;
    phi[iz] += step * pi[iz];
}

// One thread per z. stage[iz*max_alloc+is_local] = phi_cur[iz].
__global__ void write_snapshot_kernel(
    const double * __restrict__ phi_cur, double * __restrict__ stage,
    int n_z, int max_alloc, int is_local)
{
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    if (iz >= n_z) return;
    stage[iz * max_alloc + is_local] = phi_cur[iz];
}

// One thread per (iz,is) pair, is in [0,n_s_local). Copies stage's first
// n_s_local columns (fixed max_alloc_ stride) into dst's tightly-packed
// [iz*n_s_local+is] layout -- dst is d_phi_, unchanged from ProcessBatch()'s
// own expectations.
__global__ void repack_batch_kernel(
    const double * __restrict__ stage, double * __restrict__ dst,
    int n_z, int max_alloc, int n_s_local)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_z * n_s_local) return;
    int is = idx % n_s_local;
    int iz = idx / n_s_local;
    dst[iz * n_s_local + is] = stage[iz * max_alloc + is];
}

// One thread per z. Same-buffer column copy: stage[:,to_is] = stage[:,from_is]
// -- used to carry the last-written slice forward as the next batch's halo.
__global__ void carry_halo_kernel(
    double * __restrict__ stage, int n_z, int max_alloc, int from_is, int to_is)
{
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    if (iz >= n_z) return;
    stage[iz * max_alloc + to_is] = stage[iz * max_alloc + from_is];
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GpuIntegrator::GpuIntegrator(const Setup &setup, int param,
                             int panels_per_oscillation, int s_batch_size,
                             const Checkpoint *seed,
                             const StreamCheckpoint *resume)
    : n_k_(setup.n_k), n_w_(setup.n_w), n_z_(setup.z.size()),
      n_t_(static_cast<std::size_t>(setup.n_t)),
      ds_(setup.ds * setup.how_often_ds),
      dz_(std::abs(setup.z[1] - setup.z[0])),
      t_cut_base_(setup.t_cut), t_m_base_(setup.t_m),
      t_max_base_(setup.t_max), t_0_(setup.t_0),
      d_(setup.d), cutoff_type_(setup.cutoff_type),
      z_(setup.z), wlist_(setup.wlist), times_(setup.times),
      param_floor_(param > 0 ? (param + 1) & ~1 : -1),
      panels_per_oscillation_(panels_per_oscillation),
      s_batch_size_(s_batch_size), max_alloc_(s_batch_size + 1)
{
    if (s_batch_size_ < 1)
        throw std::runtime_error("GpuIntegrator: s_batch_size must be >= 1");

    klist_ = linspace(0., 1., static_cast<int>(n_k_));
    accum_.assign(n_t_ * n_w_ * n_k_ * 6, 0.);
    accum_plateau_.assign(n_w_ * n_k_ * 6, 0.);

    // --- Seed from a prior checkpoint, if given ---
    // Every requested cutoff must be strictly later than the checkpoint's
    // own seed time: the plateau/transition-window decomposition this
    // relies on only holds going forward -- a requested time at or before
    // seed_time would need this row's u-integral re-run from s=0 for that
    // cutoff, which a seeded run does not do.
    if (seed) {
        for (double t : times_) {
            if (t <= seed->seed_time)
                throw std::runtime_error(
                    "GpuIntegrator: seeded run requires every setup.times[] "
                    "entry to be strictly greater than the checkpoint's "
                    "seed_time (" + std::to_string(seed->seed_time) +
                    "); got " + std::to_string(t));
        }
        const std::size_t n_wk = n_w_ * n_k_;
        if (seed->plateau_re.size() != n_wk || seed->plateau_im.size() != n_wk)
            throw std::runtime_error(
                "GpuIntegrator: checkpoint plateau_re/plateau_im size does "
                "not match this setup's n_w*n_k -- checkpoint is from a "
                "different (n_w, n_k) than the current setup.h5");
        has_seed_    = true;
        seed_t_cut_  = t_cut_base_ - (t_m_base_ - seed->seed_time);
        seed_re_     = seed->plateau_re;
        seed_im_     = seed->plateau_im;
    }

    // --- Resume mid-stream progress from a prior, interrupted attempt at
    // this EXACT times range, if given (independent of `seed` above -- see
    // the constructor's doc comment). ---
    if (resume) {
        const std::size_t n_accum   = n_t_ * n_w_ * n_k_ * 6;
        const std::size_t n_plateau = n_w_ * n_k_ * 6;
        if (resume->accum.size() != n_accum || resume->accum_plateau.size() != n_plateau)
            throw std::runtime_error(
                "GpuIntegrator: StreamCheckpoint accum size does not match this "
                "setup's n_t*n_w*n_k -- checkpoint is from a different times "
                "request than the current setup.h5");
        accum_         = resume->accum;
        accum_plateau_ = resume->accum_plateau;
        skip_until_    = resume->batches_done;
    }

    // --- Build the persistent phi0 interpolator directly from setup.phi0 ---
    // (build_phi2's original logic only ever read input_phi[0], i.e. the t=0
    // profile identical to setup.phi0 -- this has zero dependence on the
    // evolved history, so it can be built once here instead of per-batch.)
    auto max_it = std::max_element(setup.phi0.begin(), setup.phi0.end());
    int phimid  = static_cast<int>(max_it - setup.phi0.begin());

    z0_new_.assign(z_.begin() + phimid, z_.end());
    double z0 = z0_new_[0];
    for (double &zi : z0_new_) zi -= z0;

    phi0_new_.assign(setup.phi0.begin() + phimid, setup.phi0.end());
    phi0_interp_.emplace(z0_new_, phi0_new_);

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GpuIntegrator: device=" << prop.name
              << "  n_w=" << n_w_ << " n_k=" << n_k_
              << " n_z=" << n_z_ << " s_batch_size=" << s_batch_size_
              << " (max_alloc=" << max_alloc_ << ")"
              << "\n  per-batch zbuf/intbuf=" << (n_w_*n_k_*max_alloc_*6*8)/(1<<20)
              << " MB  cumbuf=" << (n_w_*n_k_*max_alloc_*16*8)/(1<<20) << " MB\n\n";

    // --- Allocate persistent device buffers ONCE, sized for max_alloc_ ---
    // (batch-sized, not full-s-grid-sized) and reused for every ProcessBatch()
    // call -- no per-batch cudaMalloc/cudaFree.
    auto t_device_setup = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMalloc(&d_phi_,  n_z_ * max_alloc_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi2_, n_z_ * max_alloc_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s_,    max_alloc_ * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_phi_cur_,    n_z_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pi_cur_,     n_z_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi_evolve_, n_z_ * max_alloc_ * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_z_, n_z_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_z_, z_.data(), n_z_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w_, n_w_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_w_, wlist_.data(), n_w_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_k_, n_k_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_k_, klist_.data(), n_k_ * sizeof(double), cudaMemcpyHostToDevice));

    const std::size_t batch_total = n_w_ * n_k_ * static_cast<std::size_t>(max_alloc_);
    CUDA_CHECK(cudaMalloc(&d_zbuf_,   batch_total * 6  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_intbuf_, batch_total * 6  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_plateau_intbuf_, batch_total * 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cumbuf_, batch_total * 16 * sizeof(double)));

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::cout << "Timing phase=gpu_device_setup seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_device_setup).count()
              << "\nGPU memory after setup: free=" << free_b/(1<<20)
              << " MB  total=" << total_b/(1<<20) << " MB\n";

#ifdef GW_KERNEL_TIMING
    CUDA_CHECK(cudaMalloc(&d_timebuf_, batch_total * 5 * sizeof(long long)));
#endif
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

GpuIntegrator::~GpuIntegrator()
{
    cudaFree(d_phi_);
    cudaFree(d_phi2_);
    cudaFree(d_z_);
    cudaFree(d_w_);
    cudaFree(d_k_);
    cudaFree(d_s_);
    cudaFree(d_phi_cur_);
    cudaFree(d_pi_cur_);
    cudaFree(d_phi_evolve_);
    cudaFree(d_zbuf_);
    cudaFree(d_intbuf_);
    cudaFree(d_plateau_intbuf_);
    cudaFree(d_cumbuf_);
#ifdef GW_KERNEL_TIMING
    cudaFree(d_timebuf_);
#endif
}

// ---------------------------------------------------------------------------
// phi2 construction, per batch (CPU) -- reuses the persistent phi0_interp_
// built once in the constructor; the math is otherwise identical to the
// original build_phi2 (cu, pre-rewrite), just evaluated over the batch's own
// s values instead of the full [0, n_s) range.
// ---------------------------------------------------------------------------

void GpuIntegrator::build_phi2_batch(const std::vector<double> &batch_s,
                                      std::vector<double> &phi2_host) const
{
    const std::size_t n_batch = batch_s.size();
    phi2_host.resize(n_batch * n_z_);
    for (std::size_t is = 0; is < n_batch; ++is) {
        double s_val = batch_s[is];
        for (std::size_t iz = 0; iz < n_z_; ++iz) {
            double z_val = iz * dz_;
            double r1 = std::sqrt(s_val*s_val + (z_val - d_/2.)*(z_val - d_/2.));
            double r2 = std::sqrt(s_val*s_val + (z_val + d_/2.)*(z_val + d_/2.));
            phi2_host[iz * n_batch + is] = (*phi0_interp_)(r1) + (*phi0_interp_)(r2);
        }
    }
}

// ---------------------------------------------------------------------------
// ProcessBatch -- uploads one batch (<= max_alloc_ slices, index 0 being the
// halo slice for every batch after the first), computes this batch's
// z-integrals once, then loops ALL cutoff times internally (the incremental
// plateau/transition-window algorithm is preserved exactly, just scoped to
// this batch's cum state instead of the whole run's), folding each cutoff's
// linear contribution into the persistent, batch-independent accum_.
// ---------------------------------------------------------------------------

bool GpuIntegrator::AdvanceAndCheckSkip(bool is_last_batch)
{
    // --- Format-B resume: skip batches already folded into accum_/accum_plateau_ ---
    // Field evolution still regenerates this batch's data (main.cpp restarts
    // it from s=0 every run -- cheap), but it's never uploaded or handed to
    // a kernel here, since its contribution is already in accum_. Guards
    // is_last_batch too: if the interrupted run got as far as streaming
    // everything (just never reached its own finalize/checkpoint-delete
    // step), this lets a resume skip straight to being finalized_ with zero
    // GPU work at all.
    const int this_batch = next_batch_index_++;
    if (this_batch < skip_until_) {
        if (is_last_batch)
            finalized_ = true;
        return true;
    }
    return false;
}

void GpuIntegrator::ProcessBatch(const std::vector<std::vector<double>> &batch_phi,
                                 const std::vector<double> &batch_s,
                                 bool is_first_batch, bool is_last_batch)
{
    if (finalized_)
        throw std::runtime_error(
            "GpuIntegrator::ProcessBatch called after the last batch was already processed");

    if (AdvanceAndCheckSkip(is_last_batch))
        return;

    const int n_s_local = static_cast<int>(batch_phi.size());
    if (n_s_local < 1 || n_s_local > max_alloc_)
        throw std::runtime_error(
            "GpuIntegrator::ProcessBatch: batch size " + std::to_string(n_s_local) +
            " out of range (1.." + std::to_string(max_alloc_) + ")");

    // --- Flatten phi to host 1-D array, transposed to [iz*n_s_local+is] ---
    std::vector<double> phi_host(static_cast<std::size_t>(n_s_local) * n_z_);
    for (int is = 0; is < n_s_local; ++is)
        for (std::size_t iz = 0; iz < n_z_; ++iz)
            phi_host[iz * n_s_local + is] = batch_phi[is][iz];
    CUDA_CHECK(cudaMemcpy(d_phi_, phi_host.data(), phi_host.size() * sizeof(double), cudaMemcpyHostToDevice));

    ProcessUploadedBatch(n_s_local, batch_s, is_first_batch, is_last_batch);
}

// ---------------------------------------------------------------------------
// ProcessUploadedBatch -- shared tail of ProcessBatch()/RunEvolution().
// Assumes d_phi_'s first n_z_*n_s_local doubles are ALREADY populated (by
// whichever caller); builds/uploads phi2 and s, computes this batch's
// z-integrals once, then loops ALL cutoff times internally (the incremental
// plateau/transition-window algorithm is preserved exactly, just scoped to
// this batch's cum state instead of the whole run's), folding each cutoff's
// linear contribution into the persistent, batch-independent accum_.
// ---------------------------------------------------------------------------

void GpuIntegrator::ProcessUploadedBatch(int n_s_local, const std::vector<double> &batch_s,
                                         bool is_first_batch, bool is_last_batch)
{
    // Local index 0 is ALWAYS skipped as an output/accumulation point,
    // regardless of is_first_batch: in batch 0 it's the true global s=0
    // sample (no i_s-1 predecessor exists, so no s-derivative can be formed
    // -- matches the original unconditional "if (i_s==0) return"/"if (s==0)
    // return" checks); in every later batch it's the halo slice carried
    // forward from the previous batch's last NEW slice, which was already
    // computed and folded into accum_ there -- reprocessing it here would
    // double-count it. Either way, its only purpose here is to supply the
    // i_s-1 reference for local index 1's finite-difference derivative.
    // A batch with n_s_local==1 (only the halo/initial slice) legitimately
    // contributes nothing -- the loops below over i_s in [1, n_s_local) are
    // simply empty in that case.
    const int i_s_start = 1;

    const auto t_batch = std::chrono::steady_clock::now();

    // --- Build phi2 for this batch on the host ---
    std::vector<double> phi2_host;
    build_phi2_batch(batch_s, phi2_host);

    // --- Upload this batch's phi2/s (phi is already in d_phi_) ---
    CUDA_CHECK(cudaMemcpy(d_phi2_, phi2_host.data(), phi2_host.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_,    batch_s.data(),   batch_s.size()   * sizeof(double), cudaMemcpyHostToDevice));

    const std::size_t total = n_w_ * n_k_ * static_cast<std::size_t>(n_s_local);
    constexpr int BLOCK = 256;
    const int grid = (static_cast<int>(total) + BLOCK - 1) / BLOCK;

    // --- z-integrals for this batch only ---
    CUDA_CHECK(cudaMemset(d_zbuf_, 0, total * 6 * sizeof(double)));
    precompute_z_kernel<<<grid, BLOCK>>>(
        d_phi_, d_phi2_, n_s_local, static_cast<int>(n_z_),
        static_cast<int>(n_w_), static_cast<int>(n_k_), i_s_start,
        ds_, dz_, d_z_, d_w_, d_k_, d_zbuf_);
    CUDA_CHECK(cudaGetLastError());

    // --- Reset this batch's incremental plateau state ---
    // (each s belongs to exactly one batch -- cum state must NOT persist
    // across batches, only across the cutoff-time loop within this one).
    // If this run was seeded from a checkpoint, t_cut_prev_ starts at the
    // seed's own t_cut instead of -inf, so the first i_t processed in every
    // batch skips re-integrating the u-range the seed's row already covered
    // (see the constructor's seed_t_cut_ derivation).
    CUDA_CHECK(cudaMemset(d_cumbuf_, 0, total * 16 * sizeof(double)));
    t_cut_prev_ = has_seed_ ? seed_t_cut_ : kNegInfSentinel;
    CUDA_CHECK(cudaDeviceSynchronize());

    n_floor_ = param_floor_ > 0 ? param_floor_ : 8192;

    std::vector<double> buf(total * 6);
    std::vector<double> plateau_buf;   // only filled/used for i_t == n_t_-1
    double kernel_seconds = 0., transfer_seconds = 0., reduce_seconds = 0.;

    for (std::size_t i_t = 0; i_t < n_t_; ++i_t) {
        const auto t_kernel = std::chrono::steady_clock::now();

        double shift  = t_m_base_  - times_[i_t];
        double t_cut  = t_cut_base_ - shift;
        double t_m    = t_m_base_   - shift;
        double t_max  = t_max_base_ - shift;

        CUDA_CHECK(cudaMemset(d_intbuf_, 0, total * 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_plateau_intbuf_, 0, total * 6 * sizeof(double)));
        gw_kernel<<<grid, BLOCK>>>(
            n_s_local, static_cast<int>(n_w_), static_cast<int>(n_k_),
            i_s_start, is_last_batch,
            ds_, d_w_, d_k_, d_s_, d_zbuf_,
            t_cut, t_m, t_0_, t_max,
            cutoff_type_, n_floor_, panels_per_oscillation_, t_cut_prev_,
            d_intbuf_, d_plateau_intbuf_, d_cumbuf_
#ifdef GW_KERNEL_TIMING
            , d_timebuf_
#endif
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        kernel_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_kernel).count();

        const auto t_transfer = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(buf.data(), d_intbuf_, buf.size() * sizeof(double), cudaMemcpyDeviceToHost));
        const bool is_last_i_t = (i_t == n_t_ - 1);
        if (is_last_i_t) {
            plateau_buf.resize(total * 6);
            CUDA_CHECK(cudaMemcpy(plateau_buf.data(), d_plateau_intbuf_,
                                  plateau_buf.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        transfer_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_transfer).count();

        const auto t_reduce = std::chrono::steady_clock::now();
        double *acc = &accum_[i_t * n_w_ * n_k_ * 6];
        for (std::size_t iw = 0; iw < n_w_; ++iw) {
            for (std::size_t ik = 0; ik < n_k_; ++ik) {
                double sums[6] = {0., 0., 0., 0., 0., 0.};
                double plateau_sums[6] = {0., 0., 0., 0., 0., 0.};
                for (int is = i_s_start; is < n_s_local; ++is) {
                    std::size_t base = (iw * n_k_ * n_s_local + ik * n_s_local + is) * 6;
                    for (int c = 0; c < 6; ++c) sums[c] += buf[base + c];
                    if (is_last_i_t)
                        for (int c = 0; c < 6; ++c) plateau_sums[c] += plateau_buf[base + c];
                }
                double *dst = &acc[(iw * n_k_ + ik) * 6];
                for (int c = 0; c < 6; ++c) dst[c] += sums[c];
                if (is_last_i_t) {
                    double *pdst = &accum_plateau_[(iw * n_k_ + ik) * 6];
                    for (int c = 0; c < 6; ++c) pdst[c] += plateau_sums[c];
                }
            }
        }
        reduce_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_reduce).count();

        t_cut_prev_ = t_cut;
    }

    std::cout << "Timing phase=batch size=" << n_s_local
              << " is_first=" << is_first_batch << " is_last=" << is_last_batch
              << " kernel_seconds=" << kernel_seconds
              << " transfer_seconds=" << transfer_seconds
              << " reduction_seconds=" << reduce_seconds
              << " total_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_batch).count()
              << "\n";

    if (is_last_batch)
        finalized_ = true;
}

// ---------------------------------------------------------------------------
// RunEvolution -- on-device field evolution (s=0..smax), streamed straight
// into ProcessUploadedBatch() at the same cadence Evolution+SBatchWindow use
// on the CPU path. Mirrors Evolution::Evolve()'s exact step/snapshot
// sequence (evolution.cpp) and SBatchWindow's exact batching/halo bookkeeping
// (s_batch_window.h) -- see gpu_integrator.cuh's doc comment. Must stay in
// lockstep with both if either changes.
// ---------------------------------------------------------------------------

void GpuIntegrator::RunEvolution(const Setup &setup,
                                 std::function<void(bool)> on_batch_boundary)
{
    const auto t_run = std::chrono::steady_clock::now();
    double integration_seconds = 0.;

    const int n_z = static_cast<int>(n_z_);
    const double ds = setup.ds;
    const int how_often_ds = setup.how_often_ds;
    const int baby_steps   = setup.baby_steps;
    const int n_steps = std::max(1, static_cast<int>(std::round(setup.smax / ds)));
    const int last_saved_i = (n_steps / how_often_ds) * how_often_ds;
    const int batch_size = s_batch_size_;

    constexpr int BLOCK = 256;
    const int grid_z = (n_z + BLOCK - 1) / BLOCK;

    const DevicePotentialParams pp = setup.potential->to_device_params();

    CUDA_CHECK(cudaMemcpy(d_phi_cur_, setup.phi0.data(), n_z_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_pi_cur_, 0, n_z_ * sizeof(double)));

    // --- Batch bookkeeping, mirrors SBatchWindow::Push exactly, but writes
    // snapshots into d_phi_evolve_ instead of buffering host vectors. ---
    int  is_local         = 0;
    bool first_batch_done = false;
    std::vector<double> batch_s;
    batch_s.reserve(static_cast<std::size_t>(max_alloc_));

    auto push_snapshot = [&](double s, bool is_last_snapshot) {
        write_snapshot_kernel<<<grid_z, BLOCK>>>(d_phi_cur_, d_phi_evolve_, n_z, max_alloc_, is_local);
        CUDA_CHECK(cudaGetLastError());
        batch_s.push_back(s);
        ++is_local;

        const int new_count = is_local - (first_batch_done ? 1 : 0);
        if (new_count < batch_size && !is_last_snapshot)
            return;

        const int n_s_local       = is_local;
        const bool is_first_batch = !first_batch_done;

        if (!AdvanceAndCheckSkip(is_last_snapshot)) {
            const auto t_int = std::chrono::steady_clock::now();
            const int total_zs   = n_z * n_s_local;
            const int grid_repack = (total_zs + BLOCK - 1) / BLOCK;
            repack_batch_kernel<<<grid_repack, BLOCK>>>(
                d_phi_evolve_, d_phi_, n_z, max_alloc_, n_s_local);
            CUDA_CHECK(cudaGetLastError());
            ProcessUploadedBatch(n_s_local, batch_s, is_first_batch, is_last_snapshot);
            integration_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_int).count();
        }
        first_batch_done = true;

        if (on_batch_boundary)
            on_batch_boundary(is_last_snapshot);

        if (is_last_snapshot) {
            batch_s.clear();
            is_local = 0;
            return;
        }

        // --- Carry the last-written slice forward as the next batch's halo. ---
        carry_halo_kernel<<<grid_z, BLOCK>>>(d_phi_evolve_, n_z, max_alloc_, n_s_local - 1, 0);
        CUDA_CHECK(cudaGetLastError());
        const double halo_s = batch_s.back();
        batch_s.clear();
        batch_s.push_back(halo_s);
        is_local = 1;
    };

    // --- Initial snapshot: s=0, phi=phi0 (BEFORE any stepping), matching
    // the CPU constructor's sink_(phi, 0., last_saved_i_ == 0). ---
    push_snapshot(0., last_saved_i == 0);

    // --- Baby-step bootstrap (matches EvolvepiFirstHalfStep exactly -- no
    // snapshot taken during this phase, matching the CPU original). ---
    {
        const double baby_ds = (0.5 * ds) / static_cast<double>(baby_steps - 1);
        for (int i = 1; i < baby_steps; ++i) {
            const double s = (i - 1) * baby_ds;
            evolve_pi_kernel<<<grid_z, BLOCK>>>(d_phi_cur_, d_pi_cur_, n_z, dz_, s, baby_ds, pp);
            CUDA_CHECK(cudaGetLastError());
            phi_update_kernel<<<grid_z, BLOCK>>>(d_phi_cur_, d_pi_cur_, n_z, baby_ds);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // --- Main loop (matches Evolve() exactly, including skipping the pi
    // update at i==1 -- pi is carried over from the baby-step bootstrap). ---
    for (int i = 1; i <= n_steps; ++i) {
        if (i > 1) {
            const double s = (i - 1) * ds;
            evolve_pi_kernel<<<grid_z, BLOCK>>>(d_phi_cur_, d_pi_cur_, n_z, dz_, s, ds, pp);
            CUDA_CHECK(cudaGetLastError());
        }
        phi_update_kernel<<<grid_z, BLOCK>>>(d_phi_cur_, d_pi_cur_, n_z, ds);
        CUDA_CHECK(cudaGetLastError());

        if (i % how_often_ds == 0)
            push_snapshot(i * ds, i == last_saved_i);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_run).count();
    std::cout << "Timing phase=field_evolution seconds=" << (total_seconds - integration_seconds) << "\n";
    std::cout << "Timing phase=gpu_integration seconds=" << integration_seconds << "\n";
}

// ---------------------------------------------------------------------------
// Finalize / FinalizeAmplitude -- pure CPU reads of accum_, no kernel launch.
// Combines the 6 linear totals (summed over every batch's local i_s) into
// re/im, THEN squares and k-integrates -- identical math to the original
// RunAndReduce's tail, just reading from the persistent accumulator instead
// of a freshly-copied full-s-grid intbuf.
// ---------------------------------------------------------------------------

std::vector<double> GpuIntegrator::FinalizeImpl(int i_t,
                                                std::vector<double> *out_amp_re,
                                                std::vector<double> *out_amp_im) const
{
    if (!finalized_)
        throw std::runtime_error(
            "GpuIntegrator::Finalize called before the last batch was processed");
    if (i_t < 0 || static_cast<std::size_t>(i_t) >= n_t_)
        throw std::runtime_error("GpuIntegrator::Finalize: i_t out of range");

    const double dk = klist_[1] - klist_[0];
    const double *acc = &accum_[static_cast<std::size_t>(i_t) * n_w_ * n_k_ * 6];

    std::vector<double> spectrum(n_w_, 0.);

    for (std::size_t iw = 0; iw < n_w_; ++iw) {
        double w = wlist_[iw];
        double int_k = 0.;

        for (std::size_t ik = 0; ik < n_k_; ++ik) {
            double k      = klist_[ik];
            double Onemkk = 1. - k*k;
            double TwokSq = 2. * k * std::sqrt(Onemkk);

            const double *s6 = &acc[(iw * n_k_ + ik) * 6];
            double szz_r = s6[0], szz_i = s6[1];
            double sxa_r = s6[2], sxa_i = s6[3];
            double sxz_r = s6[4], sxz_i = s6[5];

            double re = szz_r*Onemkk + sxa_r - TwokSq*sxz_r;
            double im = szz_i*Onemkk + sxa_i - TwokSq*sxz_i;
            // If this run was seeded from a checkpoint, add the seed's own
            // plateau contribution (everything up to the seed row's t_cut)
            // -- this run's accum_ only ever holds ITS OWN increment, built
            // from t_cut_prev_ starting at seed_t_cut_ (see ProcessBatch),
            // so re/im here is otherwise only the piece from the seed's
            // time onward, not the full result. The combination is linear
            // (Onemkk/TwokSq are fixed per (w,k)), so adding seed_re_/
            // seed_im_ here -- rather than needing the seed's own raw
            // szz/sxa/sxz -- is exact, not an approximation.
            if (has_seed_) {
                re += seed_re_[iw * n_k_ + ik];
                im += seed_im_[iw * n_k_ + ik];
            }
            if (out_amp_re) {
                (*out_amp_re)[iw * n_k_ + ik] = re;
                (*out_amp_im)[iw * n_k_ + ik] = im;
            }
            double intk_val = (re*re + im*im) * w*w*w * 2.*M_PI;

            double fk = (ik == 0 || ik == n_k_-1) ? 1. : 2.;
            int_k += intk_val * dk * fk;
        }

        spectrum[iw] = int_k;
    }

    return spectrum;
}

std::vector<double> GpuIntegrator::Finalize(int i_t) const
{
    return FinalizeImpl(i_t, nullptr, nullptr);
}

AmplitudeResult GpuIntegrator::FinalizeAmplitude(int i_t) const
{
    AmplitudeResult res;
    res.w      = wlist_;
    res.klist  = klist_;
    res.amp_re.assign(n_w_ * n_k_, 0.);
    res.amp_im.assign(n_w_ * n_k_, 0.);
    res.spectrum = FinalizeImpl(i_t, &res.amp_re, &res.amp_im);
    return res;
}

// ---------------------------------------------------------------------------
// Checkpointing -- see gpu_integrator.cuh's Checkpoint doc comment.
// CombinePlateau applies the SAME linear (Onemkk/TwokSq) combination
// FinalizeImpl uses on accum_, but to accum_plateau_ instead, and stops
// before squaring/k-integration -- accum_plateau_ only ever holds the LAST
// cutoff index's pre-transition-window contribution (see ProcessBatch),
// summed over every batch this run processed.
// ---------------------------------------------------------------------------

void GpuIntegrator::CombinePlateau(std::vector<double> &plateau_re,
                                   std::vector<double> &plateau_im) const
{
    plateau_re.assign(n_w_ * n_k_, 0.);
    plateau_im.assign(n_w_ * n_k_, 0.);

    for (std::size_t iw = 0; iw < n_w_; ++iw) {
        for (std::size_t ik = 0; ik < n_k_; ++ik) {
            double k      = klist_[ik];
            double Onemkk = 1. - k*k;
            double TwokSq = 2. * k * std::sqrt(Onemkk);

            const double *s6 = &accum_plateau_[(iw * n_k_ + ik) * 6];
            double szz_r = s6[0], szz_i = s6[1];
            double sxa_r = s6[2], sxa_i = s6[3];
            double sxz_r = s6[4], sxz_i = s6[5];

            plateau_re[iw * n_k_ + ik] = szz_r*Onemkk + sxa_r - TwokSq*sxz_r;
            plateau_im[iw * n_k_ + ik] = szz_i*Onemkk + sxa_i - TwokSq*sxz_i;
        }
    }
}

GpuIntegrator::Checkpoint GpuIntegrator::MakeCheckpoint() const
{
    if (!finalized_)
        throw std::runtime_error(
            "GpuIntegrator::MakeCheckpoint called before the last batch was processed");

    Checkpoint cp;
    cp.seed_time = times_.back();

    CombinePlateau(cp.plateau_re, cp.plateau_im);

    // Chain: if THIS run was itself seeded, its own plateau (just computed)
    // only covers seed_t_cut_ onward -- add the prior seed back in so the
    // checkpoint this run produces is the FULL plateau from s=0 of the
    // original row, exactly like FinalizeImpl does for the spectrum output.
    if (has_seed_) {
        for (std::size_t i = 0; i < cp.plateau_re.size(); ++i) {
            cp.plateau_re[i] += seed_re_[i];
            cp.plateau_im[i] += seed_im_[i];
        }
    }

    return cp;
}

GpuIntegrator::StreamCheckpoint GpuIntegrator::MakeStreamCheckpoint() const
{
    StreamCheckpoint sc;
    sc.batches_done    = next_batch_index_;
    sc.accum           = accum_;
    sc.accum_plateau   = accum_plateau_;
    return sc;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::vector<double> GpuIntegrator::linspace(double a, double b, int n)
{
    std::vector<double> v(n);
    double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) v[i] = a + i * step;
    return v;
}

std::vector<double> GpuIntegrator::geomspace(double a, double b, std::size_t n)
{
    if (n == 0) return {};
    std::vector<double> v(n);
    double la = std::log10(a), lb = std::log10(b);
    double step = (lb - la) / (n - 1);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::pow(10., la + i * step);
    return v;
}
