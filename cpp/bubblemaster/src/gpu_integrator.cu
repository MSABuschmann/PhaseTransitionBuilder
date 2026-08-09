#include "gpu_integrator.cuh"
#include "interpolator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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
    double Se_c, Se_s;   // β sums (even-index, endpoint weight = 0.5)
    double So_c, So_s;   // γ sums (odd-index)
    double alpha_c;      //  gN*sin(wuN) - g0*sin(wu0)   → I_cos α term
    double alpha_s;      //  g0*cos(wu0) - gN*cos(wuN)   → I_sin α term
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
// N for one sub-interval: 64 panels per Bessel oscillation, floor=n_floor
// (default 8192, matching CPU Filon's FILON_N_MIN — see filon.h).
// Uses delta-ib = ib(b) - ib(a) so each segment is resolved independently.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int filon_N(double a, double b, double sign,
                                        double w, double Sqrt1mkk, double s,
                                        int n_floor)
{
    const double u2s_a = a * a + sign;
    const double ib_a  = (u2s_a > 0.) ? w * Sqrt1mkk * s * sqrt(u2s_a) : 0.;
    const double u2s_b = b * b + sign;
    const double ib_b  = (u2s_b > 0.) ? w * Sqrt1mkk * s * sqrt(u2s_b) : 0.;
    int N = max(n_floor, (int)(64.0 * (ib_b - ib_a) / (2.0 * M_PI)) + 2);
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
// Incremental streaming Filon u-integral — split at t_cut/s to eliminate the
// C² kink, dead zone removed (upper limit t_max/s, not 1+t_max/s), N from
// delta-ib, same as the original filon_u. The difference: the plateau piece
// [umin, u_split] doesn't depend on i_t (C1==1 there identically), so instead
// of re-running Filon over the whole growing plateau every kernel launch
// (time step), only the NEW slice since the previous launch's u_split
// (t_cut_prev/s) is integrated and added into the persisted cum_re/cum_im
// (references directly into this thread's slot of the device-resident
// d_cumbuf_ — see GpuIntegrator::d_cumbuf_). The transition window
// [u_split, u_top] has fixed width and just slides in u as t_cut advances,
// so it's still recomputed fresh every launch.
//
// Split into a zz-only and an xyz-only variant (rather than one templated
// function) since each needs a different number/shape of cum_re/cum_im
// arguments — mirrors the two distinct call shapes already used in
// gw_kernel (one at s for zz, one at s_off for xx/yy/xz).
// ---------------------------------------------------------------------------

__device__ void filon_zz_incremental(
    double s, double Sqrt1mkk, double w,
    double sign, double umin,
    double t_cut, double t_m, double t_0, double t_max,
    int cutoff_type, int n_floor, double t_cut_prev,
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
        int N = filon_N(plateau_prev, plateau_now, sign, w, Sqrt1mkk, s, n_floor);
        double d_r, d_i, dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i;
        filon_segment<true, false>(s, Sqrt1mkk, w, sign,
            plateau_prev, plateau_now, N,
            t_cut, t_m, t_0, t_max, cutoff_type,
            d_r, d_i, dum_r, dum_i, dum2_r, dum2_i, dum3_r, dum3_i);
        cum_re += d_r; cum_im += d_i;
    }

    double win_r = 0., win_i = 0.;
    if (u_top > plateau_now) {
        int N = filon_N(plateau_now, u_top, sign, w, Sqrt1mkk, s, n_floor);
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
    int cutoff_type, int n_floor, double t_cut_prev,
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
        int N = filon_N(plateau_prev, plateau_now, sign, w, Sqrt1mkk, s, n_floor);
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
        int N = filon_N(plateau_now, u_top, sign, w, Sqrt1mkk, s, n_floor);
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
// Main kernel — one thread per (i_w, i_k, i_s)
//
// Writes 6 linear per-s contributions to intbuf[i_w * n_k*n_s*6 + ...].
// CPU finalization reduces over i_s and computes the nonlinear |re+i*im|^2.
//
// The plateau portion of the u-integral is carried across kernel launches
// (time steps) in cumbuf — a persistent device buffer allocated once in
// GpuIntegrator's constructor, NOT reset between calls (unlike intbuf, which
// is scratch, memset every launch). Each thread owns a fixed 16-double slot
// at cumbuf[idx*16 .. idx*16+15]; see GpuIntegrator::d_cumbuf_ for the layout.
// ---------------------------------------------------------------------------

__global__ void gw_kernel(
    const double * __restrict__ phi,     // [n_s * n_z]
    const double * __restrict__ phi2,    // [n_s * n_z]
    int n_s, int n_z, int n_w, int n_k,
    double ds, double dz,
    const double * __restrict__ z_arr,   // [n_z]
    const double * __restrict__ w_arr,   // [n_w]
    const double * __restrict__ k_arr,   // [n_k]
    const double * __restrict__ s_arr,   // [n_s]
    double t_cut, double t_m, double t_0, double t_max,
    int cutoff_type, int n_floor, double t_cut_prev,
    double * __restrict__ intbuf,        // [n_w * n_k * n_s * 6]
    double * __restrict__ cumbuf         // [n_w * n_k * n_s * 16] — persists across launches
#ifdef GW_KERNEL_TIMING
    , long long * __restrict__ timebuf   // [n_w*n_k*n_s*5] per-thread cycle counts:
                                          // [u_integral, zz, xa, xz, tail] — diagnostic
                                          // build only, see Makefile's gpu_profile target.
#endif
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_w * n_k * n_s) return;

    int i_s = idx % n_s;
    int i_k = (idx / n_s) % n_k;
    int i_w = idx / (n_k * n_s);

    double s = s_arr[i_s];
    if (s == 0.) return;

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

    // This thread's persistent cum slot — references bind straight to global
    // memory, so += on these writes through immediately (no manual copy-back).
    double *c = cumbuf + idx * 16;
    double &cum_zz1_re = c[0],  &cum_zz1_im = c[1],  &cum_zz2_re = c[2],  &cum_zz2_im = c[3];
    double &cum_xx1_re = c[4],  &cum_xx1_im = c[5],  &cum_xx2_re = c[6],  &cum_xx2_im = c[7];
    double &cum_yy1_re = c[8],  &cum_yy1_im = c[9],  &cum_yy2_re = c[10], &cum_yy2_im = c[11];
    double &cum_xz1_re = c[12], &cum_xz1_im = c[13], &cum_xz2_re = c[14], &cum_xz2_im = c[15];

    // --- u-integrals (incremental — see filon_zz_incremental/filon_xyz_incremental) ---
    double zz_r1, zz_i1, zz_r2, zz_i2;
    filon_zz_incremental(s, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor, t_cut_prev,
                        cum_zz1_re, cum_zz1_im, zz_r1, zz_i1);
    filon_zz_incremental(s, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor, t_cut_prev,
                        cum_zz2_re, cum_zz2_im, zz_r2, zz_i2);

    double xx1, xi1, yy1, yi1, xz1, xzi1;
    double xx2, xi2, yy2, yi2, xz2, xzi2;
    filon_xyz_incremental(s_off, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor, t_cut_prev,
                        cum_xx1_re, cum_xx1_im, cum_yy1_re, cum_yy1_im, cum_xz1_re, cum_xz1_im,
                        xx1, xi1, yy1, yi1, xz1, xzi1);
    filon_xyz_incremental(s_off, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type, n_floor, t_cut_prev,
                        cum_xx2_re, cum_xx2_im, cum_yy2_re, cum_yy2_im, cum_xz2_re, cum_xz2_im,
                        xx2, xi2, yy2, yi2, xz2, xzi2);

#ifdef GW_KERNEL_TIMING
    long long t1 = clock64();
#endif

    // --- z-integrals ---
    // phi/phi2 layout is [iz * n_s + is] (transposed vs. the natural
    // [is][iz]) so that consecutive threads (consecutive i_s) hit
    // consecutive memory for a fixed iz -- coalesced. See d_phi_'s
    // declaration in the header for why.
    double iz1_zz = 0., iz2_zz = 0.;
    for (int iz = 1; iz < n_z; ++iz) {
        double zm   = z_arr[iz] - dz * 0.5;
        double czm  = cos(w * k * zm);
        double q1   = (phi [iz*n_s + i_s] - phi [(iz-1)*n_s + i_s]) / dz;
        double q2   = (phi2[iz*n_s + i_s] - phi2[(iz-1)*n_s + i_s]) / dz;
        iz1_zz += dz * 2.0 * czm * q1 * q1;
        iz2_zz += dz * 2.0 * czm * q2 * q2;
    }

#ifdef GW_KERNEL_TIMING
    long long t2 = clock64();
#endif

    double iz1_xa = 0., iz2_xa = 0.;
    for (int iz = 0; iz < n_z; ++iz) {
        double fz  = (iz == 0 || iz == n_z-1) ? 0.5 : 1.0;
        double cz  = cos(w * k * z_arr[iz]);
        double q1  = (phi [iz*n_s + i_s] - phi [iz*n_s + (i_s-1)]) / ds;
        double q2  = (phi2[iz*n_s + i_s] - phi2[iz*n_s + (i_s-1)]) / ds;
        iz1_xa += fz * dz * 2.0 * cz * q1 * q1;
        iz2_xa += fz * dz * 2.0 * cz * q2 * q2;
    }

#ifdef GW_KERNEL_TIMING
    long long t3 = clock64();
#endif

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

#ifdef GW_KERNEL_TIMING
    long long t4 = clock64();
#endif

    // --- Accumulate linear per-s contributions ---
    double fac    = (i_s == 0 || i_s == n_s-1) ? 0.5 : 1.;
    double pre_zz = fac * (double)i_s * (double)i_s * ds * ds * ds;
    double pre_xa = 0.5 * s_off * s_off * ds;
    double pre_xz = -s_off * s_off * ds;

    int base = (i_w * n_k * n_s + i_k * n_s + i_s) * 6;
    intbuf[base + 0] = pre_zz * (zz_r1*iz1_zz + zz_r2*iz2_zz);
    intbuf[base + 1] = pre_zz * (zz_i1*iz1_zz + zz_i2*iz2_zz);
    intbuf[base + 2] = pre_xa * ((xx1*k_sq - yy1)*iz1_xa + (xx2*k_sq - yy2)*iz2_xa);
    intbuf[base + 3] = pre_xa * ((xi1*k_sq - yi1)*iz1_xa + (xi2*k_sq - yi2)*iz2_xa);
    intbuf[base + 4] = pre_xz * (xz1*iz1_xz + xz2*iz2_xz);
    intbuf[base + 5] = pre_xz * (xzi1*iz1_xz + xzi2*iz2_xz);

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
// Constructor
// ---------------------------------------------------------------------------

GpuIntegrator::GpuIntegrator(const std::vector<std::vector<double>> &input_phi,
                              const Setup &setup, int param)
    : n_k_(setup.n_k), n_w_(setup.n_w),
      ds_(setup.ds * setup.how_often_ds),
      dz_(std::abs(setup.z[1] - setup.z[0])),
      t_cut_base_(setup.t_cut), t_m_base_(setup.t_m),
      t_max_base_(setup.t_max), t_0_(setup.t_0),
      d_(setup.d), cutoff_type_(setup.cutoff_type),
      z_(setup.z), wlist_(setup.wlist), times_(setup.times),
      param_floor_(param > 0 ? (param + 1) & ~1 : -1)
{
    n_z_ = z_.size();
    n_s_ = input_phi.size();
    slist_ = linspace(0., (n_s_ - 1) * ds_, static_cast<int>(n_s_));
    klist_ = linspace(0., 1., static_cast<int>(n_k_));

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GpuIntegrator: device=" << prop.name
              << "  n_w=" << n_w_ << " n_k=" << n_k_
              << " n_s=" << n_s_ << " n_z=" << n_z_
              << "\n  intbuf=" << (n_w_*n_k_*n_s_*6*8)/(1<<20) << " MB\n\n";

    // --- Flatten phi to host 1-D array, transposed to [iz * n_s + is] (see
    // d_phi_'s declaration in the header for why) ---
    std::vector<double> phi_host(n_s_ * n_z_);
    for (std::size_t is = 0; is < n_s_; ++is)
        for (std::size_t iz = 0; iz < n_z_; ++iz)
            phi_host[iz * n_s_ + is] = input_phi[is][iz];

    // --- Build phi2 on host ---
    std::vector<double> phi2_host;
    build_phi2(input_phi, phi2_host);

    // --- Upload to device ---
    auto alloc_and_copy = [&](double **dptr, const std::vector<double> &hv) {
        CUDA_CHECK(cudaMalloc(dptr, hv.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(*dptr, hv.data(),
                              hv.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    };

    alloc_and_copy(&d_phi_,  phi_host);
    alloc_and_copy(&d_phi2_, phi2_host);
    alloc_and_copy(&d_z_,    z_);
    alloc_and_copy(&d_w_,    wlist_);
    alloc_and_copy(&d_k_,    klist_);
    alloc_and_copy(&d_s_,    slist_);

    CUDA_CHECK(cudaMalloc(&d_intbuf_, n_w_ * n_k_ * n_s_ * 6 * sizeof(double)));

    // Persistent plateau-accumulator buffer: allocated once, zeroed once, and
    // NOT reset between Compute() calls (unlike d_intbuf_ above) — each
    // thread's slot carries the running plateau integral forward across time
    // steps. See gw_kernel's cumbuf usage and d_cumbuf_'s declaration.
    CUDA_CHECK(cudaMalloc(&d_cumbuf_, n_w_ * n_k_ * n_s_ * 16 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cumbuf_, 0, n_w_ * n_k_ * n_s_ * 16 * sizeof(double)));

#ifdef GW_KERNEL_TIMING
    CUDA_CHECK(cudaMalloc(&d_timebuf_, n_w_ * n_k_ * n_s_ * 5 * sizeof(long long)));
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
    cudaFree(d_intbuf_);
    cudaFree(d_cumbuf_);
#ifdef GW_KERNEL_TIMING
    cudaFree(d_timebuf_);
#endif
}

// ---------------------------------------------------------------------------
// Compute / ComputeAmplitude — thin wrappers over the shared launch+reduce
// path in RunAndReduce().
// ---------------------------------------------------------------------------

std::vector<double> GpuIntegrator::Compute(int i_t)
{
    return RunAndReduce(i_t, nullptr, nullptr);
}

AmplitudeResult GpuIntegrator::ComputeAmplitude(int i_t)
{
    AmplitudeResult res;
    res.w      = wlist_;
    res.klist  = klist_;
    res.amp_re.assign(n_w_ * n_k_, 0.);
    res.amp_im.assign(n_w_ * n_k_, 0.);
    res.spectrum = RunAndReduce(i_t, &res.amp_re, &res.amp_im);
    return res;
}

// ---------------------------------------------------------------------------
// RunAndReduce — launch kernel, then reduce on CPU
//
// The plateau portion of the u-integral is accumulated incrementally across
// calls in device memory (d_cumbuf_, allocated once in the constructor), so
// this mutates persisted state and must be called with strictly increasing
// i_t starting at 0.
// ---------------------------------------------------------------------------

std::vector<double> GpuIntegrator::RunAndReduce(int i_t,
                                                 std::vector<double> *out_amp_re,
                                                 std::vector<double> *out_amp_im)
{
    if (i_t != last_i_t_processed_ + 1) {
        throw std::runtime_error(
            "GpuIntegrator::RunAndReduce: i_t must be called in strictly "
            "increasing order starting at 0 (plateau u-integral is "
            "accumulated incrementally in device memory); got i_t=" +
            std::to_string(i_t) + " after last_i_t_processed_=" +
            std::to_string(last_i_t_processed_));
    }

    double shift  = t_m_base_  - times_[i_t];
    double t_cut  = t_cut_base_ - shift;
    double t_m    = t_m_base_   - shift;
    double t_max  = t_max_base_ - shift;
    const double t_cut_prev = t_cut_prev_;

    n_floor_ = param_floor_ > 0 ? param_floor_ : 8192;

    std::cout << "GpuIntegrator::RunAndReduce i_t=" << i_t
              << " shift=" << shift << " t_m=" << t_m
              << " n_floor=" << n_floor_
              << (out_amp_re ? "  (amplitude)" : "") << "\n";

    std::size_t total = n_w_ * n_k_ * n_s_;
    CUDA_CHECK(cudaMemset(d_intbuf_, 0, total * 6 * sizeof(double)));

    constexpr int BLOCK = 256;
    int grid = (static_cast<int>(total) + BLOCK - 1) / BLOCK;

    gw_kernel<<<grid, BLOCK>>>(
        d_phi_, d_phi2_,
        static_cast<int>(n_s_), static_cast<int>(n_z_),
        static_cast<int>(n_w_), static_cast<int>(n_k_),
        ds_, dz_,
        d_z_, d_w_, d_k_, d_s_,
        t_cut, t_m, t_0_, t_max,
        cutoff_type_, n_floor_, t_cut_prev,
        d_intbuf_, d_cumbuf_
#ifdef GW_KERNEL_TIMING
        , d_timebuf_
#endif
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef GW_KERNEL_TIMING
    // Diagnostic-build-only: sum clock64() cycles per phase across all
    // threads and print the relative breakdown. Since the z-integral loops
    // run a fixed n_z iterations regardless of thread, and the sum (rather
    // than max) is a standard proxy for "how much this phase matters" when
    // per-phase costs are reasonably uniform across threads, this is a rough
    // but honest proportional breakdown, not a precise wall-clock accounting.
    {
        std::vector<long long> tbuf(total * 5);
        CUDA_CHECK(cudaMemcpy(tbuf.data(), d_timebuf_,
                              tbuf.size() * sizeof(long long),
                              cudaMemcpyDeviceToHost));
        long long sums[5] = {0, 0, 0, 0, 0};
        for (std::size_t i = 0; i < total; ++i)
            for (int p = 0; p < 5; ++p)
                sums[p] += tbuf[i * 5 + p];
        long long grand_total = sums[0] + sums[1] + sums[2] + sums[3] + sums[4];
        static const char *names[5] = {"u_integral", "zz_loop", "xa_loop", "xz_loop", "tail"};
        std::cout << "  [timing breakdown, i_t=" << i_t << "]";
        if (grand_total > 0) {
            for (int p = 0; p < 5; ++p) {
                double pct = 100.0 * static_cast<double>(sums[p])
                                   / static_cast<double>(grand_total);
                std::cout << "  " << names[p] << "=" << pct << "%";
            }
        }
        std::cout << "\n";
    }
#endif

    // --- Copy intermediate buffer back and reduce on CPU ---
    std::vector<double> buf(total * 6);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_intbuf_,
                          buf.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

    const double dk = klist_[1] - klist_[0];

    std::vector<double> spectrum(n_w_, 0.);

    for (std::size_t iw = 0; iw < n_w_; ++iw) {
        double w = wlist_[iw];
        double int_k = 0.;

        for (std::size_t ik = 0; ik < n_k_; ++ik) {
            double k      = klist_[ik];
            double Onemkk = 1. - k*k;
            double TwokSq = 2. * k * std::sqrt(Onemkk);

            // Reduce over i_s for this (iw, ik)
            double szz_r = 0., szz_i = 0.;
            double sxa_r = 0., sxa_i = 0.;
            double sxz_r = 0., sxz_i = 0.;

            for (std::size_t is = 0; is < n_s_; ++is) {
                std::size_t base = (iw * n_k_ * n_s_ + ik * n_s_ + is) * 6;
                szz_r += buf[base + 0];
                szz_i += buf[base + 1];
                sxa_r += buf[base + 2];
                sxa_i += buf[base + 3];
                sxz_r += buf[base + 4];
                sxz_i += buf[base + 5];
            }

            double re = szz_r*Onemkk + sxa_r - TwokSq*sxz_r;
            double im = szz_i*Onemkk + sxa_i - TwokSq*sxz_i;
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

    t_cut_prev_ = t_cut;
    last_i_t_processed_ = i_t;
    return spectrum;
}

// ---------------------------------------------------------------------------
// phi2 construction (CPU — runs once in constructor)
// ---------------------------------------------------------------------------

void GpuIntegrator::build_phi2(const std::vector<std::vector<double>> &input_phi,
                                std::vector<double> &phi2_host) const
{
    auto max_it = std::max_element(input_phi[0].begin(), input_phi[0].end());
    int phimid  = static_cast<int>(max_it - input_phi[0].begin());

    std::vector<double> z_new(z_.begin() + phimid, z_.end());
    double z0 = z_new[0];
    for (double &zi : z_new) zi -= z0;

    std::vector<double> phi0_new(input_phi[0].begin() + phimid,
                                 input_phi[0].end());
    Interpolator phi0_interp(z_new, phi0_new);

    // Transposed to [iz * n_s + is] to match d_phi_'s layout (see header).
    phi2_host.resize(n_s_ * n_z_);
    for (std::size_t is = 0; is < n_s_; ++is) {
        double s_val = is * ds_;
        for (std::size_t iz = 0; iz < n_z_; ++iz) {
            double z_val = iz * dz_;
            double r1 = std::sqrt(s_val*s_val + (z_val - d_/2.)*(z_val - d_/2.));
            double r2 = std::sqrt(s_val*s_val + (z_val + d_/2.)*(z_val + d_/2.));
            phi2_host[iz * n_s_ + is] = phi0_interp(r1) + phi0_interp(r2);
        }
    }
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
