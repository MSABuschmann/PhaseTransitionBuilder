#include "gpu_integrator.cuh"
#include "interpolator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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
// Streaming Filon u-integral — templated to compute only the needed
// stress-tensor components and avoid dead arithmetic in the main kernel.
//
//   NEED_ZZ  = true for calls at time s (zz component feeds the zz z-integral)
//   NEED_XYZ = true for calls at time s_off (xx, yy, xz feed the other z-integrals)
// ---------------------------------------------------------------------------

template <bool NEED_ZZ, bool NEED_XYZ>
__device__ void filon_u(double s, double Sqrt1mkk, double w,
                         double sign, double umin,
                         double t_cut, double t_m, double t_0, double t_max,
                         int cutoff_type,
                         double &zz_r, double &zz_i,
                         double &xx_r, double &xx_i,
                         double &yy_r, double &yy_i,
                         double &xz_r, double &xz_i)
{
    const double umax   = 1.0 + t_max / s;
    const double u2s_mx = umax*umax + sign;
    const double ib_max = (u2s_mx > 0.) ? w * Sqrt1mkk * s * sqrt(u2s_mx) : 0.;
    int N = max(2048, (int)(8.0 * ib_max / (2.0 * M_PI)) + 2);
    N = (N + 1) & ~1;   // round up to even

    const double h     = (umax - umin) / N;
    const double omega = w * s;

    FStream fzz = {}, fxx = {}, fyy = {}, fxz = {};

    for (int i = 0; i <= N; ++i) {
        double u       = umin + i * h;
        double u2s     = u*u + sign;
        double u2s_pos = fmax(0., u2s);
        double ib      = w * Sqrt1mkk * s * sqrt(u2s_pos);
        double c1      = d_C1(s*u, t_cut, t_m, t_0, t_max, cutoff_type);

        double bj0 = d_j0(ib), bj1 = d_j1(ib);
        double bj0m2, bj0p2;
        if (ib < 1e-14) {
            bj0m2 = 1.0; bj0p2 = 1.0;
        } else {
            double toi = 2.0 / ib;
            bj0m2 = 2.0*bj0 - toi*bj1;
            bj0p2 = toi*bj1;
        }

        double wu = omega * u;
        double cw = cos(wu), sw = sin(wu);

        if (NEED_ZZ)  fs_update(fzz, i, N, bj0 * c1,          cw, sw);
        if (NEED_XYZ) {
            fs_update(fxx, i, N, u2s * bj0m2 * c1,            cw, sw);
            fs_update(fyy, i, N, u2s * bj0p2 * c1,            cw, sw);
            fs_update(fxz, i, N, sign*sqrt(u2s_pos)*bj1*c1,   cw, sw);
        }
    }

    double alpha, beta, gamma_f;
    d_filon_coeffs(omega * h, alpha, beta, gamma_f);

    if (NEED_ZZ)  { fs_result(fzz, alpha, beta, gamma_f, h, zz_r, zz_i); }
    if (NEED_XYZ) {
        fs_result(fxx, alpha, beta, gamma_f, h, xx_r, xx_i);
        fs_result(fyy, alpha, beta, gamma_f, h, yy_r, yy_i);
        fs_result(fxz, alpha, beta, gamma_f, h, xz_r, xz_i);
    }
}

// ---------------------------------------------------------------------------
// Main kernel — one thread per (i_w, i_k, i_s)
//
// Writes 6 linear per-s contributions to intbuf[i_w * n_k*n_s*6 + ...].
// CPU finalization reduces over i_s and computes the nonlinear |re+i*im|^2.
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
    int cutoff_type,
    double * __restrict__ intbuf         // [n_w * n_k * n_s * 6]
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

    double k_sq     = k * k;
    double Onemkk   = 1. - k_sq;
    double Sqrt1mkk = sqrt(Onemkk);
    double s_off    = s - 0.5 * ds;

    // --- u-integrals ---
    double zz_r1, zz_i1, zz_r2, zz_i2;
    double _u, _v;   // placeholders for unused outputs
    filon_u<true,false>(s, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type,
                        zz_r1, zz_i1, _u, _v, _u, _v, _u, _v);
    filon_u<true,false>(s, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type,
                        zz_r2, zz_i2, _u, _v, _u, _v, _u, _v);

    double xx1, xi1, yy1, yi1, xz1, xzi1;
    double xx2, xi2, yy2, yi2, xz2, xzi2;
    filon_u<false,true>(s_off, Sqrt1mkk, w, -1., 1.,
                        t_cut, t_m, t_0, t_max, cutoff_type,
                        _u, _v, xx1, xi1, yy1, yi1, xz1, xzi1);
    filon_u<false,true>(s_off, Sqrt1mkk, w, +1., 0.,
                        t_cut, t_m, t_0, t_max, cutoff_type,
                        _u, _v, xx2, xi2, yy2, yi2, xz2, xzi2);

    // --- z-integrals ---
    double iz1_zz = 0., iz2_zz = 0.;
    for (int iz = 1; iz < n_z; ++iz) {
        double zm   = z_arr[iz] - dz * 0.5;
        double czm  = cos(w * k * zm);
        double q1   = (phi [i_s*n_z + iz] - phi [i_s*n_z + iz-1]) / dz;
        double q2   = (phi2[i_s*n_z + iz] - phi2[i_s*n_z + iz-1]) / dz;
        iz1_zz += dz * 2.0 * czm * q1 * q1;
        iz2_zz += dz * 2.0 * czm * q2 * q2;
    }

    double iz1_xa = 0., iz2_xa = 0.;
    for (int iz = 0; iz < n_z; ++iz) {
        double fz  = (iz == 0 || iz == n_z-1) ? 0.5 : 1.0;
        double cz  = cos(w * k * z_arr[iz]);
        double q1  = (phi [i_s*n_z + iz] - phi [(i_s-1)*n_z + iz]) / ds;
        double q2  = (phi2[i_s*n_z + iz] - phi2[(i_s-1)*n_z + iz]) / ds;
        iz1_xa += fz * dz * 2.0 * cz * q1 * q1;
        iz2_xa += fz * dz * 2.0 * cz * q2 * q2;
    }

    double iz1_xz = 0., iz2_xz = 0.;
    for (int iz = 1; iz < n_z; ++iz) {
        double szm  = sin(w * k * (z_arr[iz] - dz * 0.5));
        double ds1  = 0.5/ds * (phi [i_s*n_z+iz]   - phi [(i_s-1)*n_z+iz]
                               + phi [i_s*n_z+iz-1] - phi [(i_s-1)*n_z+iz-1]);
        double dz1  = 0.5/dz * (phi [i_s*n_z+iz]   - phi [i_s*n_z+iz-1]
                               + phi [(i_s-1)*n_z+iz] - phi [(i_s-1)*n_z+iz-1]);
        double ds2  = 0.5/ds * (phi2[i_s*n_z+iz]   - phi2[(i_s-1)*n_z+iz]
                               + phi2[i_s*n_z+iz-1] - phi2[(i_s-1)*n_z+iz-1]);
        double dz2  = 0.5/dz * (phi2[i_s*n_z+iz]   - phi2[i_s*n_z+iz-1]
                               + phi2[(i_s-1)*n_z+iz] - phi2[(i_s-1)*n_z+iz-1]);
        iz1_xz += dz * 2.0 * szm * ds1 * dz1;
        iz2_xz += dz * 2.0 * szm * ds2 * dz2;
    }

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
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GpuIntegrator::GpuIntegrator(const std::vector<std::vector<double>> &input_phi,
                              const Setup &setup, int /*param*/)
    : n_k_(setup.n_k), n_w_(setup.n_w),
      ds_(setup.ds * setup.how_often_ds),
      dz_(std::abs(setup.z[1] - setup.z[0])),
      t_cut_base_(setup.t_cut), t_m_base_(setup.t_m),
      t_max_base_(setup.t_max), t_0_(setup.t_0),
      d_(setup.d), cutoff_type_(setup.cutoff_type),
      z_(setup.z), wlist_(setup.wlist), times_(setup.times)
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

    // --- Flatten phi to host 1-D array ---
    std::vector<double> phi_host(n_s_ * n_z_);
    for (std::size_t is = 0; is < n_s_; ++is)
        for (std::size_t iz = 0; iz < n_z_; ++iz)
            phi_host[is * n_z_ + iz] = input_phi[is][iz];

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
}

// ---------------------------------------------------------------------------
// Compute — launch kernel, then reduce on CPU
// ---------------------------------------------------------------------------

std::vector<double> GpuIntegrator::Compute(int i_t) const
{
    double shift  = t_m_base_  - times_[i_t];
    double t_cut  = t_cut_base_ - shift;
    double t_m    = t_m_base_   - shift;
    double t_max  = t_max_base_ - shift;

    std::cout << "GpuIntegrator::Compute i_t=" << i_t
              << " shift=" << shift << " t_m=" << t_m << "\n";

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
        cutoff_type_,
        d_intbuf_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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
            double intk_val = (re*re + im*im) * w*w*w * 2.*M_PI;

            double fk = (ik == 0 || ik == n_k_-1) ? 1. : 2.;
            int_k += intk_val * dk * fk;
        }

        spectrum[iw] = int_k;
        std::cout << "  w=" << w << " spec=" << int_k << "\n";
    }

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

    phi2_host.resize(n_s_ * n_z_);
    for (std::size_t is = 0; is < n_s_; ++is) {
        double s_val = is * ds_;
        for (std::size_t iz = 0; iz < n_z_; ++iz) {
            double z_val = iz * dz_;
            double r1 = std::sqrt(s_val*s_val + (z_val - d_/2.)*(z_val - d_/2.));
            double r2 = std::sqrt(s_val*s_val + (z_val + d_/2.)*(z_val + d_/2.));
            phi2_host[is * n_z_ + iz] = phi0_interp(r1) + phi0_interp(r2);
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
