"""
Compares the C++ `coherent` binary's output against a Python reference
saved from notebook 06's `coherent_sum_n64` at a small, fast-to-compute
t_max. See notebooks/06_n64_reconstruction.ipynb for how to save the
reference (a few lines added at the end of Section 6, saving w_amp,
P_coh, t_max, and the active-pair count to an .npz via np.savez).

Usage:
    python scripts/validate_coherent_cpp.py <python_ref.npz> <cpp_out.h5>
"""
import sys
import h5py
import numpy as np

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    ref = np.load(sys.argv[1])
    with h5py.File(sys.argv[2]) as f:
        w_cpp = f['w'][:]
        n_t_max = int(f.attrs['n_t_max'])
        n_w = int(f.attrs['n_w'])
        P_cpp_all = f['P_coh'][:].reshape(n_t_max, n_w)
        n_active_cpp = f['n_active'][:]
        t_max_cpp = f['t_max'][:]

    t_max_ref = float(ref['t_max'])
    row = int(np.argmin(np.abs(t_max_cpp - t_max_ref)))
    if abs(t_max_cpp[row] - t_max_ref) > 1e-6:
        print(f"WARNING: no exact t_max match in cpp output for {t_max_ref} "
              f"(closest: {t_max_cpp[row]})")

    P_cpp = P_cpp_all[row]
    P_ref = ref['P_coh']
    n_active_ref = int(ref['n_active'])

    print(f"t_max: ref={t_max_ref}  cpp={t_max_cpp[row]}")
    print(f"n_active: ref={n_active_ref}  cpp={int(n_active_cpp[row])}")
    if n_active_ref != int(n_active_cpp[row]):
        print("FAIL: active-pair count mismatch -- check masking/threshold "
              "logic before looking at P_coh at all.")
        sys.exit(1)

    if not np.allclose(w_cpp, ref['w']):
        print("FAIL: w grid mismatch between python reference and cpp output.")
        sys.exit(1)

    ok = np.allclose(P_cpp, P_ref, rtol=1e-8, atol=P_ref.max() * 1e-12)
    max_rel = np.max(np.abs(P_cpp - P_ref) / np.maximum(np.abs(P_ref), 1e-300))
    print(f"P_coh allclose(rtol=1e-8): {ok}   max relative diff: {max_rel:.3e}")
    print("P_coh ref:", P_ref)
    print("P_coh cpp:", P_cpp)

    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
