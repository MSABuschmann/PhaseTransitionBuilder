"""Tests for ptbuilder.patch_fraction (python -m unittest -v tests.test_patch_fraction)."""
from pathlib import Path
import unittest
import numpy as np
from ptbuilder.patch_fraction import (response, load_response, effective_weights,
                                       pair_separation, DEFAULT_CALIBRATION)


class ReferenceTests(unittest.TestCase):
    def test_interpolate_then_square(self):
        fn = response([0, 1], [1, .5], alpha=2)
        self.assertEqual(float(fn(.5)), 9/16)
        self.assertNotEqual(float(fn(.5)), 5/8)

    def test_rebounds_and_tail(self):
        f = response([0, 1, 2, 3], [1, .3, .5, .2])
        self.assertGreater(f(2), f(1))
        self.assertEqual(float(f(3)), .2**2)
        self.assertEqual(float(f(3 + 1e-8)), 0)
        g = response([0, 1, 2, 3], [1, .3, .5, .2], tail="terminal")
        self.assertEqual(float(g(10)), .2**2)

    def test_actual_pair_distance(self):
        f = response([0, 1, 2], [1, .5, .25], alpha=1)
        got = effective_weights([1, 0, 0], [0, 1, 2], 2, f)
        np.testing.assert_array_equal(got, [1, 1, .75])

    def test_constant_active_weights(self):
        t = np.arange(5.)
        f = response([0, 1, 2], [1, .5, .25], alpha=2)
        for value in [0, .3, 1]:
            w = np.full(len(t), value)
            np.testing.assert_array_equal(effective_weights(w, t, 1, f), w)

    def test_disabled_is_exact_baseline(self):
        w = np.array([0, 1, .5, 1, 0.])
        np.testing.assert_array_equal(effective_weights(w, np.arange(5.), 1, None), w)

    def test_reactivation_fixture_and_partial_times(self):
        t = np.arange(8.)
        w = [0, 1, .5, 1, 0, 0, 0, 0]
        for alpha, expected, qhalf, qlast in [
            (1, [0, 1, 1, 1, 1, .5, .25, 0], 4.25, 4.75),
            (2, [0, 1, 1, 1, 1, .25, .0625, 0], 4.125, 4.3125),
        ]:
            f = response([0, 1, 2], [1, .5, .25], alpha)
            got = effective_weights(w, t, 1, f)
            np.testing.assert_array_equal(got, expected)
            g = response([0, 1, 2], [1, .5, .25], alpha, "terminal")
            terminal = effective_weights(w, t, 1, g)
            self.assertEqual(terminal[-1], .25**alpha)

    def test_signed_increments_in_reconstruction(self):
        # reconstruct_pair_spectrum keeps signed left-endpoint increments
        from ptbuilder.analysis import reconstruct_pair_spectrum
        interp = lambda pts: np.interp(pts[:, 1], [0, 1, 2], [0, 2, 1])[:, None]
        got = reconstruct_pair_spectrum(4., np.array([1, .5, .5]), np.array([0., 1, 2]), 2.,
                                        interp, np.array([0., 2.]), np.array([4.]))
        self.assertEqual(float(got[0]), 1.5)

    def test_input_rejection(self):
        for x in ([0, np.nan, 2], [0, 1, 1]):
            with self.assertRaises(ValueError): response(x, [1, .5, .25])
        with self.assertRaises(ValueError): response([0, 1], [1, 1.1])
        with self.assertRaises(ValueError): effective_weights([0, 1], [0, np.nan], 1, None)
        with self.assertRaises(ValueError): effective_weights([0, 1], [0, 1], 0, None)

    def test_refined_grid_extension(self):
        from ptbuilder.patch_fraction import refined_grid
        np.testing.assert_allclose(refined_grid([0, 1, 2], 10, 2), np.arange(0, 10.01, .5))
        # endpoint between regular points: included exactly once
        np.testing.assert_allclose(refined_grid([0, 1, 2], 3.3, 1), [0, 1, 2, 3, 3.3])
        np.testing.assert_allclose(refined_grid([0, 1, 2], 3.3, 2), [0, .5, 1, 1.5, 2, 2.5, 3, 3.15, 3.3])
        # truncation inside the original grid is unchanged
        np.testing.assert_allclose(refined_grid([0, 1, 2, 3], 1.5, 2), [0, .5, 1, 1.25, 1.5])
        np.testing.assert_allclose(refined_grid([0, 1, 2, 3], 3, 1), [0, 1, 2, 3])
        # nonuniform grids need an explicit spacing
        with self.assertRaises(ValueError): refined_grid([0, 1, 3], 5, 1)
        np.testing.assert_allclose(refined_grid([0, 1, 3], 5, 1, extension_spacing=1), [0, 1, 3, 4, 5])

    def test_extension_requires_zero_terminal_weights(self):
        from ptbuilder.patch_fraction import damped_weights
        from types import SimpleNamespace
        prof = SimpleNamespace(rin_0=1., rout_0=2., rmid_0=1.5)
        f = response([0, 1, 2], [1, .5, .25], alpha=1)
        t = np.array([0., 1, 2])
        with self.assertRaises(ValueError):
            damped_weights(np.array([[0, 1, .5]]), t, [3.], prof, f, t_end=5.)
        w, tg = damped_weights(np.array([[0, 1, 0.]]), t, [3.], prof, f, t_end=5.)
        np.testing.assert_allclose(tg, [0, 1, 2, 3, 4, 5])

    def test_measured_data_contract(self):
        for lb in [.84, .069]:
            for geom in ["fixed_annulus", "fixed_rapidity"]:
                fn = load_response(lb, geom)
                self.assertEqual(float(fn(0)), 1.)
                self.assertEqual(float(fn(100)), 0.)
        with self.assertRaises(ValueError): load_response(.5)


if __name__ == "__main__": unittest.main()
