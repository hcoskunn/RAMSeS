"""
Standalone unit tests for the Monte Carlo robustness-explainability layer.
Mocks the module's heavy imports (Metrics.metrics, Utils.model_selection_utils,
loguru) and loads Monte_Carlo_Simulation.py by file path. numpy / scipy /
matplotlib are real; sklearn is intentionally NOT mocked so the surrogate
integration test can `importorskip` it (runs where sklearn is installed).
"""

import os
import sys
import types
import importlib.util
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")


# ── Mock the module's heavy imports ─────────────────────────────────────────
def _mk(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


for _n in ("Metrics", "Metrics.metrics", "Utils", "Utils.model_selection_utils", "loguru"):
    _mk(_n)
sys.modules["Metrics.metrics"].range_based_precision_recall_f1_auc = lambda *a, **k: (0, 0, 0.5, 0.5, None)
sys.modules["Metrics.metrics"].prauc = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].f1_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].f1_soft_score = lambda *a, **k: (0.5,) * 7
sys.modules["Utils.model_selection_utils"].evaluate_model = lambda *a, **k: {}


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **k: None


sys.modules["loguru"].logger = _Logger()

# ── Load Monte_Carlo_Simulation.py by path ──────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "mc_sim", os.path.join(_THIS, "Monte_Carlo_Simulation.py"))
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _fake_data(n=60, two_class=True):
    labels = np.zeros((1, n))
    if two_class:
        labels[0, :max(1, n // 6)] = 1
    ent = types.SimpleNamespace(labels=labels, Y=np.zeros((3, n)))
    return types.SimpleNamespace(entities=[ent])


# ════════════════════════════════════════════════════════════════════════════
# 1.  monte_carlo_noise_sweep
# ════════════════════════════════════════════════════════════════════════════

class TestSweep(unittest.TestCase):

    def test_shapes_and_grid(self):
        models = ["A", "B"]
        levels = np.linspace(0.0, 0.5, 6)

        def ev(m, level):
            return (0.9 - level if m == "A" else 0.2 + level, 0.5)

        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, models,
                                         noise_levels=levels, repeats=4, evaluate_fn=ev)
        self.assertIsNotNone(out)
        self.assertEqual(out["F1"].shape, (6 * 4, 2))
        self.assertEqual(out["noise"].shape, (6 * 4,))
        self.assertTrue(np.allclose(np.unique(out["noise"]), levels))

    def test_infeasible_returns_none(self):
        # too small
        self.assertIsNone(mc.monte_carlo_noise_sweep(_fake_data(n=20), {}, ["A"],
                                                     evaluate_fn=lambda m, l: (0.5, 0.5)))
        # single class
        self.assertIsNone(mc.monte_carlo_noise_sweep(_fake_data(two_class=False), {}, ["A"],
                                                     evaluate_fn=lambda m, l: (0.5, 0.5)))

    def test_nan_on_model_failure(self):
        def ev(m, level):
            if m == "bad":
                raise RuntimeError("boom")
            return (0.7, 0.6)
        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, ["good", "bad"],
                                         noise_levels=[0.0, 0.1], repeats=2, evaluate_fn=ev)
        self.assertTrue(np.all(np.isnan(out["F1"][:, 1])))      # bad column all NaN
        self.assertFalse(np.any(np.isnan(out["F1"][:, 0])))     # good column finite


# ════════════════════════════════════════════════════════════════════════════
# 2.  compute_noise_curves
# ════════════════════════════════════════════════════════════════════════════

class TestNoiseCurves(unittest.TestCase):

    def test_crossover_winregions_breakdown(self):
        models = ["A", "B"]
        grid = np.array([0.0, 0.2, 0.4, 0.6])
        # A leads at low noise then collapses below 0.5; B stays robust (always ≥0.5).
        # scores: A = [0.9, 0.7, 0.3, 0.1] ; B = [0.55, 0.65, 0.6, 0.8]
        noise = grid.copy()
        score = np.array([[0.9, 0.55], [0.7, 0.65], [0.3, 0.6], [0.1, 0.8]])
        cur = mc.compute_noise_curves(noise, grid, score, models, breakdown_threshold=0.5)
        # winner flips A→B between 0.2 and 0.4.
        self.assertEqual(len(cur["crossovers"]), 1)
        self.assertEqual(cur["crossovers"][0]["from_model"], "A")
        self.assertEqual(cur["crossovers"][0]["to_model"], "B")
        self.assertAlmostEqual(cur["crossovers"][0]["noise"], 0.4)
        # A's win-region covers the low-noise band.
        self.assertEqual(cur["win_regions"]["A"], [(0.0, 0.2)])
        # A breaks down (first mean <0.5) at 0.4; B never.
        self.assertAlmostEqual(cur["breakdown_points"]["A"], 0.4)
        self.assertIsNone(cur["breakdown_points"]["B"])


# ════════════════════════════════════════════════════════════════════════════
# 3.  compute_ranking_stability
# ════════════════════════════════════════════════════════════════════════════

class TestRankingStability(unittest.TestCase):

    def test_constant_ranking_tau_one(self):
        models = ["A", "B", "C"]
        grid = np.array([0.0, 0.3])
        # A>B>C at every level → τ ≈ 1 everywhere.
        noise = np.array([0.0, 0.3])
        score = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.05]])
        st = mc.compute_ranking_stability(noise, grid, score, models)
        self.assertTrue(np.all(st["tau_per_level"] > 0.99))

    def test_reversed_ranking_tau_drops(self):
        models = ["A", "B", "C"]
        grid = np.array([0.0, 0.3])
        # low noise A>B>C ; high noise C>B>A (full reversal) → τ negative at high.
        noise = np.array([0.0, 0.3])
        score = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
        st = mc.compute_ranking_stability(noise, grid, score, models)
        self.assertLess(st["tau_per_level"][1], 0.0)


# ════════════════════════════════════════════════════════════════════════════
# 4.  winner-label / single-class surrogate path (no sklearn)
# ════════════════════════════════════════════════════════════════════════════

class TestWinnerLabelSingleClass(unittest.TestCase):

    def test_single_winner_no_sklearn(self):
        # A wins every row → single-class path returns without importing sklearn.
        models = ["A", "B"]
        noise = np.array([0.0, 0.1, 0.2, 0.3])
        score = np.array([[0.9, 0.1]] * 4)
        info = mc.train_noise_winner_surrogate(noise, score, models)
        self.assertTrue(info["feasible"])
        self.assertEqual(info["classes"], ["A"])
        self.assertEqual(info["win_rates"]["A"], 1.0)
        self.assertIn("Always A", info["rules_text"])


# ════════════════════════════════════════════════════════════════════════════
# 5.  Integration — needs sklearn (skips where mocked/missing)
# ════════════════════════════════════════════════════════════════════════════

class TestExplainMonteCarloIntegration(unittest.TestCase):

    @staticmethod
    def _flip_eval(m, level):
        # A leads below s≈0.35, B above (0.9−s == 0.2+s ⇒ s = 0.35); same for F1 & PR.
        s = (0.9 - level) if m == "A" else (0.2 + level)
        return (s, s)

    def test_winner_surrogate_threshold(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        models = ["A", "B"]
        sweep = mc.monte_carlo_noise_sweep(_fake_data(), {}, models,
                                           noise_levels=np.linspace(0.0, 0.5, 15),
                                           repeats=6, evaluate_fn=self._flip_eval)
        info = mc.train_noise_winner_surrogate(sweep["noise"], sweep["F1"], models)
        self.assertTrue(info["feasible"])
        self.assertGreater(info["train_accuracy"], 0.9)
        self.assertIsNotNone(info["root_threshold"])
        self.assertTrue(0.25 < info["root_threshold"] < 0.45)

    def test_orchestrator_writes_report_and_plots(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        models = ["A", "B"]
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                res = mc.explain_monte_carlo(
                    _fake_data(), {"A": object(), "B": object()}, models,
                    "TEST", "e1", noise_levels=np.linspace(0.0, 0.5, 15), repeats=5,
                    explain=True, evaluate_fn=self._flip_eval,
                )
                self.assertIsInstance(res, dict)
                for k in ("sweep", "curves_f1", "stability_f1", "winner_f1",
                          "permodel_f1", "n_trials"):
                    self.assertIn(k, res)
                # A→B crossover should be detected on the F1 curves.
                self.assertGreaterEqual(len(res["curves_f1"]["crossovers"]), 1)
                out = os.path.join("myresults", "robustness", "MonteCarlo", "TEST", "e1")
                for fname in (
                    "TEST_e1_MonteCarlo_explainability.txt",
                    "TEST_e1_MonteCarlo_noise_curves_F1.png",
                    "TEST_e1_MonteCarlo_noise_curves_PRAUC.png",
                    "TEST_e1_MonteCarlo_noise_curves_F1_plain.png",
                    "TEST_e1_MonteCarlo_noise_curves_PRAUC_plain.png",
                    "TEST_e1_MonteCarlo_ranking_stability.png",
                    "TEST_e1_MonteCarlo_surrogate_tree_F1.png",
                ):
                    self.assertTrue(os.path.exists(os.path.join(out, fname)), fname)
            finally:
                os.chdir(cwd)

    def test_explain_false_returns_none(self):
        self.assertIsNone(mc.explain_monte_carlo(
            _fake_data(), {}, ["A"], "X", "Y", explain=False))


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
