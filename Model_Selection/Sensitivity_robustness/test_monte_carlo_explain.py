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
# See test_off_by_explain: `Utils.pipeline_spec` is stdlib-only and is wanted for
# real, so the stand-in gets a __path__ while the mocked sibling still shadows.
sys.modules["Utils"].__path__ = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Utils")]
sys.modules["Metrics.metrics"].range_based_precision_recall_f1_auc = lambda *a, **k: (0, 0, 0.5, 0.5, None)
sys.modules["Metrics.metrics"].prauc = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].f1_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].f1_soft_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].rank_key = lambda v: v
sys.modules["Metrics.metrics"].vus_score = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].vus_window = lambda *a, **k: 8
sys.modules["Utils.model_selection_utils"].evaluate_model = lambda *a, **k: {}
sys.modules["Utils.model_selection_utils"].ScoringTimeout = type(
    "ScoringTimeout", (Exception,), {})


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


class TestProductionRanking(unittest.TestCase):
    """The production summary publishes ONE ranking, by the run's own fitness."""

    @staticmethod
    def _results(vus):
        return {
            "A": {"f1_scores": [0.9], "pr_auc_scores": [0.5], "vus_scores": vus and [0.2]},
            "B": {"f1_scores": [0.5], "pr_auc_scores": [0.9], "vus_scores": vus and [0.8]},
        }

    def test_the_fitness_decides_the_order(self):
        """A and B tie on 0.5*F1 + 0.5*PR-AUC, and VUS is what separates them,
        so a fitness that names VUS must reorder them."""
        by_f1 = mc.summarize_results(self._results(True), metrics=("f1",))
        self.assertEqual(by_f1["ranked"], ["A", "B"])
        by_vus = mc.summarize_results(self._results(True), metrics=("vus",))
        self.assertEqual(by_vus["ranked"], ["B", "A"])

    def test_weights_move_the_order(self):
        """The whole point of the collapse: a weighted fitness ranks by those
        weights instead of publishing one ranking per metric."""
        r = self._results(False)
        self.assertEqual(
            mc.summarize_results(r, metrics={"f1": 0.9, "pr_auc": 0.1})["ranked"],
            ["A", "B"])
        self.assertEqual(
            mc.summarize_results(r, metrics={"f1": 0.1, "pr_auc": 0.9})["ranked"],
            ["B", "A"])

    def test_an_uncomputable_term_sorts_last(self):
        """renormalise=False: C is not scored on F1 and PR-AUC alone and then
        compared against detectors that also carried a VUS term."""
        results = self._results(True)
        results["C"] = {"f1_scores": [0.99], "pr_auc_scores": [0.99],
                        "vus_scores": [float("nan")]}
        s = mc.summarize_results(results, metrics=("f1", "pr_auc", "vus"))
        self.assertEqual(s["ranked"][-1], "C")
        self.assertTrue(np.isnan(s["C"]["fitness"]))



def _trials(matrix, names, vus=None):
    """Trial x model score lists in the shape monte_carlo_simulation produces."""
    matrix = np.asarray(matrix, dtype=float)
    return {n: {"f1_scores": list(matrix[:, j]),
                "pr_auc_scores": list(matrix[:, j]),
                "vus_scores": ([] if vus is None else list(np.asarray(vus)[:, j]))}
            for j, n in enumerate(names)}


# A clear winner, a block of three that trade places, and a clear loser.
_CONTESTED = [[0.90, 0.60, 0.55, 0.50, 0.10],
              [0.91, 0.50, 0.60, 0.55, 0.11],
              [0.92, 0.55, 0.50, 0.60, 0.09],
              [0.90, 0.60, 0.55, 0.50, 0.10],
              [0.91, 0.50, 0.60, 0.55, 0.12]]
_NAMES = ["A", "B", "C", "D", "E"]


# ════════════════════════════════════════════════════════════════════════════
# 1.  The trial matrices
# ════════════════════════════════════════════════════════════════════════════

class TestTrialMatrices(unittest.TestCase):

    def test_matrices_are_trials_by_models(self):
        pt = mc.build_trial_matrices(_trials(_CONTESTED, _NAMES), 5)
        self.assertEqual(pt["FIT"].shape, (5, 5))
        self.assertEqual(pt["model_names"], _NAMES)
        self.assertEqual(pt["n_trials"], 5)

    def test_a_detector_that_timed_out_is_left_out(self):
        """A timeout pops the model wholesale, so its list is short and the
        columns would stop lining up if it were kept."""
        results = _trials(_CONTESTED, _NAMES)
        results["C"]["f1_scores"] = [0.5, 0.5]
        pt = mc.build_trial_matrices(results, 5)
        self.assertNotIn("C", pt["model_names"])
        self.assertEqual(pt["FIT"].shape, (5, 4))

    def test_the_fitness_cell_uses_the_runs_weights(self):
        f1 = np.full((2, 1), 0.8)
        pr = np.full((2, 1), 0.4)
        results = {"A": {"f1_scores": list(f1[:, 0]),
                         "pr_auc_scores": list(pr[:, 0]), "vus_scores": []}}
        pt = mc.build_trial_matrices(results, 2, {"f1": 0.75, "pr_auc": 0.25})
        self.assertAlmostEqual(pt["FIT"][0, 0], 0.75 * 0.8 + 0.25 * 0.4, places=6)

    def test_no_usable_trials_gives_nothing(self):
        self.assertEqual(mc.build_trial_matrices({}, 5), {})


# ════════════════════════════════════════════════════════════════════════════
# 2.  Per-trial ranking, spread, leave-one-out
# ════════════════════════════════════════════════════════════════════════════

class TestTrialSummary(unittest.TestCase):

    @staticmethod
    def _summary(matrix=None, names=None, ranked=None):
        matrix = _CONTESTED if matrix is None else matrix
        names = _NAMES if names is None else names
        pt = mc.build_trial_matrices(_trials(matrix, names), len(matrix))
        return mc.summarize_trials(pt, ranked)

    def test_each_trial_is_ranked_on_its_own(self):
        s = self._summary()
        ranks = s["ranks"]
        self.assertEqual(ranks.shape, (5, 5))
        # A is best in every trial, E worst in every trial.
        self.assertTrue(all(ranks[i, 0] == 1 for i in range(5)))
        self.assertTrue(all(ranks[i, 4] == 5 for i in range(5)))
        # The contested block really does change order.
        self.assertGreater(len(set(tuple(r[1:4]) for r in ranks)), 1)

    def test_wins_count_the_trials_not_the_average(self):
        s = self._summary()
        self.assertEqual(s["wins"]["A"], 5)
        self.assertEqual(sum(s["wins"].values()), 5)

    def test_rank_range_reports_best_and_worst_placing(self):
        s = self._summary()
        self.assertEqual(s["rank_ranges"]["A"], (1, 1))
        self.assertEqual(s["rank_ranges"]["B"], (2, 4))

    def test_defeats_name_the_detector_that_led_each_lost_trial(self):
        """Not the runner-up by mean: B is second overall but never tops a
        trial, while C and D each take one off the winner."""
        matrix = [[0.90, 0.80, 0.30, 0.30],
                  [0.90, 0.80, 0.30, 0.30],
                  [0.90, 0.80, 0.30, 0.30],
                  [0.40, 0.38, 0.45, 0.30],
                  [0.40, 0.38, 0.30, 0.42]]
        s = self._summary(matrix, ["A", "B", "C", "D"])
        d = s["defeats"]
        self.assertEqual(d["winner"], "A")
        self.assertEqual(d["n_defeats"], 2)
        self.assertEqual([e["trial"] for e in d["events"]], [4, 5])
        self.assertEqual([e["detector"] for e in d["events"]], ["C", "D"])
        self.assertAlmostEqual(d["margin_max"], 0.05, places=6)
        self.assertAlmostEqual(d["margin_min"], 0.02, places=6)
        # Each challenger carries where it finished, and B is not among them.
        self.assertEqual([(c["detector"], c["place"]) for c in d["challengers"]],
                         [("C", 3), ("D", 4)])

    def test_a_winner_that_led_every_trial_has_no_defeats(self):
        matrix = [[0.9, 0.6, 0.3]] * 4
        self.assertEqual(self._summary(matrix, ["A", "B", "C"])["defeats"]["n_defeats"], 0)

    def test_one_challenger_taking_two_trials_is_listed_once(self):
        matrix = [[0.90, 0.80], [0.90, 0.80], [0.90, 0.80],
                  [0.40, 0.45], [0.40, 0.43]]
        d = self._summary(matrix, ["A", "B"])["defeats"]
        self.assertEqual(len(d["challengers"]), 1)
        self.assertEqual(d["challengers"][0]["trials"], 2)

    def test_the_published_ranking_sets_the_order(self):
        """The explanation must walk detectors the way the card does, even when
        the passed ranking and the means disagree."""
        s = self._summary(ranked=["E", "D", "C", "B", "A"])
        self.assertEqual(s["order"], ["E", "D", "C", "B", "A"])

    def test_leave_one_trial_out_keeps_a_winner_that_led_every_trial(self):
        s = self._summary()
        self.assertTrue(s["leave_one_out"]["stable"])
        self.assertEqual(s["leave_one_out"]["flips"], [])

    def test_leave_one_trial_out_names_the_trial_that_carries_first_place(self):
        """A wins on the mean only because of trial 1; drop it and B leads."""
        matrix = [[0.99, 0.50],
                  [0.40, 0.50],
                  [0.40, 0.50]]
        s = self._summary(matrix, ["A", "B"])
        loo = s["leave_one_out"]
        self.assertEqual(loo["winner"], "A")
        self.assertFalse(loo["stable"])
        self.assertEqual([f["trial"] for f in loo["flips"]], [1])
        self.assertEqual(loo["flips"][0]["winner"], "B")

    def test_the_gap_is_a_median_so_one_huge_gap_cannot_skew_it(self):
        """Four gaps of 0.05 and one of 0.60: the mean would report 0.16 and
        read as a well separated ranking, the median reports the 0.05 that
        actually separates neighbouring places."""
        row = [0.90, 0.85, 0.80, 0.75, 0.70, 0.10]
        s = self._summary([row, row], ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(s["median_spread"], 0.0)
        self.assertAlmostEqual(s["median_gap"], 0.05, places=6)


class TestWinnerDecomposition(unittest.TestCase):

    @staticmethod
    def _winner(f1, pr, names=("A", "B")):
        results = {n: {"f1_scores": list(np.asarray(f1)[:, j]),
                       "pr_auc_scores": list(np.asarray(pr)[:, j]),
                       "vus_scores": []} for j, n in enumerate(names)}
        pt = mc.build_trial_matrices(results, len(f1))
        return mc.summarize_trials(pt)["winner"]

    def test_leading_on_every_term_is_recorded_as_such(self):
        w = self._winner([[0.9, 0.5], [0.9, 0.5]], [[0.8, 0.4], [0.8, 0.4]])
        self.assertEqual(w["winner"], "A")
        self.assertEqual(w["runner_up"], "B")
        self.assertTrue(w["led_every_term"])

    def test_a_deficit_covered_by_another_term_is_recorded_as_such(self):
        """A wins the fitness while losing PR-AUC — a different reason to rank
        first, so the flag has to separate the two cases."""
        w = self._winner([[0.9, 0.5], [0.9, 0.5]], [[0.3, 0.4], [0.3, 0.4]])
        self.assertEqual(w["winner"], "A")
        self.assertFalse(w["led_every_term"])
        deltas = {t["metric"]: t["delta"] for t in w["terms"]}
        self.assertGreater(deltas["f1"], 0)
        self.assertLess(deltas["pr_auc"], 0)

    def test_the_margin_is_reported_per_trial(self):
        w = self._winner([[0.9, 0.5], [0.6, 0.5]], [[0.9, 0.5], [0.6, 0.5]])
        self.assertEqual(w["ahead_in_trials"], 2)
        self.assertAlmostEqual(w["margin_max"], 0.4, places=6)
        self.assertAlmostEqual(w["margin_min"], 0.1, places=6)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Orchestrator: report, figures, IR
# ════════════════════════════════════════════════════════════════════════════

class TestOrchestrator(unittest.TestCase):

    def _run(self, tmp, metrics=("f1", "pr_auc"), vus=None):
        os.chdir(tmp)
        pt = mc.build_trial_matrices(_trials(_CONTESTED, _NAMES, vus), 5, metrics)
        return mc.explain_monte_carlo(pt, _NAMES, "DS", "e1", explain=True,
                                      metrics=metrics, noise_level=0.1)

    def test_explain_false_returns_none(self):
        self.assertIsNone(mc.explain_monte_carlo({}, [], "DS", "e1", explain=False))

    def test_no_trials_returns_none(self):
        self.assertIsNone(mc.explain_monte_carlo({}, [], "DS", "e1", explain=True))

    def test_writes_report_and_figures(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self._run(tmp)
                directory = os.path.join(tmp, "myresults", "robustness",
                                         "MonteCarlo", "DS", "e1")
                names = set(os.listdir(directory))
                self.assertIn("DS_e1_MonteCarlo_trial_ranks.png", names)
                self.assertIn("DS_e1_MonteCarlo_trial_fitness.png", names)
                self.assertIn("DS_e1_MonteCarlo_explainability.txt", names)
            finally:
                os.chdir(cwd)

    def test_every_fitness_component_gets_its_own_figure(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self._run(tmp, metrics=("f1", "pr_auc", "vus"),
                          vus=np.full((5, 5), 0.5))
                names = set(os.listdir(os.path.join(
                    tmp, "myresults", "robustness", "MonteCarlo", "DS", "e1")))
                for tag in ("F1", "PRAUC", "VUS"):
                    self.assertIn(f"DS_e1_MonteCarlo_trial_{tag}.png", names)
            finally:
                os.chdir(cwd)

    def test_a_one_term_fitness_draws_no_component_figure(self):
        """The component figures exist to show which term moved; with one term
        they would duplicate the fitness figure exactly."""
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self._run(tmp, metrics=("f1",))
                names = set(os.listdir(os.path.join(
                    tmp, "myresults", "robustness", "MonteCarlo", "DS", "e1")))
                self.assertIn("DS_e1_MonteCarlo_trial_fitness.png", names)
                self.assertNotIn("DS_e1_MonteCarlo_trial_F1.png", names)
            finally:
                os.chdir(cwd)

    def test_the_report_carries_every_trials_ranking(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self._run(tmp)
                text = _read_report(tmp)
                for i in range(1, 6):
                    self.assertIn(f"Trial {i}:", text)
                self.assertIn("Median fitness spread between trials", text)
                self.assertIn("Leave-one-trial-out", text)
            finally:
                os.chdir(cwd)

    def test_no_sweep_is_run(self):
        """The layer reads the trials the ranking averages; a second experiment
        is exactly what this change removed."""
        for gone in ("monte_carlo_noise_sweep", "compute_noise_curves",
                     "train_noise_winner_surrogate", "plot_ranking_stability"):
            self.assertFalse(hasattr(mc, gone), gone)


def _read_report(tmp):
    directory = os.path.join(tmp, "myresults", "robustness", "MonteCarlo", "DS", "e1")
    name = [f for f in os.listdir(directory) if f.endswith("_explainability.txt")][0]
    with open(os.path.join(directory, name)) as f:
        return f.read()


if __name__ == "__main__":
    unittest.main(verbosity=2)
