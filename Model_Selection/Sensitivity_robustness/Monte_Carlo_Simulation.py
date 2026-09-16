import numpy as np
import copy
import os
from typing import Any, Dict, List, Optional
from Metrics.metrics import (range_based_precision_recall_f1_auc,
                             rank_key, vus_score, vus_window)
from Utils.model_selection_utils import evaluate_model, ScoringTimeout
from Utils.pipeline_spec import (DEFAULT_DECISION_METRICS, combine_metrics,
                                 decision_metric_formula, metrics_required)
from loguru import logger
import matplotlib.pyplot as plt
from Explainability import ir

# Keys `summarize_results` adds beside the per-model entries. Everything that
# walks the summary skips these, so a ranking cannot leak into a loop that
# expects model names.
RANKING_KEYS = ("ranked",)


def add_noise_to_data(data, noise_level=0.1):
    """Add Gaussian noise to the data."""
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise


def monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100,
                           noise_level=0.1, metrics=DEFAULT_DECISION_METRICS):
    """
    Perform Monte Carlo simulation to assess model robustness.

    Args:
        test_data: Original test data.
        trained_models: Dictionary of trained models.
        model_names: List of model names.
        dataset: Dataset name.
        entity: Entity name.
        n_simulations: Number of Monte Carlo simulations.
        noise_level: Level of noise to add to the data.

    Returns:
        (results, per_trial). `results` holds one score list per model and metric;
        `per_trial` holds the same numbers as trial x model matrices, which is
        what the explainability layer reads. Both are empty when the data is
        infeasible.
    """
    # Validation: Check if data is too small for Monte Carlo testing
    labels = test_data.entities[0].labels
    
    # Ensure labels are 2D
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    
    min_data_size = 50  # Minimum required data points for Monte Carlo
    data_size = labels.shape[1] if labels.ndim > 1 else labels.shape[0]
    
    if data_size < min_data_size:
        logger.warning(f"Monte Carlo simulation skipped: data size {data_size} < minimum {min_data_size}")
        return {}, {}

    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Monte Carlo simulation skipped: only one class present in labels (unique values: {unique_labels})")
        return {}, {}

    results = {model_name: {'f1_scores': [], 'pr_auc_scores': [], 'vus_scores': []}
               for model_name in model_names}
    # Noise does not change the series length, so one window covers every
    # simulation and keeps the VUS values on one scale.
    want_vus = 'vus' in metrics_required(metrics)
    vus_win = vus_window(test_data.entities[0].Y) if want_vus else None

    for sim in range(n_simulations):
        logger.info(f"Simulation {sim + 1}/{n_simulations}")
        noisy_data = copy.deepcopy(test_data)
        noisy_data.entities[0].Y = add_noise_to_data(noisy_data.entities[0].Y, noise_level)

        for model_name in model_names:
            model = trained_models.get(model_name)
            if not model or model_name not in results:
                continue
            try:
                evaluation = evaluate_model(noisy_data, model, model_name)
            except ScoringTimeout:
                results.pop(model_name, None)
                continue
            y_true = evaluation['anomaly_labels'].flatten()
            y_scores = evaluation['entity_scores'].flatten()
            _, _, f1, pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
            # f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
            # pr_auc = prauc(y_true, y_scores)
            results[model_name]['f1_scores'].append(f1)
            results[model_name]['pr_auc_scores'].append(pr_auc)
            if want_vus:
                results[model_name]['vus_scores'].append(vus_score(y_scores, y_true, vus_win))

    return results, build_trial_matrices(results, n_simulations, metrics)


def build_trial_matrices(results, n_simulations: int,
                         metrics=DEFAULT_DECISION_METRICS) -> Dict[str, Any]:
    """Per-model score lists -> trial x model matrices, plus the per-trial fitness.

    A model that timed out is dropped from `results` wholesale, so every list
    that survives has one entry per trial and the columns stay aligned: column j
    of every matrix is the same detector and row i is the same noise draw.
    """
    models = [m for m, v in results.items()
              if len(v.get('f1_scores') or []) == n_simulations]
    if not models or n_simulations < 1:
        return {}
    want = metrics_required(metrics)

    def column(model: str, key: str) -> List[float]:
        vals = results[model].get(key) or []
        if len(vals) != n_simulations:
            return [float('nan')] * n_simulations
        return [float(v) for v in vals]

    f1 = np.array([column(m, 'f1_scores') for m in models], dtype=float).T
    pr = np.array([column(m, 'pr_auc_scores') for m in models], dtype=float).T
    vus = np.array([column(m, 'vus_scores') for m in models], dtype=float).T
    fitness = np.empty_like(f1)
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            fitness[i, j] = combine_metrics(
                metrics, {'f1': f1[i, j], 'pr_auc': pr[i, j], 'vus': vus[i, j]},
                renormalise=False)
    return {"model_names": models, "n_trials": int(n_simulations),
            "F1": f1, "PR": pr, "VUS": vus, "FIT": fitness, "metrics": want}


def run_monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100,
                               noise_level=0.1, explain=False, metrics=DEFAULT_DECISION_METRICS):
    """Run the entire Monte Carlo simulation process."""
    # Run Monte Carlo simulation
    results, per_trial = monte_carlo_simulation(
        test_data, trained_models, model_names, dataset, entity, n_simulations,
        noise_level, metrics=metrics)

    # Handle empty results (when data is too small or invalid)
    if not results:
        logger.warning("Monte Carlo simulation returned empty results")
        return []

    # Summarize results
    summary = summarize_results(results, metrics=metrics)

    # Handle empty summary
    if not summary or 'ranked' not in summary:
        logger.warning("Monte Carlo summary is empty or incomplete")
        return []

    # Print summary and rankings
    print("Summary of Monte Carlo Simulation:")
    for model_name, model_metrics in summary.items():
        if model_name not in RANKING_KEYS:
            print(f"Model: {model_name}")
            print(f"  F1 Mean: {model_metrics['f1_mean']:.4f}, F1 Std: {model_metrics['f1_std']:.4f}")
            print(f"  PR AUC Mean: {model_metrics['pr_auc_mean']:.4f}, "
                  f"PR AUC Std: {model_metrics['pr_auc_std']:.4f}")

    print(f"\nModels ranked by fitness ({decision_metric_formula(metrics)}):")
    ranked_models = []
    for rank, model_name in enumerate(summary['ranked'], 1):
        print(f"{rank}. {model_name}")
        ranked_models.append(model_name)

    # Save summary
    save_summary(summary, dataset, entity, metrics=metrics, per_trial=per_trial,
                 noise_level=noise_level)

    # One F1/PR-AUC histogram pair per detector, no longer drawn. Eleven figures
    # per entity that say separately what the noise curves say together, and the
    # comparison between detectors is the whole point of this stage — which is
    # why the WebUI stopped listing them. plot_monte_carlo_results is kept below
    # so a single detector's distribution can still be minted by hand.
    # plot_monte_carlo_results(results, summary, model_names, dataset, entity)

    # Explainability reads the trials the ranking was built from; it runs no
    # experiment of its own, so it cannot disagree with the ranking it explains.
    if explain:
        try:
            explain_monte_carlo(per_trial, ranked_models, dataset, entity,
                                explain=True, metrics=metrics,
                                noise_level=noise_level)
        except Exception as e:
            logger.error(f"Monte Carlo explainability failed (non-fatal): {e}")

    return ranked_models




def summarize_results(results, metrics=DEFAULT_DECISION_METRICS):
    """Summarize the results of Monte Carlo simulation and rank models.

    One ranking, by the run's own fitness over the per-model means. Ranking each
    metric separately let a three-metric fitness publish three winners, none of
    them weighted as the run configuration asked for.
    """
    summary = {}
    for model_name, model_metrics in results.items():
        f1_mean = np.mean(model_metrics['f1_scores'])
        f1_std = np.std(model_metrics['f1_scores'])
        pr_auc_mean = np.mean(model_metrics['pr_auc_scores'])
        pr_auc_std = np.std(model_metrics['pr_auc_scores'])
        vus = [v for v in (model_metrics.get('vus_scores') or []) if not np.isnan(v)]
        entry = {
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'pr_auc_mean': pr_auc_mean,
            'pr_auc_std': pr_auc_std,
            'vus_mean': float(np.mean(vus)) if vus else float('nan'),
            'vus_std': float(np.std(vus)) if vus else float('nan'),
        }
        # renormalise=False: a detector missing a term must not be scored on a
        # narrower fitness than the ones it is ranked against.
        entry['fitness'] = combine_metrics(
            metrics, {'f1': f1_mean, 'pr_auc': pr_auc_mean,
                      'vus': entry['vus_mean']}, renormalise=False)
        summary[model_name] = entry

    ranked = sorted(summary.items(), key=lambda x: rank_key(x[1]['fitness']),
                    reverse=True)
    summary['ranked'] = [item[0] for item in ranked]

    return summary


def plot_monte_carlo_results(results, summary, model_names, dataset, entity):
    """Plot the results of Monte Carlo simulation."""
    for model_name in model_names:
        f1_scores = results[model_name]['f1_scores']
        pr_auc_scores = results[model_name]['pr_auc_scores']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=20, alpha=0.7, color='blue')
        plt.title(f'{model_name} F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['f1_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['f1_mean'] + summary[model_name]['f1_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["f1_mean"]:.2f}\nStd: {summary[model_name]["f1_std"]:.2f}',
                 color='red')

        plt.subplot(1, 2, 2)
        plt.hist(pr_auc_scores, bins=20, alpha=0.7, color='green')
        plt.title(f'{model_name} PR AUC Scores')
        plt.xlabel('PR AUC Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['pr_auc_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['pr_auc_mean'] + summary[model_name]['pr_auc_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["pr_auc_mean"]:.2f}\nStd: {summary[model_name]["pr_auc_std"]:.2f}',
                 color='red')

        plt.tight_layout()

        # Save the plot
        directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
        os.makedirs(directory, exist_ok=True)
        filename = f'{dataset}_{entity}_{model_name}_MonteCarloResults.png'
        plt.savefig(os.path.join(directory, filename), dpi=300)

        # plt.show()


def save_summary(summary, dataset, entity, metrics=DEFAULT_DECISION_METRICS,
                 per_trial=None, noise_level=None):
    """Save the summary of Monte Carlo simulation to a file."""
    directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    summary_file = os.path.join(directory, f'{dataset}_{entity}_MonteCarloSummary.txt')

    with open(summary_file, 'w') as f:
        f.write("Summary of Monte Carlo Simulation:\n")
        for model_name, model_metrics in summary.items():
            if model_name not in RANKING_KEYS:
                f.write(f"Model: {model_name}\n")
                f.write(f"  F1 Mean: {model_metrics['f1_mean']:.4f}, "
                        f"F1 Std: {model_metrics['f1_std']:.4f}\n")
                f.write(f"  PR AUC Mean: {model_metrics['pr_auc_mean']:.4f}, "
                        f"PR AUC Std: {model_metrics['pr_auc_std']:.4f}\n")
                if not np.isnan(model_metrics.get('vus_mean', float('nan'))):
                    f.write(f"  VUS Mean: {model_metrics['vus_mean']:.4f}, "
                            f"VUS Std: {model_metrics['vus_std']:.4f}\n")
                f.write(f"  Fitness  : {model_metrics['fitness']:.4f}\n")

        f.write(f"\nModels ranked by fitness ({decision_metric_formula(metrics)}):\n")
        for rank, model_name in enumerate(summary['ranked'], 1):
            f.write(f"{rank}. {model_name}\n")

        if per_trial:
            fit = per_trial["FIT"]
            models = per_trial["model_names"]
            level = "" if noise_level is None else f" at noise level {noise_level}"
            f.write(f"\nFitness per trial ({per_trial['n_trials']} independent "
                    f"noise draws{level}):\n")
            header = "".join(f"{'trial ' + str(i + 1):>10}"
                             for i in range(per_trial["n_trials"]))
            f.write(f"  {'model':<22}{header}\n")
            for name in summary['ranked']:
                if name not in models:
                    continue
                col = fit[:, models.index(name)]
                cells = "".join(f"{v:>10.4f}" for v in col)
                f.write(f"  {name:<22}{cells}\n")



# ════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Robustness Explainability
#
#  The ranking averages several independent draws of the same noise, and this
#  layer explains it from those draws — no second experiment, no proxy metric,
#  so every number here is one the ranking was actually built from. It answers
#  why the winner leads (which fitness terms earned it, whether it led the
#  trials rather than only their mean, whether first place survives dropping any
#  one trial) and how far down the ranking that reasoning still holds.
# ════════════════════════════════════════════════════════════════════════════

def rank_per_trial(fit: np.ndarray) -> np.ndarray:
    """Trial x model fitness -> trial x model rank, 1 = best. NaN sorts last."""
    n_trials, n_models = fit.shape
    ranks = np.zeros((n_trials, n_models), dtype=int)
    for i in range(n_trials):
        row = fit[i]
        order = sorted(range(n_models),
                       key=lambda j: rank_key(row[j]), reverse=True)
        for place, j in enumerate(order, 1):
            ranks[i, j] = place
    return ranks


def leave_one_trial_out(fit: np.ndarray, models: List[str]) -> Dict[str, Any]:
    """Re-rank with each trial removed in turn; report which removals move first place.

    The same question LOFO asks of the ensemble: is the published claim a
    property of the detectors, or of the particular sample it was computed from.
    """
    n_trials = fit.shape[0]
    winner = models[int(np.argmax(np.nanmean(fit, axis=0)))]
    flips: List[Dict[str, Any]] = []
    for i in range(n_trials):
        kept = [t for t in range(n_trials) if t != i]
        alt = models[int(np.argmax(np.nanmean(fit[kept], axis=0)))]
        if alt != winner:
            flips.append({"trial": i + 1, "winner": alt})
    return {"winner": winner, "n_trials": n_trials, "flips": flips,
            "stable": not flips}


def winner_defeats(fit: np.ndarray, models: List[str],
                   order: List[int]) -> Dict[str, Any]:
    """The trials the top-ranked detector did not lead, and who led them instead.

    The runner-up by mean need not be any of these: a detector can finish second
    overall without ever topping a trial. Each challenger carries its own place
    in the published ranking, because one beaten by the detector ranked second
    and one beaten by the detector ranked ninth are different findings.
    """
    if not order:
        return {}
    win_j = order[0]
    place = {j: p for p, j in enumerate(order, 1)}
    events: List[Dict[str, Any]] = []
    for i in range(fit.shape[0]):
        lead_j = int(np.argmax(fit[i]))
        if lead_j == win_j:
            continue
        events.append({"trial": i + 1, "detector": models[lead_j],
                       "place": place[lead_j],
                       "margin": float(fit[i, lead_j] - fit[i, win_j])})
    if not events:
        return {"winner": models[win_j], "n_defeats": 0, "events": [],
                "challengers": []}
    challengers = []
    for j in order:
        hits = [e for e in events if e["detector"] == models[j]]
        if hits:
            challengers.append({"detector": models[j], "place": place[j],
                                "trials": len(hits)})
    margins = [e["margin"] for e in events]
    return {"winner": models[win_j], "n_defeats": len(events),
            "events": events, "challengers": challengers,
            "margin_min": float(min(margins)), "margin_max": float(max(margins))}


def winner_decomposition(per_trial: Dict[str, Any], order: List[int]) -> Dict[str, Any]:
    """How the winner's fitness advantage over the runner-up is made up.

    Reports each fitness term separately, because leading on every term and
    covering a deficit on one with a lead on another are different reasons to
    rank first.
    """
    if len(order) < 2:
        return {}
    models = per_trial["model_names"]
    win_j, run_j = order[0], order[1]
    parts = {"f1": per_trial["F1"], "pr_auc": per_trial["PR"], "vus": per_trial["VUS"]}
    terms = []
    for name in per_trial["metrics"]:
        mat = parts[name]
        w = float(np.nanmean(mat[:, win_j]))
        r = float(np.nanmean(mat[:, run_j]))
        terms.append({"metric": name, "winner": w, "runner_up": r, "delta": w - r})
    fit = per_trial["FIT"]
    margins = fit[:, win_j] - fit[:, run_j]
    return {
        "winner": models[win_j],
        "runner_up": models[run_j],
        "terms": terms,
        "led_every_term": all(t["delta"] > 0 for t in terms),
        "margin_mean": float(np.nanmean(margins)),
        "margin_min": float(np.nanmin(margins)),
        "margin_max": float(np.nanmax(margins)),
        "ahead_in_trials": int(np.sum(margins > 0)),
        "margins": [float(v) for v in margins],
    }


def summarize_trials(per_trial: Dict[str, Any],
                     ranked: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Every structure the Monte Carlo explanation is built from. Pure."""
    if not per_trial or not per_trial.get("model_names"):
        return None
    models = list(per_trial["model_names"])
    fit = np.asarray(per_trial["FIT"], dtype=float)
    means = np.nanmean(fit, axis=0)
    # The ranking the stage published, so the explanation orders detectors the
    # way the card does; the means are the fallback when it was not passed in.
    if ranked:
        order = [models.index(m) for m in ranked if m in models]
        order += [j for j in range(len(models)) if j not in order]
    else:
        order = sorted(range(len(models)), key=lambda j: rank_key(means[j]),
                       reverse=True)

    ranks = rank_per_trial(fit)
    spreads = np.nanmax(fit, axis=0) - np.nanmin(fit, axis=0)
    wins = {models[j]: int(np.sum(ranks[:, j] == 1)) for j in range(len(models))}
    rank_ranges = {models[j]: (int(ranks[:, j].min()), int(ranks[:, j].max()))
                   for j in range(len(models))}

    ordered_means = [means[j] for j in order]
    gaps = [abs(a - b) for a, b in zip(ordered_means, ordered_means[1:])]

    return {
        "model_names": models,
        "order": [models[j] for j in order],
        "n_trials": int(per_trial["n_trials"]),
        "fitness": fit,
        "means": {models[j]: float(means[j]) for j in range(len(models))},
        "ranks": ranks,
        "wins": wins,
        "rank_ranges": rank_ranges,
        "spreads": {models[j]: float(spreads[j]) for j in range(len(models))},
        # Medians, not means: the gaps between neighbouring places are heavily
        # skewed by the few large ones at the bottom of the ranking, and a mean
        # gap there reads as though the whole order were well separated.
        "median_spread": float(np.nanmedian(spreads)),
        "median_gap": float(np.nanmedian(gaps)) if gaps else float('nan'),
        "leave_one_out": leave_one_trial_out(fit, models),
        "defeats": winner_defeats(fit, models, order),
        "winner": winner_decomposition(per_trial, order),
        "metrics": list(per_trial["metrics"]),
    }


# ── Plots ────────────────────────────────────────────────────────────────────

# Enough lines to read the crossings without the legend swallowing the axes.
BUMP_CHART_MODELS = 10


def _mc_explain_rcparams() -> None:
    plt.rcParams.update({"figure.autolayout": False, "axes.grid": True,
                         "grid.alpha": 0.3})


def _mc_dir(dataset, entity) -> str:
    directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_trial_ranks(summary: Dict[str, Any], dataset, entity,
                     top_k: int = BUMP_CHART_MODELS) -> None:
    """Rank of each detector in each trial; crossing lines are unsettled places."""
    _mc_explain_rcparams()
    models, ranks = summary["model_names"], summary["ranks"]
    shown = summary["order"][:top_k]
    n_trials = summary["n_trials"]
    x = np.arange(1, n_trials + 1)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * n_trials + 3), 6))
    for name in shown:
        y = ranks[:, models.index(name)]
        ax.plot(x, y, marker='o', linewidth=1.8, markersize=5, label=name)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Rank by fitness")
    ax.set_xticks(x)
    ax.invert_yaxis()
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=8)
    fig.savefig(f"{_mc_dir(dataset, entity)}/{dataset}_{entity}_MonteCarlo_trial_ranks.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_trial_scores(summary: Dict[str, Any], matrix: np.ndarray, tag: str,
                      xlabel: str, dataset, entity) -> None:
    """One row per detector: its score in every trial, beside the mean the ranking uses."""
    _mc_explain_rcparams()
    models = summary["model_names"]
    order = summary["order"]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(order) + 1.2)))
    for row, name in enumerate(order):
        col = matrix[:, models.index(name)]
        y = len(order) - row
        ax.scatter(col, [y] * len(col), s=26, alpha=0.65, color='#4C72B0',
                   zorder=3)
        ax.scatter([np.nanmean(col)], [y], marker='|', s=420, linewidths=2.0,
                   color='#C44E52', zorder=4)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(list(reversed(order)), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.grid(False, axis='y')
    fig.savefig(f"{_mc_dir(dataset, entity)}/{dataset}_{entity}_MonteCarlo_trial_{tag}.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# Figure tag and axis label per fitness component, drawn under the browse button
# beside the fitness figure so a reader can see which term moved.
_COMPONENT_FIGURES = (("f1", "F1", "F1", "F1"),
                      ("pr_auc", "PR", "PRAUC", "PR-AUC"),
                      ("vus", "VUS", "VUS", "VUS"))


def explain_monte_carlo(per_trial: Dict[str, Any],
                        ranked: Optional[List[str]] = None,
                        dataset: str = "", entity: str = "",
                        explain: bool = False,
                        metrics=DEFAULT_DECISION_METRICS,
                        noise_level: Optional[float] = None,
                        ) -> Optional[Dict[str, Any]]:
    """
    Monte Carlo robustness explainability, read off the trials the ranking
    averages. Writes a report + figures under
    myresults/robustness/MonteCarlo/{dataset}/{entity}/ and emits the IR.

    Returns the computed structures when explain=True; None otherwise (and None,
    with a logged note, when there are no usable trials).
    """
    if not explain:
        return None
    summary = summarize_trials(per_trial, ranked)
    if summary is None:
        logger.warning("Monte Carlo explainability skipped: no usable trials.")
        return None

    fitness_formula = decision_metric_formula(metrics)
    summary["fitness_formula"] = fitness_formula
    summary["noise_level"] = noise_level

    plot_trial_ranks(summary, dataset, entity)
    plot_trial_scores(summary, summary["fitness"], "fitness",
                      f"Fitness ({fitness_formula})", dataset, entity)
    parts = {"f1": per_trial["F1"], "pr_auc": per_trial["PR"], "vus": per_trial["VUS"]}
    if len(summary["metrics"]) > 1:
        for token, _, tag, label in _COMPONENT_FIGURES:
            if token in summary["metrics"]:
                plot_trial_scores(summary, parts[token], tag, label, dataset, entity)

    _write_mc_report(summary, dataset, entity)

    try:
        ir.write_stage_ir(
            ir.build_monte_carlo_ir(dataset, entity, summary, list(ranked or [])),
            dataset, entity, "ir_monte_carlo")
    except Exception as e:
        logger.error(f"Monte Carlo IR emission failed (non-fatal): {e}")

    return summary


def _write_mc_report(summary: Dict[str, Any], dataset, entity) -> None:
    path = os.path.join(_mc_dir(dataset, entity),
                        f'{dataset}_{entity}_MonteCarlo_explainability.txt')
    models, order = summary["model_names"], summary["order"]
    fit, ranks = summary["fitness"], summary["ranks"]
    n_trials = summary["n_trials"]
    level = summary.get("noise_level")

    with open(path, 'w') as f:
        f.write("=== Monte Carlo Robustness Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Detectors ({len(models)}): {', '.join(order)}\n")
        f.write(f"Trials: {n_trials} independent draws of the same Gaussian noise"
                + (f" at level {level}\n" if level is not None else "\n"))
        f.write(f"Fitness: {summary['fitness_formula']}\n")
        f.write("(Read off the trials the ranking averages; no separate experiment.)\n\n")

        f.write("--- Ranking in each trial ---\n")
        for i in range(n_trials):
            places = sorted(range(len(models)), key=lambda j: ranks[i, j])
            f.write(f"  Trial {i + 1}: "
                    + ", ".join(f"{p}. {models[j]}"
                                for p, j in enumerate(places, 1)) + "\n")
        f.write("\n")

        f.write("--- Fitness per detector ---\n")
        head = "".join(f"{'trial ' + str(i + 1):>10}" for i in range(n_trials))
        f.write(f"  {'detector':<22}{head}{'mean':>10}{'spread':>9}{'ranks':>10}{'wins':>6}\n")
        f.write("  " + "-" * (22 + 10 * n_trials + 35) + "\n")
        for name in order:
            col = fit[:, models.index(name)]
            lo, hi = summary["rank_ranges"][name]
            cells = "".join(f"{v:>10.4f}" for v in col)
            f.write(f"  {name:<22}{cells}{summary['means'][name]:>10.4f}"
                    f"{summary['spreads'][name]:>9.4f}"
                    f"{(f'{lo}-{hi}' if lo != hi else str(lo)):>10}"
                    f"{summary['wins'][name]:>6}\n")
        f.write("\n")

        w = summary["winner"]
        if w:
            f.write("--- Why the ranking has this winner ---\n")
            f.write(f"  {w['winner']} leads the runner-up {w['runner_up']} by "
                    f"{w['margin_mean']:.4f} on average.\n")
            for term in w["terms"]:
                f.write(f"    {term['metric']:<8} {w['winner']} {term['winner']:.4f}"
                        f"  vs  {w['runner_up']} {term['runner_up']:.4f}"
                        f"  ({term['delta']:+.4f})\n")
            f.write(f"  Ahead in {w['ahead_in_trials']} of {n_trials} trials, by "
                    f"between {w['margin_min']:.4f} and {w['margin_max']:.4f}.\n")
            lo = summary["leave_one_out"]
            if lo["stable"]:
                f.write("  Leave-one-trial-out: first place is unchanged whichever "
                        "single trial is dropped.\n\n")
            else:
                f.write("  Leave-one-trial-out: "
                        + "; ".join(f"dropping trial {fl['trial']} gives "
                                    f"{fl['winner']} first place"
                                    for fl in lo["flips"]) + ".\n\n")

        f.write("--- Trial-to-trial movement against the published gaps ---\n")
        f.write(f"  Median fitness spread between trials: {summary['median_spread']:.4f}\n")
        f.write(f"  Median gap between neighbouring places: {summary['median_gap']:.4f}\n")
        f.write("\nWhere the spread exceeds the gap, neighbouring places in the ranking "
                "are closer together than the same detector's own movement between "
                "trials. The per-detector rank column above shows which places changed.\n")
