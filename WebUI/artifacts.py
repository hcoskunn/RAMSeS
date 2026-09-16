"""
Read-only reader over the explainability artifacts.

Assembles the page payload from the per-stage `nl_*.txt` and `ir_*.json` files
rather than parsing `nl_global_iter*.txt`. The global text is *derived* from the
same per-stage strings by `Explainability.llm.compose_global_narrative`, so
parsing it back would be a lossy round-trip through a format whose section
separators are `"=" * len(title)` — any wording change there would silently
break the reader with no test failure. Assembling instead also yields the
structured data the text cannot carry (decision, stage agreement, per-stage
outputs, caveats, confidence), which is what makes the page readable rather than
a 2,000-word wall. The global `.txt` stays available as a verbatim download.
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Utils.pipeline_spec import dataset_label
from WebUI import paths
from WebUI.summarize import attribute_sentences, summarize

# ── The three-vocabulary map ────────────────────────────────────────────────
#
# Three different names exist for the same stage: the CLI --stages token, the
# IR/NL stage key, and the plot directory. This tuple is the single owner of
# that mapping; plots.py, markers.py and the frontend all join on `key`.
# Order matches Explainability.llm._GLOBAL_STAGE_ORDER.
STAGES: Tuple[Dict[str, Any], ...] = (
    {"key": "ga_selection", "title": "Genetic Algorithm: Selection",
     "cli": "ga", "ir": "ir_ga_selection", "nl": "nl_ga_selection",
     "plot_group": "ga_selection", "order": 1},
    {"key": "ga_combination", "title": "Genetic Algorithm: Combination",
     "cli": "ga", "ir": "ir_ga_combination", "nl": "nl_ga_combination",
     "plot_group": "ga_combination", "order": 2},
    # One CLI token, two stages — the same split as ga_selection/ga_combination.
    # `thompson_ranking` explains mu^T mu, the criterion the detectors are
    # ordered by; `thompson_sampling` explains mu^T c_t, the estimated expected reward that
    # drove per-window selection. Neither keeps the plain name.
    #
    # `plot_group` is deliberately `ts_ranking`, not `thompson_ranking`:
    # result.js matches lazy-gallery descriptors with `id.startsWith(plot_group)`,
    # so a group prefixed by "thompson" would let one card claim the other's
    # galleries. `regimes` names the plot subdirectory whose per-regime figures
    # pair with this stage's regime atoms; its presence is what makes a stage
    # regime-bearing, replacing a hardcoded stage-key check here and in server.py.
    # `regime_noun` is what the page calls one: the two stages segment different
    # quantities, so Ranking says "Streak" where Selection says "Regime".
    {"key": "thompson_ranking", "title": "Thompson Sampling: Ranking",
     "cli": "thompson", "ir": "ir_thompson_ranking", "nl": "nl_thompson_ranking",
     "plot_group": "ts_ranking", "order": 3,
     "regimes": ["ranking_per_regime"], "regime_noun": "Streak"},
    {"key": "thompson_sampling", "title": "Thompson Sampling: Selection",
     "cli": "thompson", "ir": "ir_thompson", "nl": "nl_thompson",
     "plot_group": "thompson", "order": 4,
     "regimes": ["reward_per_regime", "shap_per_regime"], "regime_noun": "Regime"},
    {"key": "monte_carlo", "title": "Robustness: Monte Carlo",
     "cli": "montecarlo", "ir": "ir_monte_carlo", "nl": "nl_monte_carlo",
     "plot_group": "monte_carlo", "order": 5},
    {"key": "off_by_threshold", "title": "Robustness: Off-by-threshold",
     "cli": "offby", "ir": "ir_off_by", "nl": "nl_off_by",
     "plot_group": "off_by", "order": 6},
    # `plot_group` is "gan", which is prefix-free against "ga_selection" and
    # "ga_combination" — the lazy-gallery matcher joins on startswith, so a
    # group that prefixed another would let one card claim the other's galleries.
    {"key": "gan", "title": "Robustness: GAN",
     "cli": "gan", "ir": "ir_gan", "nl": "nl_gan",
     "plot_group": "gan", "order": 7},
    {"key": "rank_aggregation_robust", "title": "Robustness Aggregation",
     "cli": None, "ir": "ir_rank_aggregation_robust", "nl": "nl_rank_aggregation_robust",
     "plot_group": "rank_aggregation_robust", "order": 8, "iterated": True},
    {"key": "rank_aggregation_final", "title": "Final Aggregation",
     "cli": None, "ir": "ir_rank_aggregation_final", "nl": "nl_rank_aggregation_final",
     "plot_group": "rank_aggregation_final", "order": 9, "iterated": True},
)

STAGE_BY_KEY = {s["key"]: s for s in STAGES}

# WebUI/static/js/docs.js: {"text"} is a paragraph, {"lead", "text"} a paragraph
# opening with a bold label, {"list", "ordered"} a list, {"formula"} a block of
# notation set in the monospace face.
#
# A section's own `blocks` are its opening, before the first subsection. The
# subsections are what the sidebar navigates to and what the N.M numbering is
# built from; their titles were the bold labels the prose used to open those
# paragraphs with, moved up into a heading now that a section is long enough to
# need navigating inside. Numbering lives in the frontend, not here, so a
# section inserted later renumbers everything after it without an edit.
DOC_SECTIONS: Tuple[Dict[str, Any], ...] = (
    {"id": "overview", "title": "Overview", "stages": (), "blocks": (
        {"text": "RAMSeS selects a detection strategy for one time series rather "
                 "than assuming one detector fits all of them. What counts as an "
                 "anomaly is context-dependent, and a detector that wins on one "
                 "series degrades on another, so the framework runs a pool of "
                 "pre-trained base detectors and decides, per series, how to use "
                 "them."},
    ), "subsections": (
        {"id": "overview-pool", "title": "The base detector pool", "blocks": (
            {"text": "The detector pool is chosen from 107 detector instances "
                     "across 34 families, consisting of statistical models, "
                     "neural networks, foundation models and graph-based "
                     "detectors. Each is trained beforehand and, for every "
                     "window of the series, emits an anomaly score. Every stage "
                     "below consumes those scores, and none of them retrains a "
                     "base detector."},
            {"text": "Not every detector fits every series. POLY, TimesFM and "
                     "Series2Graph are univariate only, so they are dropped on SKAB (9 channels) and "
                     "SMD (38). ABOD is the mirror case. Since angles need more than "
                     "one dimension, it is dropped on the single-channel UCR "
                     "entities. The run configuration hides what the chosen "
                     "entity cannot use, and a detector named on the command "
                     "line anyway is skipped with a warning rather than failing "
                     "the run. Additionally, any detector that needs more than 120 "
                     "seconds for a single scoring call is killed and excluded "
                     "from the rest of that run."},
        )},
        {"id": "overview-injection", "title": "Synthetic anomalies", "blocks": (
            {"text": "The framework's ground truth is not only the dataset's own "
                     "labels. Before model selection begins, synthetic anomalies "
                     "are injected into a copy of the series, and it is on those "
                     "injected timesteps that the stages score detectors. This "
                     "is what lets every stage judge detectors against a known "
                     "answer on the same series."},
            {"text": "Nine anomaly types are available: spikes (default), "
                     "contextual, flip, speedup, noise, cutoff, scale, wander "
                     "and average. --anomaly_rate sets how much of the series is "
                     "labelled anomalous, as a fraction in (0, 1]. For spikes it "
                     "is the per-timestep injection probability, and for the "
                     "other types it sizes the injected segment. Omitting it "
                     "keeps each type's own default."},
        )},
        {"id": "overview-branches", "title": "The ensemble and single-model branches", "blocks": (
            {"text": "From the shared pool the framework runs two branches with "
                     "different aims."},
            {"list": (
                "The ensemble branch seeks the optimal subset of base "
                "detectors, whose scores, stacked and fed to a meta-learner, "
                "perform better than any other evaluated subset.",
                "The single-model branch seeks the best individual detector, "
                "combining an adaptive bandit with three robustness tests.",
            )},
        )},
        {"id": "overview-pipeline", "title": "The six offline sub-stages", "blocks": (
            {"text": "The offline stage runs six sub-stages: 6.1 Genetic "
                     "algorithm, 6.2 Thompson sampling, 6.3 GAN perturbation, "
                     "6.4 Off-by-threshold, 6.5 Monte Carlo, 6.6 Rank "
                     "aggregation. The three robustness tests are independent of "
                     "one another and each works on its own copy of the data, so "
                     "no test can see another's perturbations."},
        )},
        {"id": "overview-decision",
         "title": "Two aggregations and the final decision", "blocks": (
            {"text": "Rank aggregation runs twice. First it merges the rankings "
                     "the robustness tests produce into a robustness consensus. "
                     "Each of the three tests ranks the detectors by the run's "
                     "own fitness function, resulting in three rankings based on "
                     "the weights set in the run configuration, which are then "
                     "aggregated to a robustness consensus ranking. Then it "
                     "merges that consensus with the "
                     "Thompson ranking into the final single-model order. "
                     "Finally the framework evaluates the chosen ensemble and "
                     "the highest ranked single detector on the same data, the "
                     "held-out test split with the synthetic anomalies injected "
                     "into it, and deploys whichever scores higher on the "
                     "fitness."},
        )},
        {"id": "overview-online", "title": "Online phase and re-optimisation", "blocks": (
            {"text": "The series is split 80% offline / 20% online. The online "
                     "portion is processed in overlapping windows, and every N "
                     "windows (default 5) the framework re-optimises, "
                     "concatenating the most recent online windows while "
                     "dropping an equal number of the oldest offline samples so "
                     "the training size stays constant. This adapts to "
                     "distribution shift without forgetting earlier behaviour. "
                     "Three strategies are available: adaptive (re-optimise "
                     "periodically), fixed-best (keep the offline winner), and "
                     "fixed-random (a baseline)."},
        )},
        {"id": "overview-explained", "title": "How the explanations are produced",
         "blocks": (
            {"text": "Every stage below can be run with an explainability layer "
                     "on top of it, disabled by default and enabled with "
                     "--explain. The layers never change what the stage decides: "
                     "they observe a run as it happens without altering it, and "
                     "each one writes a report, generates a set of figures, and "
                     "a structured record of its findings."},
            {"text": "That record is what a local language model turns into the "
                     "prose on each stage's explanation. It is given canonical "
                     "sentences with the numbers already computed and rounded, "
                     "never raw output, so it composes and compresses rather "
                     "than calculating or inferring. Limitations are listed "
                     "separately and verbatim, never left to the model to "
                     "paraphrase."},
        )},
    )},
    {"id": "ga", "title": "Genetic algorithm",
     "stages": ("ga_selection", "ga_combination"), "blocks": (
        {"text": "The ensemble branch builds a stacking ensemble: the base "
                 "detectors are level-0 learners, and their scores become the "
                 "input features of a level-1 meta-learner that makes the final "
                 "call. The question is which subset of detectors to stack."},
    ), "subsections": (
        {"id": "ga-meta-learner",
         "title": "Why the meta-learner is fixed for the run", "blocks": (
            {"text": "RAMSeS defaults to a Random Forest, chosen for a comparable "
                     "fitness to an SVM at lower cost, and logistic regression, "
                     "gradient boosting and SVM can be picked instead at run "
                     "configuration. Whichever is chosen, it is not searched "
                     "over per subset, because doing so would raise the risk of "
                     "overfitting, destabilise the optimisation, and make "
                     "convergence harder to read. Fixing it restricts the search "
                     "to the subsets themselves."},
        )},
        {"id": "ga-loop", "title": "The search loop and its fitness", "blocks": (
            {"text": "The algorithm initialises a population of 20 randomly "
                     "drawn distinct subsets and runs for 20 generations at a "
                     "mutation rate of 0.1. In each generation it:"},
            {"ordered": True, "list": (
                "Trains the meta-learner on the stacked outputs of each "
                "candidate subset",
                "Scores each ensemble on a held-out validation fold",
                "Keeps the top performers as the elite pool",
                "Crosses pairs of parents drawn from that pool into new subsets",
                "Mutates some of the resulting new subsets, adding, removing or "
                "replacing a detector.",
            )},
            {"text": "Fitness is the ensemble's score on the run's chosen "
                     "metric, which may be best-threshold F1, PR-AUC, VUS, or a "
                     "weighted mean of them. That score is the objective the "
                     "search maximises. After the last generation, the "
                     "highest-scoring ensemble across all generations is "
                     "chosen."},
        )},
        {"id": "ga-meta-use",
         "title": "From detector scores to one prediction", "blocks": (
            {"text": "For a point in time-series, every detector has an "
                     "anomaly score as output. Meta-learner "
                     "gets the output of all of the level-0 detectors that are "
                     "in the subset as input and produces its own predicted "
                     "probability that the point is an anomaly. It must be "
                     "trained first, so that it learns how much to trust each "
                     "detector and in what direction. It does not average them "
                     "and it does not weight them by any external notion of "
                     "quality, the weighting is whatever fitting the training "
                     "data produced."},
        )},
        {"id": "ga-explained", "title": "What the explanation answers", "blocks": (
            {"text": "The search evaluates hundreds of subsets and reports one. "
                     "Consequently, two questions arise: why each detector ended up in "
                     "the chosen ensemble, and how the meta-learner uses them "
                     "once it has selected them."},
        )},
        {"id": "ga-why-chosen", "title": "Utility, stability and LOFO", "blocks": (
            {"text": "Two properties are measured after the search, from the "
                     "subsets the search itself evaluated. Utility is a "
                     "detector's mean marginal contribution: the average change "
                     "in fitness between the subsets that contained the detector "
                     "and those that did not. Stability is its survival rate, "
                     "the share of evaluated subsets that detector appeared in. "
                     "Each detector is placed on both axes at once and labelled "
                     "high or low on each, split at the median of the detector "
                     "pool, so \"high utility\" means above the other detectors "
                     "here rather than good in absolute terms. A third quantity, "
                     "LOFO, is reported for members of the final ensemble only: "
                     "the ensemble's fitness loss when that one detector is "
                     "removed from it. Utility is a global average over the "
                     "whole search, LOFO is local to the one ensemble that was "
                     "chosen, and the two can disagree in sign."},
            {"formula": "utility(d)   = mean{ fit(S) : d ∈ S } − mean{ fit(S) : d ∉ S }\n"
                        "stability(d) = (1/G) · Σ_g |{ individuals in generation g containing d }| / P\n"
                        "LOFO(d)      = fit(Ŝ) − fit(Ŝ \\ {d}),    d ∈ Ŝ"},
            {"text": "where S ranges over the distinct subsets the GA evaluated, "
                     "Ŝ is the chosen ensemble, G is the number of generations "
                     "and P the size of the population in each generation."},
        )},
        {"id": "ga-meta-explained",
         "title": "SHAP, PFI and ALE", "blocks": (
            {"text": "Three measures each rank the chosen detectors by how much "
                     "weight they carry, and each answers a different question. "
                     "All three read the same rows, the test split the ensemble "
                     "was finally scored on, with one detector's score column at"
                     "a time under examination."},
            {"text": "SHAP measures how much the meta-learner's anomaly "
                     "probability moves when a detector's actual output is "
                     "revealed in place of its median output, averaged over "
                     "every combination of the other detectors being revealed or "
                     "held at their medians. Because the ensemble is small, "
                     "every combination is enumerated exactly rather than "
                     "sampled. Enumerating them costs 2 to the power of the "
                     "ensemble size per row, so the rows are what has to be kept "
                     "in check: SHAP is measured on a fixed sample of 200 test "
                     "rows, a size at which the ranking was verified stable, "
                     "rather than multiplying that exponential term by the "
                     "length of the whole split."},
            {"text": "PFI measures how far the run's fitness falls when "
                     "a detector's score column is shuffled, so unlike SHAP and "
                     "ALE it uses the labels and reports reliance on the detector "
                     "rather than influence on the output."},
            {"text": "ALE cuts a detector's observed score range into 10 bins "
                     "holding roughly equal numbers of rows. Each bin is handled "
                     "on its own and only with the rows whose score for that "
                     "detector actually falls inside it. Those rows are put "
                     "through the meta-learner twice: once with this detector's "
                     "score forced down to the bin's lower edge, once forced up "
                     "to its upper edge, with every other detector left at the "
                     "value it really had. The bin's effect is the average "
                     "difference between the two anomaly probabilities, so it "
                     "reads as what happens locally when this detector alone "
                     "reports higher. Accumulating those effects across the bins "
                     "gives the curve the explanation draws. Because each bin only ever "
                     "asks about rows that occur there, the meta-learner is never "
                     "questioned about a combination of scores the data does not "
                     "contain. PFI and ALE both use every row of the split."},
            {"text": "The three measures are magnitudes: a detector is ranked on "
                     "the average size of its SHAP value, mean |SHAP|, and on the "
                     "total size of its accumulated ALE effect, total |ALE|, "
                     "which adds the bins up without letting a rise in one cancel "
                     "a fall in another. Because the three might disagree, they "
                     "are merged by the same kind of Markov rank aggregation the "
                     "single-model branch uses, giving one overall weight "
                     "ranking."},
        )},
        {"id": "ga-direction", "title": "The sign and how well it is supported",
         "blocks": (
            {"text": "A detector's sign is the direction of ALE's total "
                     "accumulated effect: positive if a rising score pushes the "
                     "meta-learner toward flagging the point as an anomaly, "
                     "negative if towards flagging normal. How well that sign is "
                     "supported is reported separately, because two things can "
                     "undermine it: an effect that changes direction across the "
                     "score range, or a detector that barely moves the "
                     "meta-learner at all. Both are named rather than allowed to "
                     "suppress the sign, since a blank reads as missing data "
                     "when the measurement was actually made."},
        )},
    )},
    {"id": "lints", "title": "Linear Thompson Sampling",
     "stages": ("thompson_ranking", "thompson_sampling"), "blocks": (
        {"text": "LinTS treats detector choice as a contextual bandit. Each "
                 "detector is an arm, each window of the series is a round, "
                 "pulling an arm means running that detector on that window and "
                 "observing how well it did. The time-series is split into "
                 "windows in the beginning."},
    ), "subsections": (
        {"id": "lints-bayesian", "title": "The context vector and the posterior", "blocks": (
            {"text": "Every window t is turned into a context vector cₜ (its "
                     "readings over the window's timesteps, one block of "
                     "entries per context feature). A detector's reward is "
                     "modelled as linear in that context: there is a weight "
                     "vector θ that turns cₜ into the reward the detector earns "
                     "on that window, and no detector knows it. What each one "
                     "holds instead is a Gaussian belief about θ, written "
                     "θ ∼ 𝒩(μₜ, Σₜ). Its mean μₜ is the detector's current best "
                     "estimate of θ after t windows, and its covariance Σₜ is "
                     "how unsure it still is. Both start at μ₀ = 0 and Σ₀ = I, "
                     "a ridge prior that keeps early updates stable."},
            {"text": "Three quantities follow, and this branch keeps them "
                     "apart because they are not interchangeable. All three "
                     "are the same product of a weight vector with the same "
                     "context, and they differ only in which weight vector "
                     "goes into it."},
            {"list": (
                "The expected reward, θᵀcₜ, is what the detector earns on this "
                "window on average. It is the quantity the model is defined by, "
                "and it can never be computed, because θ is unknown.",
                "The estimated expected reward, μₜᵀcₜ, puts the belief's mean in "
                "place of θ. It is the best estimate of the expected reward "
                "available after everything observed so far, and it is the "
                "number the Selection explanation plots and explains.",
                "The sampled expected reward, θ̃ᵀcₜ, puts a single weight "
                "vector θ̃ drawn from 𝒩(μₜ, Σₜ) in place of θ. It is the "
                "expected reward that would hold if that draw had no uncertainty, "
                "and it is what the next subsection picks on.",
            )},
            {"text": "Two consequences are worth stating plainly. First, the "
                     "estimated expected reward is an estimate and not the "
                     "exact θᵀcₜ , though it sharpens as evidence accumulates:"
                     "the more often a detector is tried, the tighter Σₜ "
                     "becomes and the closer μₜᵀcₜ sits to θᵀcₜ. "
                     "Second, the prior starts at μ₀ = 0 and μ only moves in "
                     "windows where that detector was actually run, so a "
                     "detector that has rarely been tried has an estimated "
                     "expected reward near zero whatever its true expected "
                     "reward is. A low value therefore means either a poor "
                     "detector or a barely tested one, and the number alone "
                     "does not separate the two."},
        )},
        {"id": "lints-round", "title": "Choosing a detector for a window", "blocks": (
            {"text": "With probability ε the framework picks a detector "
                     "uniformly at random. Otherwise it draws one weight vector "
                     "θ̃ from every detector's own 𝒩(μₜ, Σₜ) and picks the "
                     "detector with the highest sampled expected reward θ̃ᵀcₜ. "
                     "The pick therefore runs on the draw, not on the estimated "
                     "expected reward the explanation plots, and that is what makes it "
                     "uncertainty-aware: a detector that has rarely been tried "
                     "has a wide covariance and can win on a favourable draw, so "
                     "the run keeps testing plausible alternatives instead of "
                     "locking onto an early leader. Where the two disagree is "
                     "recorded, and is what the run reports as informed "
                     "exploration."},
        )},
        {"id": "lints-reward", "title": "The reward and the posterior update", "blocks": (
            {"text": "The chosen detector is scored over the whole window, "
                     "against labels that carry the synthetic anomalies injected "
                     "before model selection began, and rewarded with the run's "
                     "fitness, the weighted metrics chosen at run configuration. "
                     "Only the chosen detector's belief is updated, by Bayesian "
                     "linear regression on the pair (cₜ, r): the covariance "
                     "absorbs cₜcₜᵀ and the mean moves toward the observed "
                     "reward. The detectors that were not picked learn nothing "
                     "from this window."},
        )},
        {"id": "lints-epsilon", "title": "How exploration decays", "blocks": (
            {"text": "ε starts at 0.2 and is annealed after every window, so the "
                     "run begins broad and narrows toward exploiting what it has "
                     "learned."},
        )},
        {"id": "lints-output", "title": "The final ranking by ‖μ‖²", "blocks": (
            {"text": "The offline phase runs until every window is processed. "
                     "Afterwards the detectors are ranked by the overall size of "
                     "their final mean vector, ‖μ‖². This ranking is the branch's "
                     "output and is what goes into the final aggregation."},
        )},
        {"id": "lints-two-views", "title": "μₜᵀcₜ versus ‖μ‖²", "blocks": (
            {"text": "The branch produces two different numbers, which is why it "
                     "has two cards. μₜᵀcₜ is the estimated expected reward at a "
                     "given window: the branch's best guess at what that detector "
                     "would earn there, computed for every detector whether or "
                     "not it was picked, and moving as the series moves. It is "
                     "not what the pick was made on, since that was the draw, but "
                     "the draw is centred on it, so it is the yardstick the "
                     "choice is read against."},
            {"text": "‖μ‖² is the accumulated ranking score. It is context-free, "
                     "and it changes only in windows where that detector was "
                     "picked. The update "  
                     "moves μ toward the reward just observed, so a reward below "
                     "what the detector predicted pulls μ back toward zero and "
                     "the score down with it. A detector can therefore lead the "
                     "estimated-expected-reward view for much of the run and "
                     "still not finish first in the ranking."},
        )},
        {"id": "lints-note", "title": "Why contexts are normalised", "blocks": (
            {"text": "Context vectors are normalised to unit length before use. "
                     "Without it, a series with large sensor values or many "
                     "context features (SMD carries 38) makes cₜcₜᵀ dominate "
                     "the covariance update and collapse Σ."},
        )},
        {"id": "lints-explained", "title": "What the two sections explain",
         "blocks": (
            {"text": "Both numbers are explained, and they are explained "
                     "differently because they answer different questions. The "
                     "Ranking explanation takes the final ‖μ‖² and splits it across the "
                     "context features, so the question it answers is what a "
                     "detector's standing at the end of the run was built from. "
                     "The Selection explanation works window by window: it splits μₜᵀcₜ "
                     "the same way, adds a measure of how unusual each context "
                     "feature's contribution was at that window, divides the run "
                     "into stretches under one leader, and classifies every pick "
                     "the sampler made. The question it answers is how the "
                     "competition actually unfolded."},
        )},
        {"id": "lints-criterion", "title": "Splitting the score across context features", "blocks": (
            {"text": "The score ‖μ‖² is a sum of squared weights, so it splits "
                     "exactly into one number per context feature with no "
                     "approximation: a context feature's contribution to a "
                     "detector is the sum of the squares of that detector's "
                     "mean-vector weights for that context feature, and the "
                     "contributions add up to the whole score. "
                     "Its share is that contribution as a fraction of the score. "
                     "Because these are squares they are never negative, so a "
                     "small share means a context feature added little, not "
                     "that it worked against the detector. Comparing two "
                     "detectors does have direction: the difference of their "
                     "contributions, taken one context feature at a time, sums "
                     "exactly to the margin (difference) between their scores, "
                     "and a negative term marks a context feature "
                     "the rival was stronger on. The stage also reports how "
                     "often each detector was actually selected, because μ only "
                     "moves in windows where the arm was pulled, so the score "
                     "reflects exposure as well as quality. A streak is a "
                     "stretch of consecutive windows where one detector held "
                     "the highest score, read straight off the leader at each "
                     "window after a short warm-up. The warm-up lasts as many "
                     "windows as there are detectors in the pool, because each "
                     "detector starts with score zero."},
            {"formula":
                "contribution(k, p)     = Σ_{i ∈ p} μ_k[i]²\n"
                "Σ_p contribution(k, p) = ‖μ_k‖²\n"
                "margin(a, b, p)        = contribution(a, p) − contribution(b, p)\n"
                "Σ_p margin(a, b, p)    = ‖μ_a‖² − ‖μ_b‖²"},
            {"text": "where k, a and b are detectors, p is a context feature, and i runs "
                     "over the entries of that detector's final mean vector μ that "
                     "belong to context feature p."},
        )},
        {"id": "lints-dynamics", "title": "The reward split and the selection states", "blocks": (
            {"text": "This section explains the estimated expected reward μₜᵀcₜ, "
                     "the branch's best guess at what each detector would earn "
                     "on each window. The pick itself was made on a draw rather "
                     "than on this number, but the draw is centred on it, so it "
                     "is what every pick is weighed against. It splits per "
                     "context feature with the same exactness the ranking score "
                     "does, and those contributions sum to the prediction "
                     "exactly, so a detector's estimated expected reward can be "
                     "read as a list of what each context feature put into it."},
            {"text": "One difference from the ranking score matters. That split "
                     "is a sum of squares and so can never be negative, while "
                     "this one multiplies a weight by a context entry and either "
                     "can be negative. A context feature can therefore pull a "
                     "detector's estimated expected reward down on a given "
                     "window, and a negative contribution here means exactly "
                     "that, where a small share on the Ranking explanation only ever "
                     "meant a small positive one."},
            {"formula":
                "contribution(k, p)     = Σ_{i ∈ p} μ_k,ₜ[i] · cₜ[i]\n"
                "Σ_p contribution(k, p) = μ_k,ₜᵀcₜ"},
            {"text": "where k is a detector, p is a context feature, cₜ is the "
                     "window's context vector, μ_k,ₜ is detector k's mean vector "
                     "as it stood at window t, and i runs over the entries "
                     "belonging to context feature p."},
            {"text": "A second and narrower measure is reported beside it: SHAP, "
                     "how far a context feature's contribution at this window "
                     "departs from what that context feature contributes at the "
                     "average window of the run. The two answer different "
                     "questions. A context feature can supply most of a "
                     "detector's reward and still depart from its own norm not "
                     "at all, which is the case where the detector is being "
                     "carried by something entirely ordinary."},
            {"text": "The run is then divided into regimes. Each detector's "
                     "estimated expected reward is first smoothed with a rolling "
                     "mean over five windows, so that a single unusual window "
                     "cannot hand the lead over on its own, and the leader is "
                     "read off the smoothed series. A regime is a stretch of at "
                     "least three consecutive windows in which one detector held "
                     "that lead. Both the smoothing and the three-window minimum "
                     "are there to separate a real change of leader from noise, "
                     "and a change of lead that survives neither is recorded as a "
                     "blip instead."},
            {"text": "These regimes are not the same thing as the previous "
                     "stage's streaks. Streaks segment the cumulative ranking "
                     "score and are read off it directly, while regimes segment "
                     "the per-window estimated expected reward after smoothing "
                     "and impose a minimum length. The two quantities and the two "
                     "methods both differ, so the two need not line up."},
            {"text": "Finally, every choice the sampler made is classified, so "
                     "the run can be read as behaviour and not only as an "
                     "outcome. Exploitation means the detector picked was the "
                     "one with the highest estimated expected reward μₜᵀcₜ, so "
                     "the draw agreed with the current best guess. Informed "
                     "exploration means the highest sampled expected reward "
                     "θ̃ᵀcₜ belonged to some other detector, so uncertainty "
                     "rather than that best guess decided it. Random exploration means "
                     "the ε step fired and the pick was made without consulting "
                     "either. A run that is mostly random exploration is one "
                     "where ε had not yet decayed. A run that is mostly informed "
                     "exploration is one where the detectors stayed closely "
                     "matched and the beliefs uncertain."},
        )},
    )},
    {"id": "gan", "title": "GAN perturbation test", "stages": ("gan",), "blocks": (
        {"text": "The GAN test asks whether a detector has learned the structure "
                 "of normality or merely memorised its training data. It builds "
                 "anomalies that mimic realistic drifts while preserving "
                 "temporal structure, and a detector that only recognises the "
                 "patterns it was trained on fails on them."},
        {"lead": "Architecture and training.",
         "text": "A generator G and a discriminator D are instantiated as "
                 "two-layer multi-layer perceptrons with 256 hidden units, ReLU "
                 "activations and dropout for regularisation. The generator maps "
                 "G: R^d -> R^d with a tanh output layer, while the "
                 "discriminator maps D: R^d -> [0,1]. Both are optimised with "
                 "binary cross-entropy losses and Adam optimizers at a learning rate of "
                 "1e-4, for 100 epochs in mini-batches. Label smoothing on the "
                 "real and fake targets, and Gaussian noise added to both, keep "
                 "the training stable."},
        {"lead": "Data preparation.",
         "text": "GAN training uses the clean, non-augmented, split to avoid leakage. "
                 "Augmentation occurs after training, during robustness testing. "
                 "To match the generator's tanh output layer, inputs are linearly "
                 "rescaled to [-1,1] and the generated points are "
                 "mapped back before they enter the series."},
        {"lead": "Injection.",
         "text": "After training, a candidate pool is drawn from the generator, "
                 "each candidate scored by the discriminator, and each one's "
                 "ambiguity measured as its distance from the decision "
                 "threshold tau that separates normal from anomalous:"},
        {"formula":
            "C          = { x_k = G(z_k) }, k = 1..K,   z_k ~ N(0, I)\n"
            "delta_k    = | D(x_k) - tau |\n"
            "X*_B       = the B candidates minimising delta_k"},
        {"text": "The B most ambiguous candidates are kept. These are the "
                 "borderline cases: plausible enough to belong to the series, "
                 "but sitting where \"normal\" and \"anomalous\" are hardest to "
                 "separate. The same boundary supplies each kept point's label, "
                 "so the injected set contains both near-normal and "
                 "near-anomalous behaviour:"},
        {"formula": "y(x) = 1 [ D(x) >= tau ]"},
        {"lead": "Temporal integration.",
         "text": "To respect chronology the selected points are interleaved into "
                 "the stream at regular intervals within sliding windows, at an "
                 "injection budget rho of about 10% of the original number of "
                 "samples. Integration preserves ordering, updates the label "
                 "mask, and records the injection indices for traceability and "
                 "for the figures."},
        {"lead": "Evaluation and ranking.",
         "text": "Every detector is then re-evaluated on the GAN-augmented "
                 "series and ranked by the run's own fitness. That ranking feeds "
                 "the robustness consensus. A detector missing any term of the "
                 "fitness is ranked last rather than scored on the terms that "
                 "survived, which would compare it against the others on a "
                 "different quantity."},
    ), "subsections": (
        {"id": "gan-explained", "title": "Exclusive wins and the surrogate trees",
         "blocks": (
            {"text": "The perturbation here happens at the level of individual "
                     "points, so the explanation does too, and it is built the "
                     "Rather than "
                     "re-running the test under different settings, it reuses "
                     "the single production run and asks, per injected point, "
                     "what kind of generated point the winner handles correctly "
                     "that a given rival does not."},
            {"text": "For the winning detector and each other detector in turn, "
                     "an exclusive win is an injected point the winner "
                     "classified correctly and that rival did not. A small "
                     "decision tree is fitted to predict those exclusive wins "
                     "from seven properties of the point, none of which depends "
                     "on any detector. Writing x for the generated point, x_c "
                     "for its value in feature c across the point's d "
                     "features, W for the "
                     "window of the real series around the injection site, i for "
                     "the index the point was injected at and N for the length "
                     "of the augmented series:"},
            {"lead": "ambiguity.",
             "text": "How far the discriminator's score for the point sits from "
                     "the threshold tau. It is the same quantity the injection "
                     "step minimised, so 0 marks a point the discriminator "
                     "found maximally hard to place."},
            {"formula": "ambiguity(x) = | D(x) - tau |"},
            {"lead": "is_anomaly.",
             "text": "The label the discriminator's verdict gave the point: 1 "
                     "anomalous, 0 normal. The same boundary that selected it."},
            {"formula": "is_anomaly(x) = 1 [ D(x) >= tau ]"},
            {"lead": "signal_magnitude.",
             "text": "The average size of the generated values across the "
                     "injected point's features, which is how large the "
                     "injected reading is, irrespective of sign."},
            {"formula": "signal_magnitude(x) = (1/d) * sum_c | x_c |"},
            {"lead": "signal_spread.",
             "text": "How much those values differ from one another across "
                     "the injected point's features. A low spread is a "
                     "point that moved every feature together, a high one "
                     "is a point that moved them apart."},
            {"formula": "signal_spread(x) = std_c ( x_c )"},
            {"lead": "context_gap.",
             "text": "How far the generated point sits from the average of the "
                     "real series around it, averaged over the injected "
                     "point's features."},
            {"formula": "context_gap(x) = (1/d) * sum_c | x_c - mean(W_c) |"},
            {"lead": "local_volatility.",
             "text": "The standard deviation of the real series in that same "
                     "neighbourhood, averaged over the injected point's "
                     "features, which is how noisy the stretch the point "
                     "landed in already was."},
            {"formula": "local_volatility = (1/d) * sum_c std( W_c )"},
            {"lead": "position.",
             "text": "Where the point falls in the series, from 0 at the start "
                     "to 1 at the end."},
            {"formula": "position = i / N"},
            {"text": "The discriminator's raw score is deliberately not an "
                     "eighth property. The ambiguity and the label already "
                     "determine it between them, so carrying it as well would "
                     "split the importance across three descriptions of one "
                     "fact. The tree's splits become plain rules, and the "
                     "average importance across all the rival trees shows which "
                     "property best explains the winner's edge."},
            {"lead": "Importance.",
             "text": "Each surrogate is a depth-3 decision tree, and its "
                     "importances say how that tree spent its splits. A "
                     "property's importance is the total drop in Gini impurity "
                     "across every node that splits on it, each node weighted "
                     "by how many points reach it, normalised so the properties "
                     "sum to 1. A property the tree never splits on scores 0."},
            {"formula": "imp(f) = SUM over nodes n splitting on f of\n"
                        "             (N_n / N) * [ G(n) - (N_L/N_n) G(L) - (N_R/N_n) G(R) ]\n"
                        "\n"
                        "G(n) = 1 - p(n)^2 - (1 - p(n))^2"},
            {"list": (
                "f is one of the properties above.",
                "n is a node of the tree, and L and R are its left and right "
                "children.",
                "N is the number of injected points the tree was fitted on. "
                "N_n, N_L and N_R are how many of those reach n, L and R.",
                "G(n) is the Gini impurity at n, where p(n) is the share of "
                "exclusive wins among the points reaching it.",
            )},
            {"text": "Each rival gets its own tree and therefore its own "
                     "importances. The figure reports the mean across the "
                     "rivals whose comparison was not degenerate."},
            {"text": "Three limits follow from how it is computed. It is "
                     "relative within one tree, so a property's 0.55 says it "
                     "outweighed the others there, not that it matters in any "
                     "absolute sense. It describes the tree that was fitted, "
                     "not the mechanism: two properties that move together "
                     "split the credit between them arbitrarily, and at depth 3 "
                     "there are at most seven splits to go round. Additionally, "
                     "impurity-based importance favours properties that can be "
                     "cut in many places over ones that cannot: is_anomaly is "
                     "binary while position and local_volatility are "
                     "continuous, so the binary property starts at a "
                     "disadvantage against them."},
            {"text": "Each tree also reports a held-out, cross-validated "
                     "accuracy alongside the accuracy on the points it was "
                     "fitted to, because the second can look strong purely from "
                     "memorising a small set of wins. A comparison resting on "
                     "fewer exclusive wins than there are cross-validation folds "
                     "is flagged, since the held-out estimate is not stable "
                     "there. Only the prediction side is explained: correctness "
                     "is defined by thresholded predictions, and PR-AUC has no "
                     "per-point notion of right or wrong."},
        )},
    )},
    {"id": "off-by", "title": "Off-by-threshold test",
     "stages": ("off_by_threshold",), "blocks": (
        {"text": "Off-by-threshold tests ask where a detector draws the line "
                 "between \"just unusual\" and \"anomalous\", by manufacturing "
                 "points that sit almost exactly on the decision threshold."},
    ), "subsections": (
        {"id": "off-by-idea", "title": "Why borderline points are the hard case", "blocks": (
            {"text": "A point far outside the normal range is easier to detect. "
                     "A point barely outside of the usual series behaviour is "
                     "the hard case, and it is also the common case. This test "
                     "builds those points deliberately, with known labels, and "
                     "checks which detectors classify them correctly."},
        )},
        {"id": "off-by-build", "title": "Building a point from local variation", "blocks": (
            {"text": "About one point is inserted for every ten already in the "
                     "series, at evenly spaced positions. To build the point at "
                     "a given position, the framework looks at a short stretch "
                     "of the series around it and measures how much each feature "
                     "of the series varies there, its local standard "
                     "deviation. That number is "
                     "the series' own account of how much variation is ordinary "
                     "at this spot, and the injected point is drawn at roughly "
                     "that size, multiplied by a random factor drawn just above "
                     "or just below one."},
            {"text": "Scaling by local variation rather than by a fixed amount "
                     "is what makes this a boundary test. A fixed absolute "
                     "deviation would be unremarkable in a noisy stretch and "
                     "glaring in a quiet one, so a fixed perturbation would be "
                     "trivial to detect in some places and invisible in others. "
                     "Using the local variation puts every injected point at the "
                     "same relative distance from normal, independent from the "
                     "position in time-series."},
        )},
        {"id": "off-by-label", "title": "How the random factor sets the label", "blocks": (
            {"text": "The random factor decides. At or below 1, the point varies "
                     "no more than its neighbourhood does and is labelled "
                     "normal. Above 1, it exceeds the local norm and is labelled "
                     "a statistical border anomaly. Because the factor only ever "
                     "lands slightly either side of one, both classes sit right "
                     "at the boundary, which is the whole point of the test."},
        )},
        {"id": "off-by-measured", "title": "Re-ranking on the augmented series", "blocks": (
            {"text": "Every detector is re-evaluated on the augmented series and "
                     "ranked by the run's own fitness, contributing one ranking to "
                     "the robustness consensus. The augmented data is used nowhere else in the "
                     "framework. The test needs at least 100 points and both "
                     "classes present in the original labels, and is skipped "
                     "rather than run on data that cannot support it."},
        )},
        {"id": "off-by-explained", "title": "Exclusive wins and the surrogate trees",
         "blocks": (
            {"text": "The perturbation here happens at the level of individual "
                     "points, so the explanation does too. Rather than "
                     "re-running the test under different settings, it reuses "
                     "the single production run and asks, per injected point, "
                     "what kind of borderline point the winner handles that a "
                     "given rival does not."},
            {"text": "For the winning detector and each other detector in turn, "
                     "an exclusive win is an injected point the winner "
                     "classified correctly and that rival did not. A small "
                     "decision tree is fitted to predict those exclusive wins "
                     "from four properties of the point, none of which depends "
                     "on any detector. Writing s for the random factor drawn for "
                     "the point, W for the contextual window of the real series "
                     "around the injection site over the series' d features, i "
                     "for the index "
                     "the point was injected at and N for the length of the "
                     "augmented series:"},
            {"lead": "boundary_distance.",
             "text": "How far the random factor landed from 1, and therefore how "
                     "far off the threshold the point sits. A 0 marks a point "
                     "drawn exactly at the boundary."},
            {"formula": "boundary_distance(x) = | s - 1 |"},
            {"lead": "is_anomaly.",
             "text": "Whether the point was injected as an anomaly or as normal: "
                     "1 anomalous, 0 normal. The same factor that placed it."},
            {"formula": "is_anomaly(x) = 1 [ s > 1 ]"},
            {"lead": "local_volatility.",
             "text": "The standard deviation of the real series in the "
                     "neighbourhood the point landed in, averaged over the "
                     "injected point's features, which is how noisy that "
                     "stretch already was."},
            {"formula": "local_volatility = (1/d) * sum_c std( W_c )"},
            {"lead": "position.",
             "text": "Where the point falls in the series, from 0 at the start "
                     "to 1 at the end."},
            {"formula": "position = i / N"},
            {"text": "The tree's splits become plain rules, and the average "
                     "importance across all the rival trees shows which property "
                     "best explains the winner's edge."},
            {"lead": "Importance.",
             "text": "Each surrogate is a depth-3 decision tree, and its "
                     "importances say how that tree spent its splits. A "
                     "property's importance is the total drop in Gini impurity "
                     "across every node that splits on it, each node weighted "
                     "by how many points reach it, normalised so the properties "
                     "sum to 1. A property the tree never splits on scores 0."},
            {"formula": "imp(f) = SUM over nodes n splitting on f of\n"
                        "             (N_n / N) * [ G(n) - (N_L/N_n) G(L) - (N_R/N_n) G(R) ]\n"
                        "\n"
                        "G(n) = 1 - p(n)^2 - (1 - p(n))^2"},
            {"list": (
                "f is one of the properties above.",
                "n is a node of the tree, and L and R are its left and right "
                "children.",
                "N is the number of injected points the tree was fitted on. "
                "N_n, N_L and N_R are how many of those reach n, L and R.",
                "G(n) is the Gini impurity at n, where p(n) is the share of "
                "exclusive wins among the points reaching it.",
            )},
            {"text": "Each rival gets its own tree and therefore its own "
                     "importances. The figure reports the mean across the "
                     "rivals whose comparison was not degenerate."},
            {"text": "Three limits follow from how it is computed. It is "
                     "relative within one tree, so a property's 0.55 says it "
                     "outweighed the others there, not that it matters in any "
                     "absolute sense. It describes the tree that was fitted, "
                     "not the mechanism: two properties that move together "
                     "split the credit between them arbitrarily, and at depth 3 "
                     "there are at most seven splits to go round. Additionally, "
                     "impurity-based importance favours properties that can be "
                     "cut in many places over ones that cannot: is_anomaly is "
                     "binary while position and local_volatility are "
                     "continuous, so the binary property starts at a "
                     "disadvantage against them."},
            {"text": "Each tree also reports a held-out, cross-validated "
                     "accuracy alongside the accuracy on the points it was "
                     "fitted to, because the second can look strong purely from "
                     "memorising a small set of wins. A comparison resting on "
                     "fewer exclusive wins than there are cross-validation folds "
                     "is flagged, since the held-out estimate is not stable "
                     "there. Only the prediction side is explained: correctness "
                     "is defined by thresholded predictions, and PR-AUC has no "
                     "per-point notion of right or wrong."},
        )},
    )},
    {"id": "monte-carlo", "title": "Monte Carlo simulation",
     "stages": ("monte_carlo",), "blocks": (
        {"text": "In Monte Carlo, zero mean Gaussian noise is added to every "
                 "channel at every timestep, and every detector is re-evaluated "
                 "on an independent draw of that noise. The labels do not "
                 "change, the ground truth stays exactly as it was and only the "
                 "signal degrades, so this measures whether a detector's "
                 "standing survives a noisier version of the same series. The "
                 "run does this five times. Within one trial every detector is "
                 "scored on the same draw, so the trial is a fair comparison. "
                 "Across trials the draw changes, so a detector that happens to "
                 "win one trial does not necessarily rank high on the final ranking. "
                 "The scores across trials are averaged, and the detectors are ranked "
                 "by that average. The final Monte Carlo ranking is forwarded to the "
                 "robustness aggregation."},
        {"text": "The noise level is the standard deviation of the injected "
                 "Gaussian noise, and it is the same in every trial."},
    ), "subsections": (
        {"id": "mc-explained", "title": "Reading the ranking from its trials", "blocks": (
            {"text": "The explanation is built from the same five trials the "
                     "ranking averages, not from a separate experiment, so every "
                     "number in it is one the ranking was actually computed "
                     "from. It answers why the top detector leads on three "
                     "counts: which components of the fitness function its advantage came "
                     "from, whether it led the individual trials rather than "
                     "only their average, and whether first place survives "
                     "dropping any single trial and re-ranking on the remaining "
                     "four."},
        )},
        {"id": "mc-counts", "title": "How the three are computed", "blocks": (
            {"text": "The first is a term-by-term comparison between the top "
                     "detector and the runner-up. If the run's fitness is "
                     "0.5 × F1 + 0.3 × PR-AUC + 0.2 × VUS, each of those three "
                     "averages is compared on its own, which separates a top "
                     "detector that led on every term from one whose lead on a "
                     "single term covered a shortfall on another."},
            {"text": "The second records which detector scored highest in each "
                     "of the five trials. A detector can hold the best average "
                     "without having led many trials, and the count makes that "
                     "visible."},
            {"text": "The third leaves one trial out, averages the fitness over "
                     "the remaining four and re-ranks on that average. If the "
                     "detector at the top changes, first place rested on the "
                     "trial that was dropped; each of the five trials is dropped "
                     "in turn, so every one of them is checked."},
        )},
        {"id": "mc-spread", "title": "How far apart the ranked detectors are", "blocks": (
            {"text": "Averaging five trials produces a strict order of every "
                     "detector, but the trials do not always completely agree. The "
                     "explanation therefore reports two numbers side by side: how far "
                     "a detector's fitness moves between trials, and the average "
                     "fitness difference between two neighbouring places in the final "
                     "Monte Carlo ranking. Both are medians over the detectors, "
                     "because a few large gaps at the bottom of a ranking would "
                     "otherwise make the whole order look well separated."},
            {"text": "Where the movement is the larger of the two, the places "
                     "next to each other in the ranking are closer together than "
                     "the noise that produced them. The full explanation adds, "
                     "for each detector, the highest and lowest place it reached "
                     "across the trials."},
        )},
    )},
    {"id": "rank-aggregation", "title": "Rank aggregation",
     "stages": ("rank_aggregation_robust", "rank_aggregation_final"), "blocks": (
        {"text": "The single-model branch produces four orderings of the same "
                 "detectors: one from each of the three robustness tests and one "
                 "from Thompson sampling. They may disagree, and each is about a "
                 "different thing, so none can simply be preferred. Aggregation "
                 "turns them into one consensus."},
    ), "subsections": (
        {"id": "agg-method", "title": "From pairwise counts to a consensus", "blocks": (
            {"text": "For every pair of detectors, count how many of the input "
                     "rankings put i ahead of j. Normalise those counts into a "
                     "row-stochastic transition matrix, giving a random walk "
                     "over detectors that is more likely to move toward a "
                     "detector that beats the current one more often. Compute "
                     "the stationary distribution of that walk by power "
                     "iteration, and sort the detectors by it. A detector scores "
                     "highly when the detectors that beat it are themselves "
                     "rarely preferred, so the result reflects the whole "
                     "structure of agreement rather than a count of first "
                     "places."},
        )},
        {"id": "agg-twice", "title": "The two aggregation stages", "blocks": (
            {"text": "First to the three robustness rankings, one per test, "
                     "producing the robustness consensus. Then to that consensus "
                     "together with the Thompson ranking, producing the final "
                     "single-model order. The two-stage shape is deliberate: it "
                     "keeps the three robustness tests from outvoting the "
                     "Thompson ranking three to one."},
        )},
        {"id": "agg-explained", "title": "Influence, agreement and Borda", "blocks": (
            {"text": "The consensus is one ordering built from several possibly "
                     "disagreeing ones, and the question the explanation answers "
                     "is which sources shaped it the most."},
            {"text": "Each source gets two scores. Its influence is how far the "
                     "consensus moves when that source is dropped and the "
                     "aggregation is re-run without it. Its agreement is how "
                     "closely the source's own ranking matches the consensus "
                     "that was produced. The two are independent, and the "
                     "interesting cases are where they diverge."},
            {"formula": "influence(s)  = (1 − τ(R, R₋ₛ)) / 2        ∈ [0, 1]\n"
                        "agreement(s)  = τ(rₛ, R)                    ∈ [−1, 1]"},
            {"text": "where R is the consensus, R₋ₛ the consensus re-aggregated "
                     "without source s, rₛ that source's own ranking, and τ is "
                     "Kendall's tau."},
            {"text": "Each score induces its own ranking of the sources, and "
                     "those two rankings are then treated as two voters and "
                     "merged by Borda count into a single overall rank. That "
                     "rank is the verdict, and it is what \"shaped the "
                     "consensus most, second most, and so on\" reports."},
            {"formula": "borda(s) = (N − rank_influence(s)) + (N − rank_agreement(s))"},
            {"text": "where N is the number of ranking sources being merged."},
        )},
        {"id": "agg-two-source", "title": "Why the final stage uses agreement alone",
         "blocks": (
            {"text": "Where only two rankings are merged, for example in the "
                     "final aggregation, leave-one-out is degenerate, since "
                     "dropping one leaves the other unchanged, and Borda voting "
                     "over two sources says nothing. The final aggregation is "
                     "therefore explained by agreement alone: which of the two "
                     "the consensus leans toward, and by how much."},
        )},
    )},
)

DOC_SECTION_BY_STAGE = {key: section["id"]
                        for section in DOC_SECTIONS for key in section["stages"]}

# One line per term, shown on the stage card in the space the glossary used to
# fill. Ordered pairs rather than a dict so the reader meets the terms in the
# order the card's prose uses them, not alphabetically.
#
# These define the stage's vocabulary and are the same for every run, which is
# why they live here rather than in the IR: nothing about them depends on what a
# run found. The long-form glossary on /docs stays where it is, written by
# `Explainability.ir` into each stage's artifacts.
STAGE_TERMS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ga_selection": (
        ("Utility", "The average change in an ensemble's fitness when the "
                    "detector is added to it."),
        ("Stability", "The share of the subsets the algorithm evaluated that "
                      "included the detector."),
    ),
    "ga_combination": (
        ("SHAP", "How much the meta-learner's anomaly probability moves when a "
                 "detector's actual output is revealed in place of its median "
                 "output."),
        ("PFI", "How far fitness drops when that detector's scores are "
                "shuffled; label-based."),
        ("ALE", "The change in the meta-learner's anomaly probability as the "
                "detector's own score is swept across its range."),
        ("Sign", "The direction of that change: positive means the probability "
                 "rises, negative means it falls."),
    ),
    "thompson_ranking": (
        ("Score", "The overall size ‖μ‖² of a detector's final learned mean vector μ."),
        ("Share", "The fraction of that score that came from a single context feature."),
        ("Contribution", "The exact amount a single context feature added to that score."),
        ("Margin", "The gap between two detectors' scores; it traces back to each "
                   "context feature, whose contributions sum to it exactly."),
        ("Streak", "A stretch of consecutive windows in which one detector held "
                   "the highest score ‖μ‖²."),
    ),
    "thompson_sampling": (
        ("Expected reward", "θᵀcₜ, is what the detector earns on this window on "
                            "average. θ ∼ 𝒩(μₜ, Σₜ) is unknown."),
        ("Estimated expected reward", "μₜᵀcₜ, is the best estimate of the expected "
                                      "reward available after everything observed "
                                      "so far."),
        ("Sampled expected reward", "θ̃ᵀcₜ, puts a single weight vector θ̃ drawn "
                                    "from 𝒩(μₜ, Σₜ) in place of θ."),
        ("Regime", "At least three consecutive windows in which one detector held "
                   "the highest estimated expected reward."),
        ("Exploitation", "The detector with the highest estimated expected reward was "
                         "picked."),
        ("Informed exploration", "The highest sampled expected reward belonged to "
                                 "a different detector than the highest estimated "
                                 "expected reward."),
        ("Random exploration", "A forced exploration step fired, so the pick was "
                               "completely random."),
        ("SHAP", "How far a context feature's contribution to the estimated expected "
                 "reward departed from its average contribution over the run."),
    ),
    "gan": (
        ("Exclusive win", "A point the top-ranked detector classified correctly "
                          "and the named rival did not."),
        ("Ambiguity", "How far the discriminator's score for the generated point "
                      "sits from the threshold separating normal from anomalous; "
                      "0 is the hardest point to place."),
        ("Signal magnitude", "The average size of the injected point's values "
                             "across its features."),
        ("Signal spread", "How much the injected point's values differ from one "
                          "another across its features."),
        ("Context gap", "How far the generated point sits from the average of the "
                        "real series around it."),
        ("Local volatility", "The standard deviation of the series around the "
                             "injection site; how noisy that neighbourhood is."),
        ("Position", "Where the point falls in the series, from 0 at the start "
                     "to 1 at the end."),
        ("Importance", "How much of the tree's separation between exclusive "
                       "wins and the rest rests on that property, from 0 (never "
                       "split on) to 1 (all of it)."),
    ),
    "monte_carlo": (
        ("Noise level", "The standard deviation of the Gaussian noise injected "
                        "into the data, the same in every trial."),
        ("Trial", "One draw of that noise, on which every detector is scored."),
    ),
    "off_by_threshold": (
        ("Exclusive win", "A point the top-ranked detector classified correctly "
                          "and the named rival did not."),
        ("Boundary distance", "How far the point was scaled away from the "
                              "decision boundary; 0 sits exactly on it."),
        ("Local volatility", "The standard deviation of the series around the "
                             "injection site; how noisy that neighbourhood is."),
        ("Position", "Where the point falls in the series, from 0 at the start "
                     "to 1 at the end."),
        ("Importance", "How much of the tree's separation between exclusive "
                       "wins and the rest rests on that property, from 0 (never "
                       "split on) to 1 (all of it)."),
    ),
    # Both consensus cards get both: they are one stage, and the vocabulary is
    # the stage's. A two-source aggregation reports no influence — leave-one-out
    # is undefined there — but defining the word costs nothing and the card's
    # own prose is what says whether it applied.
    "rank_aggregation_robust": (
        ("Influence", "How much the aggregated ranking changes when that source "
                      "is left out."),
        ("Agreement", "How similar the source's own ranking is to the aggregated "
                      "ranking."),
    ),
}
STAGE_TERMS["rank_aggregation_final"] = STAGE_TERMS["rank_aggregation_robust"]

# Qualifies the terms list where every number in it is a position rather than a
# score, which reads backwards without saying so.
TERMS_NOTES: Dict[str, str] = {
    "ga_combination": "Lower rank = higher score",
    "rank_aggregation_robust": "Lower rank = higher score",
}

def split_info(raw: str) -> Tuple[Optional[str], str]:
    """`"INFO: glossary\\n\\nnarrative"` -> `("glossary", "narrative")`.

    Legacy: the glossary is no longer written — the documentation page carries
    it — but files from earlier runs still lead with the marker, and without
    this they would render as narrative. Anything without it is all narrative.
    """
    if raw is None:
        return None, ""
    text = raw.replace("\r\n", "\n").strip("\n")
    if not text.startswith("INFO:"):
        return None, text.strip()
    body = text[len("INFO:"):]
    info, sep, narrative = body.partition("\n\n")
    if not sep:
        # Glossary with no narrative after it.
        return info.strip(), ""
    return info.strip(), narrative.strip()


def _read_text(path: Path) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _newest(pattern_dir: Path, pattern: str) -> Optional[Path]:
    """Newest file matching `pattern` inside `pattern_dir`, by mtime.

    Iteration numbers are NOT consistent across artifact trees: the
    comprehensive report uses the CLI --iteration (default 5) while the
    explanations use OFFLINE_ITERATION = 0, and both can coexist in one
    directory. Always glob and take the newest; never derive the index.
    """
    matches = glob.glob(os.path.join(glob.escape(str(pattern_dir)), pattern))
    if not matches:
        return None
    return Path(max(matches, key=lambda p: os.path.getmtime(p)))


# A narrative is stale once its IR has been rewritten under it. This happens
# for real: `--stages X --explain` is a PARTIAL run, and app.py returns before
# the narration block, so it regenerates the IR and every plot but leaves the
# prose alone. The page then shows fresh figures beside sentences describing a
# run that no longer exists — on SKAB/7 the narrative walked 14 regimes while
# the IR and the plots had 10. Nothing else notices, because the two files are
# read independently. One second of slack absorbs same-run write ordering.
_STALE_SLACK_SECONDS = 1.0


def _mtime(path: Optional[Path]) -> Optional[float]:
    try:
        return path.stat().st_mtime if path is not None else None
    except OSError:
        return None


def _load_stage_files(ir_dir: Optional[Path], nl_dir: Optional[Path],
                      stage: Dict[str, Any]) -> Tuple[Optional[dict], Optional[str],
                                                      Optional[Path], bool]:
    """(ir_doc, raw_nl_text, nl_path, narrative_is_stale) for one stage."""
    ir_doc = raw = None
    ir_path = nl_path = None
    if ir_dir is not None:
        exact = ir_dir / f"{stage['ir']}.json"
        ir_path = exact if exact.exists() else _newest(ir_dir, f"{stage['ir']}*.json")
        if ir_path is not None:
            ir_doc = _read_json(ir_path)
    if nl_dir is not None:
        exact = nl_dir / f"{stage['nl']}.txt"
        nl_path = exact if exact.exists() else _newest(nl_dir, f"{stage['nl']}*.txt")
        if nl_path is not None:
            raw = _read_text(nl_path)
    ir_at, nl_at = _mtime(ir_path), _mtime(nl_path)
    stale = bool(ir_at and nl_at and nl_at + _STALE_SLACK_SECONDS < ir_at)
    return ir_doc, raw, nl_path, stale


def load_global_ir(dataset: str, entity: str) -> Optional[dict]:
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    if ir_dir is None:
        return None
    path = _newest(ir_dir, "ir_global_iter*.json")
    return _read_json(path) if path else None


def load_faithfulness(dataset: str, entity: str) -> Optional[dict]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    path = _newest(nl_dir, "faithfulness_iter*.json")
    return _read_json(path) if path else None


def global_text_path(dataset: str, entity: str) -> Optional[Path]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    return _newest(nl_dir, "nl_global_iter*.txt")


_ITER_RE = re.compile(r"_iter(\d+)\.txt$")


def comprehensive_path(dataset: str, entity: str) -> Optional[Path]:
    """Newest `comprehensive_results_*.txt` for this entity, or None.

    Written by the pipeline itself, not by the explainability layer, so it
    exists for runs made without `--explain` and is absent after a partial run
    (`app.py` returns before the report step). Its iteration index comes from
    `--iteration` (default 5) rather than the explanations' OFFLINE_ITERATION,
    which is exactly why this globs instead of building the filename.
    """
    report_dir = paths.resolve_entity_dir(paths.COMPREHENSIVE, dataset, entity)
    if report_dir is None:
        return None
    return _newest(report_dir, "comprehensive_results_*.txt")


def comprehensive_info(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Metadata for the report, without reading it — the page links, not inlines."""
    path = comprehensive_path(dataset, entity)
    if path is None:
        return None
    match = _ITER_RE.search(path.name)
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "name": path.name,
        "iteration": int(match.group(1)) if match else None,
        "bytes": stat.st_size,
        "generated_at": stat.st_mtime,
        "url": f"/report/{dataset}/{entity}",
        "download_url": f"/api/comprehensive/{dataset}/{entity}?download=1",
    }


def comprehensive_report(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """`comprehensive_info` plus the report text itself."""
    info = comprehensive_info(dataset, entity)
    if info is None:
        return None
    path = comprehensive_path(dataset, entity)
    return {**info, "text": _read_text(path) or "",
            "dataset_label": dataset_label(dataset)}


# Matches both Thompson stages' regime atoms — `ts.regime.N` (expected-reward
# regimes) and `tsr.regime.N` (||mu||^2 leadership regimes). Anchored on the
# suffix rather than the prefix so a third producer needs no change here.
_REGIME_RE = re.compile(r"\.regime\.(\d+)$")


def _regimes_from_ir(ir_doc: dict) -> List[Dict[str, Any]]:
    """Regime atoms, ordered, ready to pair with their per-regime plots.

    The ids are 0-based and match `regime_{NN}_w{start}-{end}_{model}.png`
    exactly, so each regime sentence can be shown beside its own plot instead of
    the reader hunting through fourteen images.
    """
    out = []
    for atom in ir_doc.get("evidence", []) or []:
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if not m:
            continue
        value = atom.get("value") or {}
        out.append({
            "index": int(m.group(1)),
            "start": value.get("start"),
            "end": value.get("end"),
            "duration": value.get("duration"),
            "leader": value.get("leader") or atom.get("subject"),
            "text": atom.get("text", ""),
        })
    return sorted(out, key=lambda r: r["index"])


def _attach_narrated_regimes(regimes: List[Dict[str, Any]], narrative: str,
                             ir_doc: dict) -> None:
    """Give each regime the sentence the model wrote about it.

    The per-regime disclosure used to show the IR's own atom text — correct but
    flat, and a second rendering of facts the narrative already covers. The
    narrated sentences are pulled out of the same paragraph the summary drops,
    so nothing is generated twice and nothing is lost by hiding them from the
    default view. `text` stays as the fallback when a regime's sentence cannot
    be located.
    """
    by_index: Dict[int, str] = {}
    for sentence, atom in attribute_sentences(narrative, ir_doc):
        if not atom or atom.get("type") != "regime":
            continue
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if m:
            idx = int(m.group(1))
            # A regime can span two sentences; keep them in narrative order.
            by_index[idx] = (by_index.get(idx, "") + " " + sentence.strip()).strip()
    for regime in regimes:
        narrated = by_index.get(regime.get("index"))
        if narrated:
            regime["narrated"] = narrated


def _headline_pick(output: Dict[str, Any]) -> Optional[str]:
    """The one detector a stage put first, whatever the stage calls that key."""
    # `top_pick_f1` is Monte Carlo's key on result trees written before the
    # stage published a single fitness ranking.
    for key in ("top_pick", "winner", "top_pick_f1"):
        value = output.get(key)
        if isinstance(value, str) and value and value != "not_available":
            return value
    return None


def _stage_faithfulness(report: Optional[dict], key: str) -> Optional[dict]:
    if not report:
        return None
    entry = (report.get("stages") or {}).get(key)
    if not entry:
        return None
    verify = entry.get("verify") or {}
    return {
        "status": entry.get("status"),
        "words": entry.get("words"),
        "hallucination_rate": verify.get("hallucination_rate"),
        "omission_rate": verify.get("omission_rate"),
        "n_claims": verify.get("n_claims"),
        "n_required": verify.get("n_required"),
        "repaired": bool(entry.get("repaired")),
    }


def build_payload(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The `/api/explanations/<ds>/<ent>` response, or None if nothing exists."""
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if ir_dir is None and nl_dir is None:
        return None

    global_ir = load_global_ir(dataset, entity)
    faith = load_faithfulness(dataset, entity)
    gtext = global_text_path(dataset, entity)

    stages_out: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for stage in STAGES:
        ir_doc, raw, nl_path, stale = _load_stage_files(ir_dir, nl_dir, stage)
        if ir_doc is None and raw is None:
            continue
        _, narrative = split_info(raw or "")
        summary = summarize(narrative, stage=stage["key"], ir_doc=ir_doc)
        output = (ir_doc or {}).get("output") or {}
        entry: Dict[str, Any] = {
            "key": stage["key"],
            "title": stage["title"],
            "order": stage["order"],
            "status": "ok" if narrative else "no_narrative",
            # Stages name their headline result differently: `top_pick` for
            # most, `winner` for off-by, `top_pick_f1` for Monte Carlo (which
            # ranks on two metrics). GA selection has no single pick — it
            # chooses a set — so None there is correct, not a gap.
            "top_pick": _headline_pick(output),
            "summary": summary["summary"],
            "summary_is_full": summary["is_full"],
            "summary_mode": summary["mode"],
            "summary_table": summary.get("table"),
            "extended_in": summary.get("extended_in"),
            # What the full-text disclosure shows. Usually the whole narrative;
            # a stage that renders some of its sentences elsewhere on the card
            # (Thompson's regime walk, beside its per-regime plots) hands back a
            # trimmed body so the page never prints them twice. `words` and the
            # download stay on the real narrative — the file on disk is the
            # verbatim record, and the length is what the model actually wrote.
            # `body` is that narrative minus any sentence restating a caveat:
            # the card lists the caveats verbatim below, so the disclosure must
            # not be a second, looser copy of them.
            "full": summary.get("extended") or summary.get("body") or narrative,
            "words": len(narrative.split()) if narrative else 0,
            # The glossary lives on the documentation page; the card keeps the
            # one-line definitions and a pointer to the section holding it.
            "terms": [list(pair) for pair in STAGE_TERMS.get(stage["key"], ())],
            "terms_note": TERMS_NOTES.get(stage["key"]),
            "doc_section": DOC_SECTION_BY_STAGE.get(stage["key"]),
            "question": (ir_doc or {}).get("question"),
            "output": output,
            "caveats": [c.get("text") for c in ((ir_doc or {}).get("caveats") or [])],
            "faithfulness": _stage_faithfulness(faith, stage["key"]),
            "plot_group": stage["plot_group"],
            "nl_file": nl_path.name if nl_path else None,
            # The narrative predates its own IR: the numbers on this card
            # may describe a previous run. Surfaced rather than silently
            # rendered, and cleared by re-running the narrator.
            "stale": stale,
        }
        if stage.get("regimes") and ir_doc:
            entry["regimes"] = _regimes_from_ir(ir_doc)
            entry["regime_noun"] = stage.get("regime_noun", "Regime")
            _attach_narrated_regimes(entry["regimes"], narrative, ir_doc)
        stages_out.append(entry)

    # Stages the global IR knows about but that produced no narrative — anything
    # a partial run skipped, or a run without --explain. Carry the IR's own note
    # so the UI states a reason instead of showing an unexplained gap.
    present = {s["key"] for s in stages_out}
    for key, info_block in sorted(((global_ir or {}).get("stages") or {}).items()):
        if key in present:
            continue
        meta = STAGE_BY_KEY.get(key)
        missing.append({
            "key": key,
            "title": (meta or {}).get("title", key.replace("_", " ").capitalize()),
            "plot_group": (meta or {}).get("plot_group"),
            "status": info_block.get("status", "not_available"),
            "note": info_block.get("note"),
        })

    decision = (global_ir or {}).get("decision") or {}
    # The final consensus IS the source of the single-model pick, so comparing
    # them always reports agreement and says nothing. Newer runs no longer emit
    # the row; filtering here means older result trees read correctly too.
    agreement = [
        {"source": name,
         "top_pick": info.get("top_pick"),
         # Absent on result trees written before rankings were carried; the
         # frontend simply renders no disclosure for those.
         "ranking": list(info.get("ranking") or []),
         "stage": info.get("stage") or name,
         "metric": info.get("metric"),
         "agrees": info.get("agrees_with_final_single")}
        # `order` is the display order the IR laid out: one row per metric, the
        # robustness stages in the same columns on every row. Trees written
        # before it existed have none, and fall back to their key order.
        for name, info in sorted(
            ((global_ir or {}).get("stage_agreement") or {}).items(),
            key=lambda kv: (kv[1].get("order") is None, kv[1].get("order"), kv[0]))
        if name != "final_consensus"
    ]
    decision_atom = next(
        (a.get("text") for a in ((global_ir or {}).get("evidence") or [])
         if a.get("id") == "global.decision"), None)

    return {
        "dataset": dataset,
        # The page header shows this rather than the directory name: the run
        # tree is keyed "Anomaly_Archive" where every reader calls it UCR.
        "dataset_label": dataset_label(dataset),
        "entity": entity,
        "iteration": (global_ir or {}).get("iteration"),
        # No global IR but a global .txt on disk means an older result tree:
        # serve the raw text, never try to parse it.
        "degraded": global_ir is None,
        "decision": decision,
        "decision_text": decision_atom,
        "agreement": agreement,
        "stages": sorted(stages_out, key=lambda s: s["order"]),
        "missing_stages": missing,
        "faithfulness": (faith or {}).get("overall"),
        "model": (faith or {}).get("model"),
        "global_text": gtext.name if gtext else None,
        "generated_at": (os.path.getmtime(gtext) if gtext else None),
        # Kept beside the explanation but never merged into it: the report is
        # the pipeline's own record of what happened, in its own numbers.
        "comprehensive": comprehensive_info(dataset, entity),
    }


def documentation(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The stage documentation: one section per pipeline stage.

    The text describes RAMSeS itself and is the same for every run, so it is
    served straight from DOC_SECTIONS. The route stays per-entity because the
    page carries a link back to the entity the reader came from, and because
    the stage cards address it that way.

    Returns None when the entity has no explanations at all, which is the one
    thing that would make that back-link point at nothing.
    """
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if ir_dir is None and nl_dir is None:
        return None
    def _blocks(source):
        # Lists and formulas are tuples in the registry; JSON wants arrays, and
        # a copy keeps a caller from mutating the module-level text.
        out = []
        for block in source:
            item = dict(block)
            if "list" in item:
                item["list"] = list(item["list"])
            out.append(item)
        return out

    sections = [{"id": s["id"], "title": s["title"],
                 "blocks": _blocks(s["blocks"]),
                 "subsections": [{"id": sub["id"], "title": sub["title"],
                                  "blocks": _blocks(sub["blocks"])}
                                 for sub in s.get("subsections", ())]}
                for s in DOC_SECTIONS]
    return {"dataset": dataset, "entity": entity, "sections": sections}


def entity_summary(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Compact card for the "previous results" list — no narrative loading."""
    global_ir = load_global_ir(dataset, entity)
    gtext = global_text_path(dataset, entity)
    if global_ir is None and gtext is None:
        return None
    faith = load_faithfulness(dataset, entity)
    decision = (global_ir or {}).get("decision") or {}
    return {
        "dataset": dataset,
        "dataset_label": dataset_label(dataset),
        "entity": entity,
        "framework_choice": decision.get("framework_choice"),
        "chosen": decision.get("chosen"),
        "n_stages": len([s for s in ((global_ir or {}).get("stages") or {}).values()
                         if s.get("status") == "ok"]),
        "hallucination_rate": ((faith or {}).get("overall") or {}).get("hallucination_rate"),
        "omission_rate": ((faith or {}).get("overall") or {}).get("omission_rate"),
        "generated_at": os.path.getmtime(gtext) if gtext else None,
    }


def known_entities() -> List[Tuple[str, str]]:
    """(dataset, entity) pairs that have explanation artifacts on disk."""
    found = []
    for root in (paths.EXPLANATIONS_NL, paths.EXPLANATIONS_IR):
        if not root.is_dir():
            continue
        for ds_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not ds_dir.is_dir():
                continue
            for ent_dir in sorted(ds_dir.iterdir(), key=lambda p: paths.natural_key(p.name)):
                if ent_dir.is_dir():
                    pair = (ds_dir.name, ent_dir.name)
                    if pair not in found:
                        found.append(pair)
    return found
