# RAMSeS — Robust & Adaptive Model Selection for Time-Series Anomaly Detection

**RAMSeS** is a research framework for **unsupervised time-series anomaly detection (TSAD)** that automatically selects and deploys the best detection strategy from two complementary approaches:

1. **Optimized Ensemble** — A stacking ensemble discovered by a Genetic Algorithm (GA) with configurable meta-learners (RF/LR/GBM/SVM)
2. **Robust Single Model** — Selected via Linear Thompson Sampling with comprehensive robustness testing

The framework provides an end-to-end pipeline for data loading, model training, robustness evaluation, and adaptive online selection.

> **Research Group:** D2IP @ TU Berlin  
> **License:** Apache 2.0  
> **Status:** Research code (actively maintained)

---

## ✨ Key Features

### Dual Selection Strategy
- **Ensemble Branch:** Genetic Algorithm optimizes detector combinations; meta-learner stacks anomaly scores
- **Single-Model Branch:** Multi-criteria evaluation combining:
  - Linear Thompson Sampling with ε-greedy exploration
  - GAN-based robustness testing (borderline synthetic anomalies)
  - Off-by-threshold sensitivity analysis
  - Monte Carlo noise stress tests
  - Markov-chain rank aggregation

### Online Adaptation
- Sliding window processing with configurable update intervals
- Multiple strategies: adaptive re-optimization, fixed-best, or fixed-random
- Optional regime shift injection for testing distribution changes
- Real-time model selection updates

### Comprehensive Evaluation
- Automatic comparison of ensemble vs. single-model performance
- Detailed performance tracking (F1, PR-AUC, fitness)
- Memory and computational overhead monitoring


---

## 📁 Repository Structure

```text
RAMSeS/
├── app.py                              # Main pipeline entrypoint
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
├── LICENSE                             # Apache 2.0 license
│
├── Algorithms/                         # Base anomaly detector implementations
│   ├── abod.py                         # Angle-Based Outlier Detection
│   ├── alad.py                         # Adversarial Learned Anomaly Detection
│   ├── anomaly_transformer.py          # Transformer-based detector
│   ├── cblof.py                        # Cluster-Based Local Outlier Factor
│   ├── cof.py                          # Connectivity-Based Outlier Factor
│   ├── dghl.py                         # Deep Generative Hierarchical Learning
│   ├── kde.py                          # Kernel Density Estimation
│   ├── lof.py                          # Local Outlier Factor
│   ├── lstmvae.py                      # LSTM Variational Autoencoder
│   ├── mean_deviation.py               # Mean Deviation
│   ├── nearest_neighbors.py            # k-Nearest Neighbors
│   ├── rnn.py                          # Recurrent Neural Network
│   ├── running_mean.py                 # Running Mean
│   └── sos.py                          # Stochastic Outlier Selection
│
├── Configs/                            # Configuration files
│   ├── config.yml                      # Default configuration
│   └── custom_config.yml               # Custom configuration template
│
├── Datasets/                           # Data loading utilities
│   ├── dataset.py                      # Dataset class definitions
│   └── load.py                         # Data loader implementations
│
├── Metrics/                            # Evaluation and GA components
│   ├── Ensemble_GA.py                  # Genetic Algorithm for ensemble optimization
│   ├── metrics.py                      # Performance metrics (F1, PR-AUC, etc.)
│   └── ranking_metrics.py              # Ranking evaluation utilities
│
├── Model_Selection/                    # Model selection algorithms
│   ├── Thompson_Sampling.py            # Linear Thompson Sampling + sliding windows
│   ├── rank_aggregation.py             # Markov-chain rank aggregation
│   ├── inject_anomalies.py             # Synthetic anomaly injection
│   └── Sensitivity_robustness/
│       ├── GAN_test.py                 # GAN-based robustness testing
│       ├── Monte_Carlo_Simulation.py   # Monte Carlo noise stress tests
│       └── off_by_threshold_testing.py # Borderline sensitivity analysis
│
├── Model_Training/                     # Model training and management
│   ├── train.py                        # Training orchestration
│   ├── trainer.py                      # Training logic
│   └── hyperparameter_grids.py         # Hyperparameter configurations
│
├── Utils/                              # Utility functions
│   ├── utils.py                        # CLI argument parsing, misc utilities
│   ├── config.py                       # Configuration file parser
│   ├── plotting.py                     # Visualization utilities
│   └── results_formatter.py            # Results formatting and export
│
└── testbed/                            # Batch evaluation configurations
    └── file_list/                      # CSV files listing datasets for batch runs
        ├── test_single.csv             # Single dataset test
        ├── test_m_skab.csv             # SKAB multivariate datasets
        └── test_u_ucr_anomaly_archive.csv  # UCR univariate datasets
```

---




## 📦 Datasets

RAMSeS uses the **Mononito** time-series repository ([arXiv:2210.01078](https://arxiv.org/abs/2210.01078)), which includes:

- **SKAB** — Multivariate sensor data from industrial systems
- **SMD** — Server Machine Dataset (multivariate)
- **UCR Anomaly Archive** — Univariate time series

### Download Instructions

1. **Download the Mononito dataset** from Google Drive:  
   [https://drive.google.com/drive/folders/1BLcaGm4bNSBueh3Hy_-dP1MKNhzfulwC?usp=share_link](https://drive.google.com/drive/folders/1BLcaGm4bNSBueh3Hy_-dP1MKNhzfulwC?usp=share_link)

2. **Extract and organize:**
   ```bash
   # Example structure
   ~/Mononito/
   ├── datasets/
   │   ├── SKAB/
   │   ├── SMD/
   │   └── UCR/
   └── trained_models/  # Created during training
   ```

3. **Update configuration** in `Configs/config.yml`:
   ```yaml
   dataset_path: "/path/to/Mononito/datasets"
   trained_model_path: "/path/to/Mononito/trained_models"
   ```

> **Licensing Note:** Please follow the original dataset licenses and cite the appropriate papers when using these datasets.

---


## ⚙️ Installation

### Prerequisites
- Python 3.9+ (tested with Python 3.11)
- CUDA-capable GPU (optional, CPU mode supported)
- 8GB+ RAM recommended

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Maxoud99/RAMSeS.git
cd RAMSeS

# Create and activate conda environment
conda env create -f environment.yml
conda activate ..

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Option 2: Pip + Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Maxoud99/RAMSeS.git
cd RAMSeS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Key Dependencies
- **PyTorch** 2.5+ (with optional CUDA support)
- **TensorFlow** 2.18+ (CPU version)
- **scikit-learn** 1.7+ (ML algorithms)
- **PyOD** 2.0+ (outlier detection)
- **NumPy**, **Pandas**, **Matplotlib** (data processing & visualization)
- **loguru** (structured logging)

See `requirements.txt` for the complete dependency list.

---

## 🚀 Quick Start

### 1. Configure Paths

Edit `Configs/config.yml` to set your dataset and model paths:

```yaml
# Paths
dataset_path: "/path/to/Mononito/datasets"
trained_model_path: "/path/to/Mononito/trained_models"

# Training parameters
downsampling: 10
min_length: 256
training_size: 1.0
model_architectures: 'all'  # or specify: 'LOF,CBLOF,NN'

# Evaluation parameters
normalize: True
evaluation_metric: 'Best F-1'
```

### 2. Train Base Models

RAMSeS automatically trains missing models on first run:

```bash
python app.py \
  --config Configs/config.yml \
  --dataset SKAB \
  --entity 5
```

This will:
- Load training data from the specified dataset/entity
- Train any missing base detectors (LOF, CBLOF, NN, etc.)
- Save trained models to `trained_model_path/SKAB/5/`

### 3. Run Model Selection

Execute the complete RAMSeS pipeline:

```bash
python app.py \
  --config Configs/config.yml \
  --dataset SKAB \
  --entity 5 \
  --parallel false
```

**Pipeline stages:**
1. Load training and test data
2. Train/load base detector models
3. Inject synthetic anomalies
4. Run Genetic Algorithm for ensemble optimization
5. Execute Thompson Sampling for single model selection
6. Perform robustness tests (GAN, Monte Carlo, Off-by-Threshold)
7. Aggregate rankings and select final model

### 4. View Results

Results are saved in human-readable format with comprehensive metrics and visualizations.

**Example output:**
```
FRAMEWORK FINAL DECISION
================================================================================

Single Model Option:
--------------------------------------------------
  Model       : LOF_2
  F1 Score    : 0.8542
  PR-AUC      : 0.7891

Ensemble Option:
--------------------------------------------------
  Models      : ['LOF_1', 'CBLOF_3', 'NN_2']
  Size        : 3
  Meta-Model  : rf
  F1 Score    : 0.8721
  PR-AUC      : 0.8103
  Fitness     : 0.8412

Final Choice:
--------------------------------------------------
  ✓ ENSEMBLE SELECTED
    Reason: Ensemble F1 (0.8721) >= Single Model F1 (0.8542)
    Improvement: +0.0179 (2.10%)
```

---

## 🎯 Configuration Options

### Command-Line Arguments

```bash
python app.py \
  --config <path>           # Path to config file (default: Configs/config.yml)
  --dataset <name>          # Dataset name (e.g., SKAB, SMD, UCR)
  --entity <id>             # Entity ID within dataset
  --parallel <true|false>   # Enable parallel model selection
```

### Online Mode (Adaptive Selection)

Enable online learning and adaptive model selection:

```bash
python app.py \
  --config Configs/config.yml \
  --dataset SKAB \
  --entity 5 \
  --enable_online \
  --update_interval 5 \
  --iteration 5 \
  --strategy adaptive \
  --inject_online_regime \
  --max_online_windows 100
```

**Online mode parameters:**
- `--enable_online` — Enable online phase processing
- `--update_interval <n>` — Re-optimize every n windows (default: 5)
- `--iteration <n>` — Number of iterations for window sizing (default: 5)
- `--strategy <mode>` — Selection strategy:
  - `adaptive` — Re-optimize model selection periodically
  - `fixed-best` — Use best offline model throughout
  - `fixed-random` — Use random model (baseline)
- `--inject_online_regime` — Inject regime shifts in online data for testing
- `--max_online_windows <n>` — Limit number of online windows processed

---

## 🔍 Thompson Sampling Explainability

RAMSeS includes an explainability layer for the Thompson Sampling branch that tracks **how model competition evolves across windows** and automatically identifies **regime shifts** — sustained periods where one model holds the highest expected reward.

### What it tracks

At every window `t`, the expected reward for model `k` is:

```
E[reward | model_k, context_t] = mu_k^T * x_t
```

where `mu_k` is the posterior mean vector and `x_t` is the standardised data window. This is computed for **all** models after each posterior update, not just the selected one.

Additionally, the context-free belief strength `||mu_k||^2` is tracked per window to show how accumulated reward information grows over time.

### How to enable

Pass `explain=True` to `run_linear_thompson_sampling()`:

```python
from Model_Selection.Thompson_Sampling import run_linear_thompson_sampling

run_linear_thompson_sampling(
    test_data, trained_models, model_names,
    dataset="SMD", entity="machine-1-1",
    iterations=50, iteration=1,
    explain=True,   # <-- enable explainability
)
```

### Output files

Two files are written per run to `myresults/Thomposon/{dataset}/{entity}/` (same directory as the standard outputs):

| File | Contents |
|---|---|
| `expected_rewards_{iterations}.png` | Line plot of every model's smoothed expected reward over windows. Regime regions are shaded by dominant model; dashed vertical lines mark regime shifts. Legend is placed outside the plot area for readability. |
| `explainability_{iterations}.txt` | Structured report: per-window table, regime summary, shift event table, blip list, and final model ranking (final `||mu_k||^2` and peak `||mu_k||^2`). |

### Interpreting results

**Expected reward plot:**
- The x-axis is the window index (time); the y-axis is `mu_k^T * x_t`, which can be negative early in training when posteriors are uninformative.
- Shaded regions show which model consistently led for a sustained period.
- Dashed lines mark confirmed regime shifts; short fluctuations that do not persist are recorded in the report as *blips* and not drawn on the plot.

**Regime shift detection:**
- A *regime* is a sustained period where one model holds the highest smoothed expected reward.
- A *regime shift* is recorded when the new dominant model persists for at least 3 consecutive windows (configurable via `min_regime_length`).
- A *blip* is a transient dominance change shorter than `min_regime_length`; it is logged in the report but not treated as a true shift.
- Smoothing is applied with a rolling mean (default window = 5) before dominance is computed, suppressing single-window noise.

**Key concepts:**
- `mu_k^T * x_t` — contextual expected reward; reflects what the agent believes model `k` will earn on the current window.
- `||mu_k||^2` — context-free belief strength; grows as the model accumulates high-reward observations regardless of the current window.
- Values near zero indicate an uninformative posterior (common in early windows before significant updates accumulate).

**Note on output paths:** All outputs (standard and explainability) share the same `myresults/Thomposon/` directory.

### Selection State Categorization

Each model selection at every window is classified into one of three **behavioral states** so you can see *why* a model was picked, not just which one:

| State | Trigger | Meaning |
|---|---|---|
| `random` | ε-greedy random branch fired | Exploration floor — independent of beliefs. |
| `exploitation` | TS-chosen and equal to `argmax_k (mu_k^T * x_t)` | Agent picked what its current posterior mean already says is best. |
| `informed_exploration` | TS-chosen but differs from the mean-based argmax | Posterior uncertainty (covariance) steered the sampled `theta_tilde` away from the mean-best guess. |

Classification happens **immediately after `sample_model()` returns, using the pre-update means** — the same beliefs that informed the decision. The ε-greedy flag takes precedence: a random pick that happens to match the argmax is still labeled `random`.

**New output file** (in the same `myresults/Thomposon/{dataset}/{entity}/` directory):

| File | Contents |
|---|---|
| `selection_states_{iterations}.png` | Two-panel figure: a per-window coloured timeline strip (red=random, green=exploitation, blue=informed_exploration) on top and a bar chart of total counts with percentage labels on the bottom. Legend placed outside the plot area. |

**Report additions** in `explainability_{iterations}.txt`:
- A new `State` column on the per-window summary table.
- A `--- Selection State Summary ---` section listing counts and percentages of each state.

**Interpreting state distributions:**
- **Many `random`** → ε hasn't decayed enough yet; the agent is still on the exploration floor.
- **Many `exploitation`** → posteriors have converged; the agent is confident and not deviating from its mean-best guess.
- **Many `informed_exploration`** → posterior uncertainty is still steering decisions; the agent is actively probing models whose mean reward looks lower but whose covariance is wide.

### Numerical Stability Fixes

The bandit math underneath the explainability layer was hardened with three fixes (ported from upstream commit `df299b1`):

- **Sherman-Morrison posterior update.** `update_posteriors()` no longer calls `np.linalg.inv` twice per step. Instead it uses the rank-1 Sherman-Morrison identity
  `Σ_new = Σ − (Σx)(Σx)ᵀ / (1 + xᵀΣx)`, with the matching mean update
  `μ_new = μ + (Σx)(r − xᵀμ) / (1 + xᵀΣx)`.
  This eliminates `SVD did not converge` crashes on high-dimensional datasets (SMD, etc.) and fixes a Python aliasing bug in which `old_precision = precision` shared a reference with `precision`, so the in-place `precision += outer(x, x)` corrupted the mean update.
- **L2-normalised context and features.** Both the `context` passed to `sample_model` and the `features` passed to `update_posteriors` are now divided by their L2 norm before use. This prevents `xxᵀ` from exploding on datasets with large sensor magnitudes and keeps `Σ` from collapsing.
- **Hard error on feature-shape mismatch.** The silent truncate/pad branch in `fit_linear_thompson_sampling` was replaced with a `ValueError`; alignment of the explainability tracking lists is preserved through the existing exception handler which NaN-pads `_exp_rewards_hist` and `_l2_norm_hist`.

**Effect on explainability outputs:** the y-axis of `expected_rewards_{iterations}.png` now lands in a moderate range (O(0.01–1)) instead of being dominated by raw sensor magnitudes, and the selection-state distribution reflects genuine bandit behaviour rather than numerical noise pinning the agent on a single model.

### SHAP Feature Attribution

To explain *why* a detector's predicted reward takes the value it does — and *why one detector is preferred over another* at a given context — RAMSeS decomposes each detector's expected reward into per-feature SHAP contributions using the **final posterior mean vectors** `μ_k`.

For the linear bandit model `E[R_k] = μ_k^T x`, SHAP values have a closed analytical form (this is exactly what `shap.LinearExplainer(..., feature_dependence="independent")` returns — no `shap` library dependency is added):

```
φ_0 = μ_k^T · E[X]               (baseline expected reward)
φ_i = μ_k[i] · (x_i − E[X_i])    (per-feature attribution)
φ_0 + Σ_i φ_i = μ_k^T x          (additivity)
```

**Choice of baseline and explanation context:**
- **Baseline `E[X]`** = mean of all L2-normalised context windows seen during training (the "average context").
- **Explanation context `x`** = the last L2-normalised window (the most recent decision point).

Per-feature SHAP `φ_i = μ_k[i] · (x_i − E[X_i])` therefore answers: *"How did each feature at the last window push the predicted reward away from what was typical?"*

**Per-channel aggregation:** the flattened feature vector has `d = n_channels × window_size` entries — raw per-feature attributions are not interpretable. We aggregate per channel by summing each channel's window-of-timesteps slice, so the outputs are reported in terms of sensor channels.

**Cross-model preference decomposition:** to explain *why* model A is preferred over model B at the last context, we decompose the preference gap with the same baseline:

```
φ_i^{A−B} = (μ_A[i] − μ_B[i]) · (x_i − E[X_i])
```

Summed per channel, this surfaces the sensors that most strongly drive the preference between the two detectors.

**New output files** (in `myresults/Thomposon/{dataset}/{entity}/`):

| File | Contents |
|---|---|
| `shap_per_model_{iterations}.png` | One horizontal-bar panel per top-K model showing the top-N channels by `\|per-channel SHAP\|`. Bars are coloured by sign (green > 0, red < 0). The panel title shows `E[R \| last]` for that model. |
| `shap_comparison_{iterations}.png` | Grouped bar chart comparing the top-K models on the union of their top-N channels. Reveals which sensors disagree the most between detectors. Legend placed outside the plot area. |

**New report sections** in `explainability_{iterations}.txt`:
- `--- SHAP Feature Attribution (final mean vectors) ---` — per-model breakdown with `E[R \| last]`, baseline `E[R]`, delta, and top-5 channels by `\|per-channel SHAP\|`.
- `--- SHAP Preference Decomposition ---` — top model vs runner-up, preference gap `(μ_top − μ_2nd)^T x_last`, and the top-5 channels driving the preference (with each model's own SHAP value and the delta).

---

## 🧠 Rank Aggregation Explainability

The Markov-chain rank aggregator fuses several ranking sources into the single ordering that drives the framework's final model choice. The explainability layer quantifies *how much each source mattered* and *whether each source agreed with the consensus*, then settles disagreements with Borda count.

The aggregation runs in two stages:

| Stage | Sources | Output |
|---|---|---|
| **Robust** (6 inputs) | `GAN_F1`, `GAN_PR_AUC`, `Borderline_F1`, `Borderline_PR_AUC`, `MonteCarlo_F1`, `MonteCarlo_PR_AUC` | the "robust" consensus ranking |
| **Final** (2 inputs) | `Robust_Aggregated`, `Thompson_Sampling` | the framework's final ranking |

Each source receives two diagnostic scores, then a resolution rank from Borda count voting:

| Score | Range | High value means |
|---|---|---|
| **LOO contribution** | `[0, 1]` | Removing this source moved the consensus a lot — high marginal influence. Computed as normalised Kendall distance `(1 − τ)/2` between the full aggregation and the leave-one-out aggregation. |
| **Kendall τ alignment** | `[−1, +1]` | This source's own ranking agrees with the consensus. |
| **Borda count** | `[0, 2(N−1)]` | Borda points awarded by the joint vote (see below). |

### Borda count voting as the default arbiter

The LOO contributions induce one ranking of the sources (most-to-least pivotal). The Kendall τ alignments induce another (most-to-least agreeing). These two rankings are treated as **two voters in a Borda count election**: a source at position `r` in a ranking of `N` sources receives `(N − r)` points from that voter, so

```
borda_count(source) = (N − loo_rank) + (N − align_rank)
```

Ranking sources by descending Borda count produces a single **resolved ranking** that settles disagreements between the LOO and Kendall views. The resolved ranking IS the verdict — the `borda_rank` column in the report shows it.

Each source also gets a descriptive pattern label:

| Pattern label | LOO rank vs Align rank | Typical interpretation |
|---|---|---|
| `influential_outlier` | LOO ≪ Alignment (high marginal influence but low agreement) | The source pushed the consensus *away* from its own ranking — it shaped the result by being a dissenting voice that others partially counteracted. |
| `redundant_agreer` | LOO ≫ Alignment (low marginal influence but high agreement) | The source agrees with the consensus but didn't change it — its view was already covered by others. |
| `consistent` | LOO ≈ Alignment | LOO and Kendall tell the same story. |

A separate **Prominent Contradictions** section highlights the sources whose LOO rank and Kendall rank differ most sharply — that's where the Borda-resolved rank is most informative.

### Output files

For each aggregation stage and iteration, two files are written to `myresults/robust_aggregated/{dataset}/{entity}/`:

| File | Contents |
|---|---|
| `aggregation_explainability_{stage}_{iteration}.png` | Grouped bar chart: three bars per source (LOO contribution, Kendall τ rescaled to `[0, 1]`, normalised Borda count). Legend placed outside the plot area. |
| `aggregation_explainability_{stage}_{iteration}.txt` | Structured report: per-source scores, per-source ranks with pattern label, the Borda-resolved source ranking, and prominent contradictions. |

`{stage}` is `robust` (Stage 1, six sources) or `final` (Stage 2, two sources).

### How to read it

- A source with **high LOO + high Kendall** drove the consensus toward its own ranking.
- A source with **high LOO + low Kendall** dissented loudly enough to bend the consensus away from itself (influential outlier).
- A source with **low LOO + high Kendall** rubber-stamped a consensus that would have formed without it (redundant).
- A source with **low LOO + low Kendall** is essentially noise — neither moved nor agreed with the result.

The feature is on by default (`explain=True`) and follows the same toggle convention as the Thompson Sampling explainability layer.

---

## 🧩 Available Algorithms

RAMSeS includes the following base detectors:

**Statistical Methods:**
- `LOF` — Local Outlier Factor
- `CBLOF` — Cluster-Based Local Outlier Factor
- `COF` — Connectivity-Based Outlier Factor
- `KDE` — Kernel Density Estimation
- `NN` — k-Nearest Neighbors (Nearest Neighbors)
- `SOS` — Stochastic Outlier Selection

**Classical Methods:**
- `MD` — Mean Deviation
- `RM` — Running Mean
- `ABOD` — Angle-Based Outlier Detection

**Deep Learning Methods:**
- `LSTMVAE` — LSTM Variational Autoencoder
- `RNN` — Recurrent Neural Network
- `DGHL` — Deep Generative Hierarchical Learning
- `ALAD` — Adversarial Learned Anomaly Detection
- `AnomalyTransformer` — Transformer-based detector

**Configuration in `Configs/config.yml`:**
```yaml
model_architectures: 'all'  # Train all available algorithms
# OR specify subset:
model_architectures: 'LOF,CBLOF,NN,LSTMVAE'
```

Each algorithm can have multiple instances with different hyperparameters (e.g., `LOF_1`, `LOF_2`, `LOF_3`) for diversity in ensemble construction.

---

## 🧪 Reproducibility Tips

- **Fix random seeds** for all stochastic components (GA, Monte Carlo, GAN, Thompson Sampling)
- **Log hyperparameters** along with results
- **Use absolute paths** for `dataset_path` and `trained_model_path`
- **Version control** your configuration files
- **Document** dataset versions and preprocessing steps

---

## ❓ Troubleshooting

### Model Loading Issues

**Problem:** `Model X not found in save_dir`

**Solution:** 
- Ensure trained models exist at `trained_model_path/DATASET/ENTITY/`
- Check that model files are named correctly (e.g., `LOF_1.pth`, `CBLOF_2.pth`)
- Set `overwrite: True` in config to retrain models

### Dataset Loading Issues

**Problem:** Empty entities in train/test data

**Solution:**
- Verify `dataset_path` points to the correct Mononito root directory
- Check that dataset follows the expected structure in `Datasets/load.py`
- Ensure dataset name matches exactly (case-sensitive)

### Visualization Issues

**Problem:** Matplotlib display errors on remote servers

**Solution:**
```bash
# Set non-interactive backend before running
export MPLBACKEND=Agg
python app.py --config Configs/config.yml --dataset SKAB --entity 5
```

Or add to your Python script:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## 📚 Citation

If you use RAMSeS in your research, please cite our paper:

```bibtex
@misc{abdelmaksoud2026ramsesrobustadaptivemodel,
      title={RAMSeS: Robust and Adaptive Model Selection for Time-Series Anomaly Detection Algorithms}, 
      author={Mohamed Abdelmaksoud and Sheng Ding and Andrey Morozov and Ziawasch Abedjan},
      year={2026},
      eprint={2602.21766},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2602.21766}, 
}
```

---

## 👥 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Contribution areas:**
- New anomaly detection algorithms
- Additional robustness tests
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## 👤 Contact

**Maintainer:** Mohamed Abdelmaksoud  
**Email:** mohamed@tu-berlin.de  
**Research Group:** D2IP @ TU Berlin

For questions, bug reports, or feature requests, please [open a GitHub issue](https://github.com/Maxoud99/RAMSeS/issues).

---

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**Important:** Individual datasets may have their own licenses. Please respect the original dataset licenses when using RAMSeS.

---

## 🙏 Acknowledgments

- The **Mononito** benchmark dataset team
- Contributors to the open-source anomaly detection libraries (PyOD, etc.)
- The D2IP research group at TU Berlin

---

**Happy Anomaly Detection! 🎯**
