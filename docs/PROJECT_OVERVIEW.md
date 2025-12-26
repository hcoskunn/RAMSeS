# RAMSeS Project Overview

## 🎯 Project Summary

**RAMSeS (Robust & Adaptive Model Selection for Time-Series Anomaly Detection)** is an advanced research framework developed at D2IP @ TU Berlin for unsupervised time-series anomaly detection (TSAD). The project intelligently selects and combines multiple anomaly detection algorithms to achieve robust performance across diverse datasets.

---

## 🏗️ Architecture Overview

### Core Concept
RAMSeS produces **two complementary outputs**:

1. **Ensemble Branch**: A Genetic Algorithm (GA) discovers optimal stacking ensembles of base detectors
2. **Single-Model Branch**: Linear Thompson Sampling selects the best individual model through robust testing

### The Pipeline Flow
```
Input Data → Train Base Models → Parallel Selection Strategies → Final Recommendations
                                 ├─ Genetic Algorithm (Ensemble)
                                 ├─ Thompson Sampling (Online)
                                 ├─ GAN Robustness Testing
                                 ├─ Off-by-Threshold Sensitivity
                                 └─ Monte Carlo Noise Testing
                                         ↓
                                 Rank Aggregation → Best Model + Best Ensemble
```

---

## 📁 Project Structure

### Main Entry Point
- **`app.py`**: The central orchestrator that runs the complete RAMSeS pipeline

### Key Directories

#### 1. **Algorithms/** & **algorithm/**
Base anomaly detection models (both directories contain same algorithms):
- **Deep Learning**: DGHL, LSTMVAE, RNN, Anomaly Transformer, ALAD
- **Statistical**: Mean Deviation (MD), Running Mean (RM), KDE
- **Distance-based**: LOF, CBLOF, COF, Nearest Neighbors (NN), ABOD, SOS
- **Base Structure**: `base_model.py`, `pyod_model.py`

Each algorithm can have multiple instances (e.g., LOF_1, LOF_2, LOF_3, LOF_4) with different hyperparameters.

#### 2. **Model_Training/**
Training infrastructure for base detectors:
- `train.py`: Main training orchestrator (1259 lines)
- `trainer.py`: Training loop implementation
- `training_args.py`: Training configuration
- `hyperparameter_grids.py`: Hyperparameter configurations for each algorithm
- `entities.py`: Entity data structures

#### 3. **Model_Selection/**
Intelligent model selection strategies:
- **`Thompson_Sampling.py`**: Linear Thompson Sampling with ε-greedy exploration (530 lines)
  - Sliding window approach for online learning
  - Bayesian posterior updates
  - Feature-based reward estimation
  
- **`rank_aggregation.py`**: Markov-chain based rank aggregator
  - Combines multiple ranking sources
  - Produces consensus rankings

- **`inject_anomalies.py`**: Synthetic anomaly injection utilities
- **`model_selection.py`**: High-level selection logic
- **`anomaly_parameters.py`**: Anomaly configuration parameters

##### Sensitivity_robustness/ Subdirectory
Robustness testing suite:
- **`GAN_test.py`**: GAN-based adversarial robustness testing
- **`off_by_threshold_testing.py`**: Borderline/sensitivity analysis
- **`Monte_Carlo_Simulation.py`**: Noise stress-testing with Monte Carlo
- **`Adversarial_testing.py`**: Additional adversarial scenarios
- **`model_stress_testing.py`**: General stress testing utilities

#### 4. **Metrics/**
Performance evaluation and optimization:
- **`Ensemble_GA.py`**: Genetic Algorithm for ensemble optimization (761 lines)
  - Population initialization
  - Fitness function (combines F1-score and PR-AUC)
  - Crossover and mutation operators
  - Meta-learner training (RF, LR, GBM, SVM)
  
- `Ensemble_Genetics.py`: Alternative GA implementation
- `Ens_GA.py`: GA utilities
- `metrics.py`: Core metrics (F1, PR-AUC, etc.)
- `ranking_metrics.py`: Ranking-specific metrics

#### 5. **Datasets/**
Data loading and management:
- **`load.py`**: Universal data loader (529 lines)
  - Supports: MSL, SMAP, SMD, Anomaly Archive, SWAT, SKAB, Apple
  - Handles train/test splits
  - Normalization and preprocessing
  
- `dataset.py`: Dataset entity classes

#### 6. **Loaders/**
Data loading utilities:
- `loader.py`: Generic data loader interface

#### 7. **Utils/**
Utility functions:
- `utils.py`: Command-line argument parsing
- `logger.py`: Logging configuration
- `data_utils.py`: Data manipulation utilities
- `model_selection_utils.py`: Model selection helpers

#### 8. **Controller/**
High-level controllers:
- `mdata/`: Data management controllers
- `mevaluation/`: Evaluation controllers
- `mmodel/`: Model management controllers

#### 9. **dao/** (Data Access Objects)
Data access layer:
- `mdata/`: Data access patterns
- `mmodel/`: Model persistence

#### 10. **distributions/**
Probability distributions for rank aggregation:
- `mallows_kendall.py`: Mallows-Kendall distance
- `mallows_model.py`: Mallows probability model
- `pl_model.py`: Plackett-Luce model
- `permutil.py`: Permutation utilities
- `sampling.py`: Distribution sampling

#### 11. **Configs/**
Configuration files:
- `config.yml`: Main configuration

#### 12. **Output Directories**
- **`myresults/`**: Experimental results
  - `robust_aggregated/{dataset}/{entity}/`: Aggregated rankings
  - `GA_Ens/{dataset}/{entity}/`: Ensemble diagnostics and plots
- **`output/`**: Legacy output directory
- **`Outputs/`**: Additional outputs
- **`Plots/`**: Visualization outputs

#### 13. **Data Directories**
- **`Mononito/`**: Main dataset repository
  - `datasets/`: Raw data
  - `trained_models/`: Pre-trained model checkpoints (.pth files)
  - `results/`: Experimental results

#### 14. **db/**
Database files:
- `tsad.db`: SQLite database for experiment tracking

#### 15. **Other Components**
- **`Services/`**: Service layer components
- **`vus/`**: Volume Under Surface metric implementation
- **`old/`**: Deprecated code and experiments

---

## 🔄 Detailed Workflow

### Step 1: Data Loading
```python
train_data = load_data(dataset='mononito', group='train', entities='entity_name')
test_data = load_data(dataset='mononito', group='test', entities='entity_name')
```
- Supports multiple benchmark datasets
- Handles downsampling and minimum length requirements
- Normalizes data automatically

### Step 2: Model Training
```python
trainer = TrainModels(dataset='mononito', entity='entity_name')
trainer.train_models(model_architectures=['CBLOF', 'LOF', 'LSTMVAE', ...])
```
- Trains multiple instances of each algorithm with different hyperparameters
- Saves checkpoints as `.pth` files
- Grid search over hyperparameter space

### Step 3: Model Selection Pipeline

#### A. Genetic Algorithm (Ensemble Search)
```python
best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(
    dataset, entity, train_data, test_data,
    algorithm_list_instances, trained_models,
    population_size=20, generations=50,
    meta_model_type='rf', mutation_rate=0.1
)
```
**Process:**
1. Initialize population of algorithm combinations
2. Inject synthetic anomalies for training
3. Train meta-learner (Random Forest, Logistic Regression, GBM, or SVM)
4. Evaluate fitness (weighted combination of F1 and PR-AUC)
5. Select parents, crossover, mutate
6. Repeat for N generations

#### B. Linear Thompson Sampling (Online Selection)
```python
thompson_model_names = run_linear_thompson_sampling(
    test_data, trained_models, model_names,
    dataset, entity, iterations=50
)
```
**Process:**
1. Initialize sliding windows over test data
2. Maintain Bayesian posteriors (mean, covariance) for each model
3. For each window:
   - Sample models using ε-greedy or Thompson sampling
   - Evaluate chosen model
   - Compute reward (F1-score)
   - Update posteriors
4. Return ranked list of best-performing models

#### C. Robustness Testing Suite

**1. GAN-based Robustness**
```python
gan_results = run_Gan(test_data, trained_models, algorithm_list)
```
- Generates borderline synthetic anomalies using GANs
- Tests model stability on adversarial examples
- Returns rankings by F1 and PR-AUC

**2. Off-by-Threshold Sensitivity**
```python
ranked_models = run_off_by_threshold(test_data, trained_models, algorithm_list)
```
- Tests sensitivity to threshold variations
- Evaluates borderline classification stability
- Identifies models robust to threshold changes

**3. Monte Carlo Noise Testing**
```python
mc_ranked = run_monte_carlo_simulation(
    test_data, trained_models, algorithm_list,
    n_simulations=100, noise_level=0.1
)
```
- Adds Gaussian noise to test data
- Runs N simulations with different noise samples
- Ranks models by average performance under noise

#### D. Rank Aggregation
```python
# Robust-only aggregation
test_for_rank = [
    gan_f1_rankings, gan_pr_rankings,
    threshold_f1_rankings, threshold_pr_rankings,
    mc_f1_rankings, mc_pr_rankings
]
robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)

# Final merge with Thompson Sampling
full_aggregated = enhanced_markov_chain_rank_aggregator_text(
    [robust_agg[1], thompson_model_names]
)
```
- Uses Markov-chain based consensus ranking
- Combines all robustness tests
- Merges robust consensus with online Thompson Sampling results

### Step 4: Output Generation
Results are saved in structured directories:
```
myresults/
├── robust_aggregated/{dataset}/{entity}/
│   └── robust_aggregated_results_{dataset}_{entity}_{iteration}.txt
└── GA_Ens/{dataset}/{entity}/
    └── ensemble_scores_{dataset}_{entity}_Data_vs_anomalies.png
```

---

## 🧮 Algorithm Instances

The framework trains multiple instances of each base detector:

```python
algorithm_list = ['DGHL', 'LSTMVAE', 'NN', 'RNN', 'MD', 'RM', 'LOF', 'CBLOF']

algorithm_list_instances = [
    'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4',
    'NN_1', 'NN_2', 'NN_3',
    'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4',
    'CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4',
    'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4',
    'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4',
    'MD_1',
    'RM_1', 'RM_2', 'RM_3'
]
```

Each instance uses different hyperparameters, providing diversity for ensemble construction.

---

## 🎨 Key Features

### 1. **Two-Branch Strategy**
- **Ensemble**: GA-optimized stacking with meta-learner
- **Single-model**: Robust online selection with Thompson Sampling

### 2. **Comprehensive Robustness Testing**
- GAN-based adversarial testing
- Threshold sensitivity analysis
- Monte Carlo noise stress-testing

### 3. **Online Learning Capability**
- Sliding window approach
- Bayesian posterior updates
- Adaptive model selection

### 4. **Rank Aggregation**
- Markov-chain consensus ranking
- Combines multiple evaluation criteria
- Robust to individual ranking noise

### 5. **Reproducible Research**
- Human-readable result summaries
- Diagnostic plots
- Complete experiment logging

---

## 📊 Supported Datasets

- **MSL** (Mars Science Laboratory)
- **SMAP** (Soil Moisture Active Passive)
- **SMD** (Server Machine Dataset)
- **Anomaly Archive** (UCR)
- **SWAT** (Secure Water Treatment)
- **SKAB** (Skoltech Anomaly Benchmark)
- **Apple** (proprietary)
- **Mononito** (comprehensive benchmark collection)

---

## 🚀 Usage Example

```bash
# 1. Train base models
cd RAMSeS

# 2. Run complete RAMSeS pipeline
python app.py \
  --dataset_path /path/to/Mononito \
  --trained_model_path /path/to/trained_models \
  --downsampling 10 \
  --min_length 256 \
  --training_size 1.0 \
  --overwrite false \
  --verbose true \
  --model_architectures "CBLOF,LOF,LSTMVAE,MD,RM,DGHL,NN,RNN"
```

---

## 🔧 Configuration

### Key Parameters

**Data Configuration:**
- `dataset_path`: Root directory for datasets
- `trained_model_path`: Directory for model checkpoints
- `downsampling`: Downsampling factor
- `min_length`: Minimum sequence length

**Training Configuration:**
- `training_size`: Fraction of training data (0.0-1.0)
- `model_architectures`: Comma-separated list of algorithms

**GA Configuration:**
- `population_size`: Size of GA population (default: 20)
- `generations`: Number of GA generations (default: 50)
- `meta_model_type`: Meta-learner type ('rf', 'lr', 'gbm', 'svm')
- `mutation_rate`: Mutation probability (default: 0.1)

**Thompson Sampling Configuration:**
- `iterations`: Number of Thompson Sampling iterations (default: 50)
- `window_size`: Sliding window size
- `step_size`: Stride between windows
- `epsilon`: ε-greedy exploration rate

**Robustness Testing:**
- `n_simulations`: Monte Carlo simulation count (default: 100)
- `noise_level`: Gaussian noise standard deviation (default: 0.1)

---

## 📈 Output Interpretation

### 1. Ensemble Results
- **Best Ensemble**: List of algorithms in optimal ensemble
- **Best F1**: F1-score of ensemble
- **Best PR-AUC**: Precision-Recall AUC of ensemble
- **Meta-model**: Type of meta-learner used (RF/LR/GBM/SVM)

### 2. Thompson Sampling Results
- Ranked list of models by cumulative reward
- Top-5 models for deployment

### 3. Robustness Rankings
- **GAN Rankings**: Performance on adversarial examples
- **Threshold Rankings**: Stability across threshold variations
- **Monte Carlo Rankings**: Performance under noise

### 4. Final Aggregated Ranking
- Consensus ranking combining all methods
- Most robust model recommendations

---

## 🔍 Key Algorithms Explained

### Genetic Algorithm (GA)
**Purpose**: Find optimal ensemble of base detectors

**Fitness Function**:
```
fitness = α × F1_score + (1-α) × PR_AUC
```

**Operations**:
- **Selection**: Tournament or roulette wheel
- **Crossover**: Single-point or uniform
- **Mutation**: Random algorithm replacement

### Linear Thompson Sampling
**Purpose**: Online adaptive model selection

**Core Idea**: Maintain Bayesian posterior over model performance parameters

**Update Rule**:
```
μ_new = μ_old + K × (reward - predicted_reward)
Σ_new = Σ_old - K × H × K^T
```
where K is Kalman gain, H is feature matrix

### Markov-Chain Rank Aggregator
**Purpose**: Combine multiple rankings into consensus

**Approach**: Models ranking as Markov chain, finds stationary distribution

---

## 🛠️ Technical Stack

**Core Libraries:**
- **PyTorch**: Deep learning models
- **Scikit-learn**: Classical ML algorithms and meta-learners
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **PyOD**: Outlier detection library
- **TensorFlow**: Alternative deep learning backend

**Optimization:**
- **Ray**: Distributed computing (optional)
- **CVXPY**: Convex optimization

**Metrics:**
- F1-score
- Precision-Recall AUC
- Volume Under Surface (VUS)

---

## 📝 Important Notes

1. **Model Path Configuration**: Update `save_dir` in `app.py` to point to your trained models directory

2. **Dataset Structure**: Ensure dataset follows expected structure for `Datasets/load.py`

3. **GPU Support**: Optional but recommended for deep learning models (DGHL, LSTMVAE, RNN)

4. **Random Seeds**: Fix seeds for reproducibility (GA, Monte Carlo, GAN sampling, Thompson)

5. **Iterations**: Default is 1 iteration. Increase for online/real-time evaluation

---

## 🎓 Research Context

**Research Group**: D2IP @ TU Berlin  
**Status**: Active research code  
**Related Work**: Mononito benchmark (arXiv:2210.01078)

**Key Contributions:**
1. Dual-branch model selection (ensemble + single)
2. Comprehensive robustness testing suite
3. Online adaptive selection via Thompson Sampling
4. Markov-chain rank aggregation
5. Reproducible TSAD evaluation framework

---

## 📚 Additional Files

- **`address_validator.py`**: Address validation utility (likely for geo-tagged data)
- **`batch_address_validator.py`**: Batch address validation
- **`draw-intro-figure.py`**: Visualization scripts
- **`plot_ga_runtime_f1_from_csv.py`**: Performance plotting
- **`f1_runtime_ga_analysis.py`**: Runtime analysis
- **`runtime_ga_analysis.py`**: GA runtime profiling

---

## 🐛 Troubleshooting

**Common Issues:**

1. **"Model X not found"**: Check `save_dir` points to directory with `.pth` files
2. **Empty train/test data**: Verify `dataset_path` and dataset structure
3. **CUDA errors**: Install correct PyTorch version for your CUDA
4. **Memory errors**: Reduce batch size or use fewer model instances
5. **Matplotlib display issues**: Set `export MPLBACKEND=Agg` for headless servers

---

## 🎯 Quick Reference

**Main Execution Order:**
1. Load data → 2. Train models → 3. GA ensemble search → 4. Thompson Sampling → 5. Robustness tests (GAN, Threshold, Monte Carlo) → 6. Rank aggregation → 7. Results output

**Key Files to Modify:**
- `app.py`: Main configuration and pipeline
- `requirements.txt`: Dependencies
- `Configs/config.yml`: Global settings
- Hyperparameter grids in `Model_Training/hyperparameter_grids.py`

**Output Locations:**
- Results: `myresults/robust_aggregated/{dataset}/{entity}/`
- Plots: `myresults/GA_Ens/{dataset}/{entity}/`
- Trained models: `{trained_model_path}/{dataset}/{entity}/`

---

This is a sophisticated, production-ready research framework for time-series anomaly detection that emphasizes robustness, adaptability, and reproducibility. The dual-branch approach provides both ensemble and single-model recommendations, each validated through multiple robustness criteria.
