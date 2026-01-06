# time_series_framework/app.py
import concurrent.futures
import copy
import logging
import os
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch as t

from Datasets.load import load_data
from Metrics.Ensemble_GA import (
    genetic_algorithm,
    evaluate_individual_models,
    fitness_function,
)
from Model_Selection.Sensitivity_robustness.GAN_test import run_Gan
from Model_Selection.Sensitivity_robustness.Monte_Carlo_Simulation import (
    run_monte_carlo_simulation,
)
from Model_Selection.Sensitivity_robustness.off_by_threshold_testing import (
    run_off_by_threshold,
)
from Model_Selection.Thompson_Sampling import (
    run_linear_thompson_sampling,
    initialize_sliding_windows,
)
from Model_Selection.inject_anomalies import Inject
from Model_Selection.rank_aggregation import enhanced_markov_chain_rank_aggregator_text
from Model_Training.train import TrainModels
from Utils.utils import get_args_from_cmdline
from comprehensive_results_writer import write_comprehensive_results

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# save_dir removed - now dynamically determined from command-line args
# Old hardcoded path was causing wrong models to load for different datasets

algorithm_list = ['LSTMVAE', 'DGHL', 'NN', 'RNN', 'LOF', 'MD', 'CBLOF']
algorithm_list_instances = ['LOF_1', 'NN_1', 'NN_2', 'RNN_1',
    'CBLOF_1', 'MD_1', 
    'DGHL_1',  'LSTMVAE_1', 
]

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def write_comprehensive_results(output_file, dataset, entity, iteration, results_dict, timing_dict):
    """
    Write comprehensive results including timing, scores, and chosen models.
    
    Parameters
    ----------
    output_file : str
        Path to output file
    dataset : str
        Dataset name
    entity : str
        Entity identifier
    iteration : int
        Iteration number
    results_dict : dict
        Dictionary containing all results
    timing_dict : dict
        Dictionary containing timing information
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RAMSeS Framework - Comprehensive Results\n")
        f.write(f"Dataset: {dataset} | Entity: {entity} | Iteration: {iteration}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # ============ COMPUTATIONAL OVERHEAD ============
        f.write("="*80 + "\n")
        f.write("COMPUTATIONAL OVERHEAD (seconds)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Per-Module Timing:\n")
        f.write("-" * 50 + "\n")
        total_module_time = 0
        for module, elapsed in timing_dict.get('modules', {}).items():
            f.write(f"  {module:<30} : {elapsed:>10.4f}s\n")
            total_module_time += elapsed
        
        f.write("-" * 50 + "\n")
        f.write(f"  {'Total Module Time':<30} : {total_module_time:>10.4f}s\n")
        f.write(f"  {'End-to-End Time':<30} : {timing_dict.get('total', 0):>10.4f}s\n")
        f.write(f"  {'Overhead (E2E - Modules)':<30} : {timing_dict.get('total', 0) - total_module_time:>10.4f}s\n\n")
        
        # ============ GENETIC ALGORITHM (ENSEMBLE) ============
        f.write("="*80 + "\n")
        f.write("GENETIC ALGORITHM - ENSEMBLE SELECTION\n")
        f.write("="*80 + "\n\n")
        
        ga_results = results_dict.get('ga', {})
        f.write(f"Best Ensemble: {ga_results.get('ensemble', 'N/A')}\n")
        f.write(f"  F1 Score    : {ga_results.get('f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {ga_results.get('pr_auc', 0):.6f}\n")
        f.write(f"  Fitness     : {ga_results.get('fitness', 0):.6f}\n")
        f.write(f"  Meta-Model  : {ga_results.get('meta_model_type', 'N/A')}\n")
        f.write(f"  Ensemble Size: {len(ga_results.get('ensemble', []))}\n\n")
        
        # ============ THOMPSON SAMPLING ============
        f.write("="*80 + "\n")
        f.write("THOMPSON SAMPLING - ONLINE MODEL SELECTION\n")
        f.write("="*80 + "\n\n")
        
        thompson_results = results_dict.get('thompson', {})
        f.write(f"Chosen Model: {thompson_results.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {thompson_results.get('f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {thompson_results.get('pr_auc', 0):.6f}\n\n")
        
        f.write("Top-5 Models (Ranked):\n")
        for i, model in enumerate(thompson_results.get('top_models', [])[:5], 1):
            f.write(f"  {i}. {model}\n")
        f.write("\n")
        
        # ============ ROBUSTNESS TESTS ============
        f.write("="*80 + "\n")
        f.write("ROBUSTNESS TESTS - INDIVIDUAL RANKINGS\n")
        f.write("="*80 + "\n\n")
        
        # GAN Results
        f.write("GAN Robustness Test:\n")
        f.write("-" * 50 + "\n")
        gan_results_rob = results_dict.get('gan_robustness', {})
        f.write(f"Chosen Model: {gan_results_rob.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {gan_results_rob.get('best_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {gan_results_rob.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(gan_results_rob.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(gan_results_rob.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # Borderline/Off-by-Threshold Results
        f.write("Borderline Sensitivity Test:\n")
        f.write("-" * 50 + "\n")
        borderline_results = results_dict.get('borderline', {})
        f.write(f"Chosen Model: {borderline_results.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {borderline_results.get('best_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {borderline_results.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(borderline_results.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(borderline_results.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # Monte Carlo Results
        f.write("Monte Carlo Simulation:\n")
        f.write("-" * 50 + "\n")
        mc_results = results_dict.get('monte_carlo', {})
        f.write(f"Chosen Model (F1)   : {mc_results.get('best_model_f1', 'N/A')}\n")
        f.write(f"Chosen Model (PR-AUC): {mc_results.get('best_model_pr_auc', 'N/A')}\n")
        f.write(f"  Best F1 Score      : {mc_results.get('best_f1', 0):.6f}\n")
        f.write(f"  Best PR-AUC Score  : {mc_results.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(mc_results.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(mc_results.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # ============ RANK AGGREGATION ============
        f.write("="*80 + "\n")
        f.write("RANK AGGREGATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        aggregation_results = results_dict.get('aggregation', {})
        
        f.write("Robust Aggregation (GAN → Borderline → Monte Carlo):\n")
        f.write("-" * 50 + "\n")
        robust_agg = aggregation_results.get('robust_agg', [])
        if isinstance(robust_agg, (list, tuple)) and len(robust_agg) > 1:
            f.write(f"  Best Model: {robust_agg[1] if len(robust_agg) > 1 else 'N/A'}\n")
            f.write(f"  Full Ranking: {robust_agg}\n")
        else:
            f.write(f"  Result: {robust_agg}\n")
        f.write("\n")
        
        f.write("Final Aggregation (Robust + Thompson Sampling):\n")
        f.write("-" * 50 + "\n")
        final_agg = aggregation_results.get('final_agg', [])
        if isinstance(final_agg, (list, tuple)) and len(final_agg) > 1:
            f.write(f"  Best Model: {final_agg[1] if len(final_agg) > 1 else 'N/A'}\n")
            f.write(f"  Full Ranking: {final_agg}\n")
        else:
            f.write(f"  Result: {final_agg}\n")
        f.write("\n")
        
        # ============ FRAMEWORK FINAL DECISION ============
        f.write("="*80 + "\n")
        f.write("FRAMEWORK FINAL DECISION\n")
        f.write("="*80 + "\n\n")
        
        final_decision = results_dict.get('final_decision', {})
        
        # Show both options
        f.write("Single Model Option:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Model       : {final_decision.get('single_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {final_decision.get('single_model_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {final_decision.get('single_model_pr_auc', 0):.6f}\n\n")
        
        f.write("Ensemble Option:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Models      : {final_decision.get('ensemble', 'N/A')}\n")
        f.write(f"  Size        : {len(final_decision.get('ensemble', []))}\n")
        f.write(f"  Meta-Model  : {final_decision.get('meta_model_type', 'N/A')}\n")
        f.write(f"  F1 Score    : {final_decision.get('ensemble_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {final_decision.get('ensemble_pr_auc', 0):.6f}\n")
        f.write(f"  Fitness     : {final_decision.get('ensemble_fitness', 0):.6f}\n\n")
        
        # Make the choice based on F1 score
        ensemble_f1 = final_decision.get('ensemble_f1', 0)
        single_f1 = final_decision.get('single_model_f1', 0)
        
        f.write("Final Choice:\n")
        f.write("-" * 50 + "\n")
        if ensemble_f1 >= single_f1:
            f.write(f"  ✓ ENSEMBLE SELECTED\n")
            f.write(f"    Reason: Ensemble F1 ({ensemble_f1:.6f}) >= Single Model F1 ({single_f1:.6f})\n")
            if ensemble_f1 > single_f1:
                improvement = ensemble_f1 - single_f1
                f.write(f"    Improvement: +{improvement:.6f} ({improvement/max(single_f1, 0.0001)*100:.2f}%)\n")
            else:
                f.write(f"    Note: F1 scores are equal\n")
        else:
            f.write(f"  ✓ SINGLE MODEL SELECTED\n")
            f.write(f"    Reason: Single Model F1 ({single_f1:.6f}) > Ensemble F1 ({ensemble_f1:.6f})\n")
            advantage = single_f1 - ensemble_f1
            f.write(f"    Advantage: +{advantage:.6f} ({advantage/max(ensemble_f1, 0.0001)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Comprehensive results saved to: {output_file}")


def load_trained_models(model_names, models_dir):
    """
    Load trained models from disk.

    Parameters
    ----------
    model_names : list[str]
        List of model instance names (e.g., 'CBLOF_1', 'LOF_3', ...).
    models_dir : str
        Directory where .pth files are stored.

    Returns
    -------
    dict[str, torch.nn.Module]
    """
    trained = {}
    missing = []
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            missing.append(name)
            logger.warning(f"Model {name} not found in {models_dir}, skipping")
            continue
        with open(path, 'rb') as fh:
            model = t.load(fh, weights_only=False)
            try:
                model.eval()
            except AttributeError:
                # If the checkpoint isn't a nn.Module, we still store it as-is.
                pass
            trained[name] = model
    
    logger.info("Loaded %d trained models from %s", len(trained), models_dir)
    if missing:
        logger.warning(f"Missing {len(missing)} models: {', '.join(missing)}")
    return trained

# ------------------------------------------------------------------------------
# Model-Selection Pipelines
# ------------------------------------------------------------------------------

def run_model_selection_algorithms_1(train_data, test_data, dataset, entity, iteration, model_list=None):
    """
    One-pass model selection pipeline in the order:
      1) GA (stacking ensemble search)
      2) Thompson Sampling (LinTS)
      3) GAN robustness test
      4) Off-by-threshold (borderline sensitivity)
      5) Monte Carlo (noise stress test)
      6) Rank aggregations (robust-only, then merged with Thompson)

    Parameters
    ----------
    model_list : list[str], optional
        List of model names to use. If None, uses global algorithm_list_instances.

    Returns (exactly 11 items):
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type,
        extra_results_dict
    """
    # Use provided model list or fall back to global
    models_to_use = model_list if model_list is not None else algorithm_list_instances
    
    timing_dict = {}
    
    # -------------------------
    # 1) Genetic Algorithm (GA)
    # -------------------------
    logger.info("  📊 Sub-stage 6.1: Genetic Algorithm (GA) - Finding best ensemble...")
    logger.info("     This will evaluate individual models and run 20 generations")
    start_time = time.time()
    best_ensemble, best_f1, best_pr_auc, best_fitness, \
    individual_predictions, base_model_predictions_train, base_model_predictions_test, \
    y_true_train, y_true_test, meta_model_type = genetic_algorithm(
        dataset, entity, train_data, test_data,
        models_to_use, trained_models,
        population_size=20, generations=20,
        meta_model_type='rf', mutation_rate=0.1,
    )
    timing_dict['1_Genetic_Algorithm'] = time.time() - start_time
    logger.info(
        "  ✓ [GA] Best ensemble=%s | F1=%.4f | PR-AUC=%.4f | fitness=%.4f | Time=%.4fs",
        best_ensemble, best_f1, best_pr_auc, best_fitness, timing_dict['1_Genetic_Algorithm']
    )

    # -----------------------------------
    # 2) Thompson Sampling (LinTS, online)
    # -----------------------------------
    logger.info("  📊 Sub-stage 6.2: Thompson Sampling - Online model selection...")
    start_time = time.time()
    thompson_model_names = run_linear_thompson_sampling(
        test_data=test_data,
        trained_models=trained_models,
        model_names=algorithm_list_instances,
        dataset=dataset,
        entity=entity,
        iterations=50,
        iteration=iteration,
    )
    timing_dict['2_Thompson_Sampling'] = time.time() - start_time
    logger.info("  ✓ [Thompson] Top-5: %s | Time=%.4fs", thompson_model_names[:5], 
                timing_dict['2_Thompson_Sampling'])

    # -------------------------
    # 3) GAN Robustness Testing
    # -------------------------
    logger.info("  📊 Sub-stage 6.3: GAN Robustness Testing...")
    start_time = time.time()
    test_data_for_gan = copy.deepcopy(test_data)
    gan_results = run_Gan(
        test_data_for_gan, trained_models, algorithm_list_instances, dataset, entity
    )
    timing_dict['3_GAN_Robustness'] = time.time() - start_time
    Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
    Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = (
        gan_results[0], gan_results[1], gan_results[2], gan_results[3]
    )
    timing_dict['3_GAN_Robustness'] = time.time() - start_time
    logger.info("  ✓ [GAN] F1 names top-5: %s | Time=%.4fs", Gan_ranked_by_f1_names[:5],
                timing_dict['3_GAN_Robustness'])
    logger.info("     [GAN] PR names top-5: %s", Gan_ranked_by_pr_auc_names[:5])

    # --------------------------------------------
    # 4) Off-by-threshold (borderline sensitivity)
    # --------------------------------------------
    logger.info("  📊 Sub-stage 6.4: Off-by-Threshold Testing...")
    start_time = time.time()
    test_data_for_borderline = copy.deepcopy(test_data)
    ranked_by_f1, ranked_by_pr_auc, \
    ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = run_off_by_threshold(
        test_data_for_borderline, trained_models, algorithm_list_instances, dataset, entity
    )
    timing_dict['4_Borderline_Sensitivity'] = time.time() - start_time
    logger.info("  ✓ [Borderline] F1 names top-5: %s | Time=%.4fs", ranked_by_f1_names_sensitivity[:5],
                timing_dict['4_Borderline_Sensitivity'])
    logger.info("     [Borderline] PR names top-5: %s", ranked_by_pr_auc_names_sensitivity[:5])

    # ---------------------------------
    # 5) Monte Carlo (noise stress test)
    # ---------------------------------
    logger.info("  📊 Sub-stage 6.5: Monte Carlo Simulation...")
    start_time = time.time()
    test_data_for_mc = copy.deepcopy(test_data)
    monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = run_monte_carlo_simulation(
        test_data_for_mc, trained_models, algorithm_list_instances, dataset, entity,
        n_simulations=2, noise_level=0.1,
    )
    timing_dict['5_Monte_Carlo'] = time.time() - start_time
    logger.info("  ✓ [MonteCarlo] F1 names top-5: %s | Time=%.4fs", monte_carlo_ranked_models_F1[:5],
                timing_dict['5_Monte_Carlo'])
    logger.info("     [MonteCarlo] PR names top-5: %s", monte_carlo_ranked_models_PR[:5])

    # -----------------------
    # 6) Rank Aggregations
    # -----------------------
    logger.info("  📊 Sub-stage 6.6: Rank Aggregation...")
    start_time = time.time()
    # Robust-only aggregation in the requested order: GAN → Borderline → Monte Carlo
    test_for_rank = [
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
    ]
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)

    # Final merge of robust aggregation vs Thompson Sampling
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)
    timing_dict['6_Rank_Aggregation'] = time.time() - start_time
    logger.info("  ✓ [Aggregation] Time=%.4fs", timing_dict['6_Rank_Aggregation'])

    # -----------------------
    # Persist a concise report
    # -----------------------
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of robust tests (order: GAN → Borderline → Monte Carlo):\n")
        f.write("\n[GAN]\n")
        f.write(f"{Gan_ranked_by_f1_names}\n{Gan_ranked_by_pr_auc_names}\n")
        f.write("\n[Borderline / Off-by-threshold]\n")
        f.write(f"{ranked_by_f1_names_sensitivity}\n{ranked_by_pr_auc_names_sensitivity}\n")
        f.write("\n[Monte Carlo]\n")
        f.write(f"{monte_carlo_ranked_models_F1}\n{monte_carlo_ranked_models_PR}\n")
        f.write("\n[Robust rank aggregate]\n")
        f.write(f"{robust_agg}\n")
        f.write("\n[Final aggregate vs Thompson]\n")
        f.write(f"{full_aggregated}\n")

    # Return extended tuple with all necessary data for comprehensive results
    return (
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type,
        # Additional data for comprehensive results
        {
            'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness},
            'thompson': thompson_model_names,
            'gan': {
                'f1_names': Gan_ranked_by_f1_names, 
                'pr_auc_names': Gan_ranked_by_pr_auc_names,
                'f1_scores': Gan_ranked_by_f1,
                'pr_auc_scores': Gan_ranked_by_pr_auc,
                'best_model': Gan_ranked_by_f1_names[0],
                'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0
            },
            'borderline': {
                'f1_names': ranked_by_f1_names_sensitivity, 
                'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                'f1_scores': ranked_by_f1,
                'pr_auc_scores': ranked_by_pr_auc,
                'best_model': ranked_by_f1_names_sensitivity[0],
                'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0
            },
            'monte_carlo': {
                'f1_names': monte_carlo_ranked_models_F1, 
                'pr_auc_names': monte_carlo_ranked_models_PR,
                # Note: Monte Carlo returns names only, not scores
                'best_model_f1': monte_carlo_ranked_models_F1[0],
                'best_model_pr_auc': monte_carlo_ranked_models_PR[0]
            },
            'robust_agg': robust_agg,
            'full_aggregated': full_aggregated,
            'timing': timing_dict
        }
    )


def run_model_selection_algorithms_2(train_data, test_data, dataset, entity, iteration, trained_models, model_list=None):
    """
    PARALLEL VERSION: Runs model selection algorithms concurrently using ThreadPoolExecutor.
    
    Executes 5 algorithms in parallel:
      1) GA (genetic algorithm for stacking ensemble)
      2) Thompson Sampling (online LinTS)
      3) GAN robustness test
      4) Off-by-threshold (borderline sensitivity)
      5) Monte Carlo (noise stress test)
    
    Parameters
    ----------
    trained_models : dict
        Dictionary of trained model instances
    model_list : list[str], optional
        List of model names to use. If None, uses global algorithm_list_instances.
        
    Returns
    -------
    Same 11-item tuple as run_model_selection_algorithms_1
    """
    # Use provided model list or fall back to global
    models_to_use = model_list if model_list is not None else algorithm_list_instances
    
    logger.info("  🚀 Starting PARALLEL model selection (5 algorithms concurrently)...")
    logger.info("  ✨ VERSION CHECK: Using run_model_selection_algorithms_2 with deepcopy fix")
    timing_dict = {}
    overall_start = time.time()
    
    # CRITICAL: Create independent copies of test_data for each algorithm
    # This prevents race conditions when algorithms modify data during processing
    logger.info("     📋 Creating independent data copies...")
    test_data_ga = copy.deepcopy(test_data)
    test_data_thompson = copy.deepcopy(test_data)
    test_data_gan = copy.deepcopy(test_data)
    test_data_borderline = copy.deepcopy(test_data)
    test_data_montecarlo = copy.deepcopy(test_data)
    logger.info("     ✓ Created 5 independent data copies for thread safety")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        logger.info("     Launching: GA | Thompson | GAN | Borderline | Monte Carlo")
        
        # Submit all 5 algorithms with their own data copies
        ga_future = executor.submit(
            genetic_algorithm,
            dataset, entity, train_data, test_data_ga,
            models_to_use, trained_models,
            20, 20, 'rf', 0.1
        )
        thompson_future = executor.submit(
            run_linear_thompson_sampling,
            test_data=test_data_thompson,
            trained_models=trained_models,
            model_names=models_to_use,
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration,
        )
        gan_future = executor.submit(
            run_Gan,
            test_data_gan, trained_models, models_to_use,
            dataset, entity
        )
        borderline_future = executor.submit(
            run_off_by_threshold,
            test_data_borderline, trained_models, models_to_use,
            dataset, entity
        )
        monte_carlo_future = executor.submit(
            run_monte_carlo_simulation,
            test_data_montecarlo, trained_models, models_to_use,
            dataset, entity, 2, 0.1
        )

        # Collect results
        logger.info("     ⏳ Waiting for parallel tasks...")
        
        best_ensemble, best_f1, best_pr_auc, best_fitness, \
        individual_predictions, base_model_predictions_train, base_model_predictions_test, \
        y_true_train, y_true_test, meta_model_type = ga_future.result()
        timing_dict['1_GA'] = time.time() - overall_start
        logger.info("     ✓ GA: ensemble=%s | F1=%.4f | PR-AUC=%.4f", best_ensemble, best_f1, best_pr_auc)
        
        thompson_model_names = thompson_future.result()
        timing_dict['2_Thompson'] = time.time() - overall_start
        logger.info("     ✓ Thompson: top-5=%s", thompson_model_names[:5])
        
        Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = gan_future.result()
        timing_dict['3_GAN'] = time.time() - overall_start
        logger.info("     ✓ GAN: F1 top-5=%s", Gan_ranked_by_f1_names[:5])
        
        ranked_by_f1, ranked_by_pr_auc, \
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = borderline_future.result()
        timing_dict['4_Borderline'] = time.time() - overall_start
        logger.info("     ✓ Borderline: F1 top-5=%s", ranked_by_f1_names_sensitivity[:5])
        
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = monte_carlo_future.result()
        timing_dict['5_MonteCarlo'] = time.time() - overall_start
        logger.info("     ✓ MonteCarlo: F1 top-5=%s", monte_carlo_ranked_models_F1[:5])

    timing_dict['0_Parallel_Total'] = time.time() - overall_start
    logger.info("  ✅ All 5 algorithms completed in %.2fs (parallel)", timing_dict['0_Parallel_Total'])

    # Rank Aggregation
    logger.info("  📊 Sub-stage 6.6: Rank Aggregation...")
    agg_start = time.time()
    test_for_rank = [
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
    ]
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)
    timing_dict['6_Aggregation'] = time.time() - agg_start
    logger.info("  ✓ Aggregation: %.2fs", timing_dict['6_Aggregation'])

    # Persist results
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of robust tests (PARALLEL execution):\n")
        f.write("\n[GAN]\n")
        f.write(f"{Gan_ranked_by_f1_names}\n{Gan_ranked_by_pr_auc_names}\n")
        f.write("\n[Borderline]\n")
        f.write(f"{ranked_by_f1_names_sensitivity}\n{ranked_by_pr_auc_names_sensitivity}\n")
        f.write("\n[Monte Carlo]\n")
        f.write(f"{monte_carlo_ranked_models_F1}\n{monte_carlo_ranked_models_PR}\n")
        f.write("\n[Robust rank aggregate]\n")
        f.write(f"{robust_agg}\n")
        f.write("\n[Final aggregate vs Thompson]\n")
        f.write(f"{full_aggregated}\n")

    # Return 11-item tuple matching sequential version
    return (
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type,
        {
            'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness},
            'thompson': thompson_model_names,
            'gan': {
                'f1_names': Gan_ranked_by_f1_names, 
                'pr_auc_names': Gan_ranked_by_pr_auc_names,
                'f1_scores': Gan_ranked_by_f1,
                'pr_auc_scores': Gan_ranked_by_pr_auc,
                'best_model': Gan_ranked_by_f1_names[0],
                'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0
            },
            'borderline': {
                'f1_names': ranked_by_f1_names_sensitivity, 
                'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                'f1_scores': ranked_by_f1,
                'pr_auc_scores': ranked_by_pr_auc,
                'best_model': ranked_by_f1_names_sensitivity[0],
                'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0
            },
            'monte_carlo': {
                'f1_names': monte_carlo_ranked_models_F1, 
                'pr_auc_names': monte_carlo_ranked_models_PR,
                'best_model_f1': monte_carlo_ranked_models_F1[0],
                'best_model_pr_auc': monte_carlo_ranked_models_PR[0]
            },
            'robust_agg': robust_agg,
            'full_aggregated': full_aggregated,
            'timing': timing_dict
        }
    )

# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

def find_num_falses(adjusted_y_pred_ind_current, test_data_copy, dataset, entity, values,
                    full_aggregated, best_ensemble, iteration):
    """
    Compute and persist the number of misclassifications for current single model(s) and ensemble.
    """
    misclassified_current = []
    for predicts in adjusted_y_pred_ind_current:
        true_values = np.array(test_data_copy.entities[0].labels)
        predicted_int = np.array(predicts).astype(int)
        incorrect = predicted_int != true_values
        misclassified_current.append(int(np.sum(incorrect)))

    misclassified_ensemble = []
    for predicts in [values[3]]:
        true_values = np.array(values[4])
        predicted_int = np.array(predicts).astype(int)
        incorrect = predicted_int != true_values
        misclassified_ensemble.append(int(np.sum(incorrect)))

    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of falses:\n")
        f.write(f"chosen model (aggregated): {full_aggregated}\n")
        f.write(f"misclassified_current: {misclassified_current}\n")
        f.write("Falses for the ensemble:\n")
        f.write(f"chosen ensemble: {best_ensemble}\n")
        f.write(f"misclassified_ensemble: {misclassified_ensemble}\n")

# ------------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------------

def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()

    # Get dataset and entity from args (command line overrides, or use defaults)
    dataset = args.get('dataset', 'skab')
    entity = str(args.get('entity', '3'))
    use_parallel = args.get('parallel', False)
    
    data_dir = args['dataset_path']
    
    logger.info("="*80)
    logger.info(f"🚀 STARTING RAMSeS EXECUTION: dataset={dataset}, entity={entity}, parallel={use_parallel}")
    logger.info("="*80)
    
    logger.info("📂 STAGE 1/7: Loading Training Data...")
    train_data = load_data(
        dataset=dataset, group='train',
        entities=entity, downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Training data loaded: {len(train_data.entities)} entity(ies)")
    
    logger.info("📂 STAGE 2/7: Loading Test Data...")
    test_data = load_data(
        dataset=dataset, group='test',
        entities=entity, downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Test data loaded: {len(test_data.entities)} entity(ies)")

    if not train_data.entities:
        logger.error("Failed to load training data. Check dataset and paths.")
        return
    if not test_data.entities:
        logger.error("Failed to load test data. Check dataset and paths.")
        return

    logger.info("🔧 STAGE 3/7: Training/Loading Models...")
    model_trainer = TrainModels(
        dataset=dataset,
        entity=entity,
        algorithm_list=algorithm_list,
        downsampling=args['downsampling'],
        min_length=args['min_length'],
        root_dir=args['dataset_path'],
        training_size=args['training_size'],
        overwrite=args['overwrite'],
        verbose=args['verbose'],
        save_dir=args['trained_model_path'],
    )

    try:
        # Train (no-op if already present, depending on TrainModels implementation)
        logger.info("  → Checking existing models and training missing ones...")
        model_trainer.train_models(model_architectures=args['model_architectures'])
        logger.info("  ✓ Model training phase complete")

        # Load trained models using the dynamic path (base_dir/dataset/entity/)
        logger.info("  → Loading trained models from disk...")
        global trained_models
        models_dir = os.path.join(args['trained_model_path'], dataset, entity)
        trained_models = load_trained_models(algorithm_list_instances, models_dir)
        if not trained_models:
            raise ValueError("No models loaded. Check model paths and ensure models are trained.")
        
        # Filter algorithm_list_instances to only include successfully loaded models
        loaded_model_names = list(trained_models.keys())
        logger.info(f"✓ Loaded {len(loaded_model_names)} models: {', '.join(loaded_model_names)}")

        logger.info("💉 STAGE 4/7: Injecting Synthetic Anomalies...")
        # Inject anomalies
        anomaly_list = ['spikes']
        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)

        train_data, _ = Inject(train_data, anomaly_list)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list)
        logger.info(f"✓ Injected anomalies: {anomaly_list}")

        logger.info("📊 STAGE 5/7: Preparing Data and Visualization...")
        # Simple visualization of injected region
        anomaly_start = int(np.argmax(test_data.entities[0].labels))
        anomaly_end = test_data.entities[0].Y.shape[1] - int(np.argmax(test_data.entities[0].labels[::-1]))
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
        axes[0].plot(test_data.entities[0].Y.flatten())
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     color='red')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data_before.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     linestyle='--')
        axes[0].set_title('Test data with Injected Anomalies', fontsize=16)
        axes[1].plot(anomaly_sizes.flatten())
        axes[1].plot(test_data.entities[0].labels.flatten(), color='red')
        axes[1].set_title('Anomaly Scores', fontsize=16)

        out_dir = f"myresults/GA_Ens/{dataset}/{entity}/"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_{anomaly_list}.png")
        plt.savefig(out_file, dpi=300)
        logger.info(f"✓ Visualization saved to {out_file}")

        logger.info("🔍 STAGE 6/7: Running Model Selection Algorithms...")
        logger.info("  This includes: GA, Thompson Sampling, GAN, Monte Carlo, Off-by-Threshold")
        # Sliding windows setup
        data = test_data_before.entities[0].Y
        targets = test_data_before.entities[0].labels
        mask = test_data_before.entities[0].mask
        iterations = 1  # Real-time loop count

        data_windows, targets_windows, new_mask, num_windows = initialize_sliding_windows(
            data, targets, mask,
            int(np.size(targets.flatten()) / iterations),
            int(np.size(targets.flatten()) / iterations) - 5,
        )

        # First window
        test_data.entities[0].Y = data_windows[0]
        test_data.entities[0].labels = targets_windows[0]
        test_data.entities[0].mask = new_mask[0]
        test_data.entities[0].n_time = int(np.size(targets_windows[0].flatten()))
        test_data.total_time = int(np.size(targets_windows[0].flatten()))

        test_data_new = copy.deepcopy(test_data)
        test_data_new, _ = Inject(test_data_new, anomaly_list)

        # Start end-to-end timing
        e2e_start_time = time.time()
        if use_parallel:
            logger.info("  ⏱ Starting model selection pipeline (PARALLEL mode)...")
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type, extra_results) = run_model_selection_algorithms_2(
                train_data, test_data_new, dataset, entity, iteration=0, 
                trained_models=trained_models, model_list=loaded_model_names
            )
        else:
            logger.info("  ⏱ Starting model selection pipeline (SEQUENTIAL mode)...")
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type, extra_results) = run_model_selection_algorithms_1(
                train_data, test_data_new, dataset, entity, iteration=0, model_list=loaded_model_names
            )
        
        # Calculate end-to-end time
        e2e_time = time.time() - e2e_start_time
        logger.info(f"✓ Model selection completed in {e2e_time:.2f}s ({e2e_time/60:.2f} min)")
        
        # Helper function to convert any value to scalar
        def to_scalar(value):
            """Convert tuple, list, array, or scalar to a single float value."""
            # Handle nested structures
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return 0.0
                value = value[0]
            if isinstance(value, np.ndarray):
                value = float(value.flat[0]) if value.size > 0 else 0.0
            value = float(value)
            # Validate F1/PR-AUC is in valid range [0, 1]
            if value < 0.0 or value > 1.0:
                logger.warning(f"Invalid F1/PR-AUC score detected: {value}. This should be between 0 and 1.")
            return value
        
        # Decide: Ensemble vs Single Model based on scores
        ensemble_f1 = to_scalar(extra_results['ga']['f1'])
        ensemble_pr_auc = to_scalar(extra_results['ga']['pr_auc'])
        
        # Get best single model from final aggregation
        best_single_model = full_aggregated[1] if isinstance(full_aggregated, (list, tuple)) and len(full_aggregated) > 1 else full_aggregated
        
        # Evaluate Thompson Sampling best model
        best_thompson_model = best_thompson if isinstance(best_thompson, str) else (best_thompson[0] if best_thompson else 'N/A')
        test_data_for_thompson = copy.deepcopy(test_data_new)
        _, thompson_scores, thompson_f1_list, thompson_pr_list = evaluate_individual_models(
            [best_thompson_model], test_data_for_thompson, trained_models
        )
        thompson_f1 = to_scalar(thompson_f1_list[0]) if len(thompson_f1_list) > 0 else 0.0
        thompson_pr_auc = to_scalar(thompson_pr_list[0]) if len(thompson_pr_list) > 0 else 0.0
        
        # Evaluate the best single model from final aggregation to get its F1 score
        test_data_for_eval = copy.deepcopy(test_data_new)
        _, single_model_scores, single_f1_list, single_pr_list = evaluate_individual_models(
            [best_single_model], test_data_for_eval, trained_models
        )
        single_model_f1 = to_scalar(single_f1_list[0]) if len(single_f1_list) > 0 else 0.0
        single_model_pr_auc = to_scalar(single_pr_list[0]) if len(single_pr_list) > 0 else 0.0
        
        # Evaluate Monte Carlo best models
        best_mc_f1_model = extra_results['monte_carlo'].get('best_model_f1', 'N/A')
        best_mc_pr_model = extra_results['monte_carlo'].get('best_model_pr_auc', 'N/A')
        test_data_for_mc = copy.deepcopy(test_data_new)
        
        mc_f1_score = 0.0
        mc_pr_auc_score = 0.0
        if best_mc_f1_model != 'N/A':
            _, _, mc_f1_list, _ = evaluate_individual_models([best_mc_f1_model], test_data_for_mc, trained_models)
            mc_f1_score = to_scalar(mc_f1_list[0]) if len(mc_f1_list) > 0 else 0.0
        
        if best_mc_pr_model != 'N/A':
            test_data_for_mc2 = copy.deepcopy(test_data_new)
            _, _, _, mc_pr_list = evaluate_individual_models([best_mc_pr_model], test_data_for_mc2, trained_models)
            mc_pr_auc_score = to_scalar(mc_pr_list[0]) if len(mc_pr_list) > 0 else 0.0
        
        # Framework decision: Choose ensemble if its F1 >= single model F1
        framework_choice = 'ensemble' if ensemble_f1 >= single_model_f1 else 'single_model'
        
        # Prepare comprehensive results using captured data
        results_dict = {
            'ga': {
                'ensemble': best_ensemble,
                'f1': ensemble_f1,
                'pr_auc': ensemble_pr_auc,
                'fitness': extra_results['ga']['fitness'],
                'meta_model_type': meta_model_type,
                'chosen_model': best_ensemble
            },
            'thompson': {
                'top_models': extra_results['thompson'],
                'best_model': best_thompson_model,
                'f1': thompson_f1,
                'pr_auc': thompson_pr_auc
            },
            'gan_robustness': {
                'f1_names': extra_results['gan']['f1_names'],
                'pr_auc_names': extra_results['gan']['pr_auc_names'],
                'best_model': extra_results['gan'].get('best_model', 'N/A'),
                'best_f1': extra_results['gan'].get('best_f1', 0.0),
                'best_pr_auc': extra_results['gan'].get('best_pr_auc', 0.0)
            },
            'borderline': {
                'f1_names': extra_results['borderline']['f1_names'],
                'pr_auc_names': extra_results['borderline']['pr_auc_names'],
                'best_model': extra_results['borderline'].get('best_model', 'N/A'),
                'best_f1': extra_results['borderline'].get('best_f1', 0.0),
                'best_pr_auc': extra_results['borderline'].get('best_pr_auc', 0.0)
            },
            'monte_carlo': {
                'f1_names': extra_results['monte_carlo']['f1_names'],
                'pr_auc_names': extra_results['monte_carlo']['pr_auc_names'],
                'best_model_f1': extra_results['monte_carlo'].get('best_model_f1', 'N/A'),
                'best_model_pr_auc': extra_results['monte_carlo'].get('best_model_pr_auc', 'N/A'),
                'best_f1': mc_f1_score,
                'best_pr_auc': mc_pr_auc_score
            },
            'aggregation': {
                'robust_agg': robust_agg,
                'final_agg': full_aggregated
            },
            'final_decision': {
                'single_model': best_single_model,
                'ensemble': best_ensemble,
                'meta_model_type': meta_model_type,
                'single_model_f1': single_model_f1,
                'single_model_pr_auc': single_model_pr_auc,
                'ensemble_f1': ensemble_f1,
                'ensemble_pr_auc': ensemble_pr_auc,
                'ensemble_fitness': extra_results['ga']['fitness'],
                'framework_choice': framework_choice,
                'chosen_model': best_ensemble if framework_choice == 'ensemble' else best_single_model
            }
        }
        
        timing_dict = {
            'modules': extra_results.get('timing', {}),
            'total': e2e_time
        }
        
        logger.info("📝 STAGE 7/7: Writing Comprehensive Results...")
        # Write comprehensive results
        comp_results_dir = f"myresults/comprehensive/{dataset}/{entity}/"
        os.makedirs(comp_results_dir, exist_ok=True)
        comp_results_file = os.path.join(
            comp_results_dir, f"comprehensive_results_{dataset}_{entity}_iter{0}.txt"
        )
        write_comprehensive_results(comp_results_file, dataset, entity, 0, results_dict, timing_dict)
        logger.info(f"✓ Results written to: {comp_results_file}")
        logger.info("="*80)
        logger.info(f"🎉 EXECUTION COMPLETE! Total Time: {e2e_time:.4f}s ({e2e_time/60:.2f} min)")
        logger.info("="*80)

        # Real-time evaluation loop (iterations=1 means it won't run)
        # update_interval controls how often we re-run the expensive model selection pipeline
        update_interval = args.get('update_interval', 5)  # Re-optimize every N windows (default: 5)
        logger.info(f"📊 Online mode: iterations={iterations}, update_interval={update_interval}")
        
        i = 1
        while i < iterations:
            logger.info(f"🔄 Processing window {i}/{iterations-1}")
            test_data.entities[0].Y = data_windows[i]
            test_data.entities[0].labels = targets_windows[i]
            test_data.entities[0].mask = new_mask[i]
            test_data.entities[0].n_time = int(np.size(targets_windows[i].flatten()))
            test_data.total_time = int(np.size(targets_windows[i].flatten()))

            # Restrict to best ensemble's models
            trained_models_new = {}
            algorithm_list_new = []
            for model in best_ensemble:
                trained_models_new[model] = trained_models[model]
                algorithm_list_new.append(model)

            test_data_new = copy.deepcopy(test_data)
            test_data_new, _ = Inject(test_data_new, anomaly_list)

            # LIGHTWEIGHT: Evaluate current best single model on this window
            test_data_new_copy = copy.deepcopy(test_data_new)
            _, adjusted_y_pred_ind_current, _, _ = evaluate_individual_models(
                [full_aggregated[0]], test_data_new_copy, trained_models
            )

            # LIGHTWEIGHT: Evaluate ensemble fitness on current window
            test_data_new_copy = copy.deepcopy(test_data_new)
            values = fitness_function(
                best_ensemble, train_data, test_data_new_copy, trained_models_new,
                individual_predictions, base_model_predictions_train, algorithm_list_instances,
                base_model_predictions_test, y_true_train, y_true_test,
                meta_model_type=meta_model_type
            )

            # Persist falses (TP/FP/FN tracking)
            test_data_new_copy = copy.deepcopy(test_data_new)
            find_num_falses(
                adjusted_y_pred_ind_current, test_data_new_copy, dataset, entity, values,
                full_aggregated[0], best_ensemble, iteration=i
            )

            # EXPENSIVE: Re-run full model selection pipeline only every N windows
            if i % update_interval == 0:
                logger.info(f"  🔄 Triggering background re-optimization at window {i} (every {update_interval} windows)...")
                reopt_start = time.time()
                test_data_new_copy = copy.deepcopy(test_data_new)
                if use_parallel:
                    (best_thompson, robust_agg, full_aggregated, best_ensemble,
                     individual_predictions, base_model_predictions_train, base_model_predictions_test,
                     y_true_train, y_true_test, meta_model_type, _) = run_model_selection_algorithms_2(
                        train_data, test_data_new_copy, dataset, entity, iteration=i, 
                        trained_models=trained_models, model_list=loaded_model_names
                    )
                else:
                    (best_thompson, robust_agg, full_aggregated, best_ensemble,
                     individual_predictions, base_model_predictions_train, base_model_predictions_test,
                     y_true_train, y_true_test, meta_model_type, _) = run_model_selection_algorithms_1(
                        train_data, test_data_new_copy, dataset, entity, iteration=i, model_list=loaded_model_names
                    )
                logger.info(f"  ✓ Re-optimization completed in {time.time() - reopt_start:.2f}s")
                logger.info(f"  → Updated best_ensemble: {best_ensemble}")
                logger.info(f"  → Updated best_single_model: {full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated}")
            else:
                logger.info(f"  ⏩ Skipping re-optimization (next at window {((i // update_interval) + 1) * update_interval})")
            
            i += 1

    except Exception:
        logger.info('Traceback for Entity: %s Dataset: %s', entity, dataset)
        logger.error(traceback.format_exc())

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Global so the selection functions can access after initial load
    trained_models = {}
    run_app(algorithm_list, algorithm_list_instances)
