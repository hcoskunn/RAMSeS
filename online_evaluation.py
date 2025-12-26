"""
Online/Real-time Evaluation Module for RAMSeS

This module handles the real-time sliding window evaluation of models,
allowing adaptive model selection as new data arrives.
"""

import copy
import logging
import os
from typing import Tuple, Dict, Any, List, Callable

import matplotlib.pyplot as plt
import numpy as np

from Metrics.Ensemble_GA import evaluate_individual_models, fitness_function
from Model_Selection.inject_anomalies import Inject
from Model_Selection.Thompson_Sampling import initialize_sliding_windows

logger = logging.getLogger(__name__)


def visualize_injected_anomalies(test_data, test_data_before, anomaly_sizes, 
                                 dataset, entity, anomaly_list):
    """
    Create visualization of injected anomalies.
    
    Parameters
    ----------
    test_data : Dataset
        Test data with injected anomalies
    test_data_before : Dataset
        Original test data before injection
    anomaly_sizes : np.ndarray
        Array of anomaly magnitudes
    dataset : str
        Dataset name
    entity : str
        Entity name
    anomaly_list : list
        List of anomaly types injected
        
    Returns
    -------
    str : Path to saved figure
    """
    anomaly_start = int(np.argmax(test_data.entities[0].labels))
    anomaly_end = test_data.entities[0].Y.shape[1] - int(np.argmax(test_data.entities[0].labels[::-1]))
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
    
    # Plot original and anomalous data
    axes[0].plot(test_data.entities[0].Y.flatten(), label='Data with anomalies')
    axes[0].plot(np.arange(anomaly_start, anomaly_end),
                 test_data.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                 color='red', label='Anomaly region')
    axes[0].plot(np.arange(anomaly_start, anomaly_end),
                 test_data_before.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                 linestyle='--', color='blue', label='Original data')
    axes[0].set_title('Test data with Injected Anomalies', fontsize=16)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot anomaly scores
    axes[1].plot(anomaly_sizes.flatten(), label='Anomaly magnitude')
    axes[1].plot(test_data.entities[0].labels.flatten(), color='red', label='Anomaly labels')
    axes[1].set_title('Anomaly Scores', fontsize=16)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    out_dir = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_{anomaly_list}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved anomaly visualization to: {out_file}")
    return out_file


def setup_sliding_windows(test_data_before, iterations, min_length=256):
    """
    Setup sliding windows for online evaluation.
    
    Parameters
    ----------
    test_data_before : Dataset
        Original test data before anomaly injection
    iterations : int
        Number of sliding window iterations
    min_length : int
        Minimum window size
        
    Returns
    -------
    tuple : (data_windows, targets_windows, new_mask, num_windows, window_size, stride)
    """
    data = test_data_before.entities[0].Y
    targets = test_data_before.entities[0].labels
    mask = test_data_before.entities[0].mask
    
    # Calculate window size and stride safely
    total_size = np.size(targets.flatten())
    if iterations <= 0:
        iterations = 1
        logger.warning("Invalid iterations value, using default: 1")
    
    window_size = max(int(total_size / iterations), min_length)
    stride = max(window_size - 5, 1)  # Ensure stride is positive
    
    logger.info(f"Sliding windows: size={window_size}, stride={stride}, iterations={iterations}")
    
    data_windows, targets_windows, new_mask, num_windows = initialize_sliding_windows(
        data, targets, mask, window_size, stride
    )
    
    return data_windows, targets_windows, new_mask, num_windows, window_size, stride


def update_window_data(test_data, data_windows, targets_windows, new_mask, window_idx):
    """
    Update test_data with the specified window.
    
    Parameters
    ----------
    test_data : Dataset
        Dataset object to update
    data_windows : list
        List of data windows
    targets_windows : list
        List of target windows
    new_mask : list
        List of mask windows
    window_idx : int
        Index of window to use
    """
    test_data.entities[0].Y = data_windows[window_idx]
    test_data.entities[0].labels = targets_windows[window_idx]
    test_data.entities[0].mask = new_mask[window_idx]
    test_data.entities[0].n_time = int(np.size(targets_windows[window_idx].flatten()))
    test_data.total_time = int(np.size(targets_windows[window_idx].flatten()))


def compute_misclassifications(adjusted_y_pred_ind_current, test_data_copy, dataset, 
                               entity, values, full_aggregated, best_ensemble, iteration):
    """
    Compute and persist the number of misclassifications for current model(s) and ensemble.
    
    Parameters
    ----------
    adjusted_y_pred_ind_current : list
        Predictions from current best model
    test_data_copy : Dataset
        Copy of test data
    dataset : str
        Dataset name
    entity : str
        Entity name
    values : tuple
        Fitness function output (contains ensemble predictions)
    full_aggregated : list/str
        Current best aggregated model
    best_ensemble : list
        Current best ensemble
    iteration : int
        Current iteration number
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
        f.write("Summary of misclassifications:\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Chosen model (aggregated): {full_aggregated}\n")
        f.write(f"Misclassified by single model: {misclassified_current}\n")
        f.write("Falses for the ensemble:\n")
        f.write(f"Chosen ensemble: {best_ensemble}\n")
        f.write(f"Misclassified by ensemble: {misclassified_ensemble}\n")
    
    logger.info(f"[Iteration {iteration}] Single model misclassifications: {misclassified_current}")
    logger.info(f"[Iteration {iteration}] Ensemble misclassifications: {misclassified_ensemble}")


def run_online_evaluation(
    train_data,
    test_data,
    test_data_before,
    dataset: str,
    entity: str,
    trained_models: Dict[str, Any],
    algorithm_list_instances: List[str],
    selection_func: Callable,
    iterations: int,
    anomaly_list: List[str],
    args: Dict[str, Any],
    initial_results: Tuple = None,
    min_length: int = 256
) -> Tuple:
    """
    Run online/real-time evaluation with sliding windows.
    
    Parameters
    ----------
    train_data : Dataset
        Training data
    test_data : Dataset
        Test data (will be modified)
    test_data_before : Dataset
        Original test data before anomalies
    dataset : str
        Dataset name
    entity : str
        Entity name
    trained_models : dict
        Dictionary of trained models
    algorithm_list_instances : list
        List of algorithm instance names
    selection_func : callable
        Model selection function (sequential or parallel)
    iterations : int
        Number of sliding window iterations
    anomaly_list : list
        List of anomaly types to inject
    args : dict
        Command-line arguments
    initial_results : tuple, optional
        Initial results from first window (if already computed)
    min_length : int
        Minimum window size
        
    Returns
    -------
    tuple : Final iteration results (10-item tuple from selection function)
    """
    logger.info("=" * 80)
    logger.info(f"Starting online evaluation with {iterations} iterations")
    logger.info("=" * 80)
    
    # Setup sliding windows
    data_windows, targets_windows, new_mask, num_windows, window_size, stride = setup_sliding_windows(
        test_data_before, iterations, min_length
    )
    
    logger.info(f"Created {num_windows} windows for evaluation")
    
    # If no initial results provided, compute for first window
    if initial_results is None:
        logger.info("Computing initial model selection for window 0...")
        update_window_data(test_data, data_windows, targets_windows, new_mask, 0)
        
        test_data_new = copy.deepcopy(test_data)
        test_data_new, _ = Inject(test_data_new, anomaly_list)
        
        initial_results = selection_func(
            train_data, test_data_new, dataset, entity, iteration=0,
            trained_models_dict=trained_models,
            model_list_instances=algorithm_list_instances,
            population_size=args.get('population_size'),
            generations=args.get('generations'),
            thompson_iterations=args.get('thompson_iterations'),
            mc_simulations=args.get('mc_simulations'),
            noise_level=args.get('noise_level'),
            meta_model_type=args.get('meta_model_type', 'rf'),
            mutation_rate=args.get('mutation_rate'),
        )
    
    # Unpack initial results
    (best_thompson, robust_agg, full_aggregated, best_ensemble,
     individual_predictions, base_model_predictions_train, base_model_predictions_test,
     y_true_train, y_true_test, meta_model_type) = initial_results
    
    logger.info(f"Initial best model: {best_thompson}")
    logger.info(f"Initial best ensemble: {best_ensemble}")
    
    # Real-time evaluation loop
    i = 1
    while i < iterations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing window {i}/{iterations-1}")
        logger.info(f"{'='*60}")
        
        # Update test data with current window
        update_window_data(test_data, data_windows, targets_windows, new_mask, i)
        
        # Restrict to best ensemble's models
        trained_models_new = {}
        algorithm_list_new = []
        for model in best_ensemble:
            trained_models_new[model] = trained_models[model]
            algorithm_list_new.append(model)
        
        test_data_new = copy.deepcopy(test_data)
        test_data_new, _ = Inject(test_data_new, anomaly_list)
        
        # Evaluate current best single model
        logger.info(f"Evaluating current best model: {full_aggregated}")
        test_data_new_copy = copy.deepcopy(test_data_new)
        _, adjusted_y_pred_ind_current, _, _ = evaluate_individual_models(
            [full_aggregated[0]] if isinstance(full_aggregated, (list, tuple)) else [full_aggregated],
            test_data_new_copy, 
            trained_models
        )
        
        # Evaluate ensemble fitness on current window
        logger.info(f"Evaluating ensemble: {best_ensemble}")
        test_data_new_copy = copy.deepcopy(test_data_new)
        values = fitness_function(
            best_ensemble, train_data, test_data_new_copy, trained_models_new,
            individual_predictions, base_model_predictions_train, algorithm_list_new,
            base_model_predictions_test, y_true_train, y_true_test,
            meta_model_type=meta_model_type
        )
        
        # Compute and persist misclassifications
        test_data_new_copy = copy.deepcopy(test_data_new)
        compute_misclassifications(
            adjusted_y_pred_ind_current, test_data_new_copy, dataset, entity, values,
            full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated,
            best_ensemble, iteration=i
        )
        
        # Update selection for next iteration
        logger.info("Running model selection for next iteration...")
        test_data_new_copy = copy.deepcopy(test_data_new)
        (best_thompson, robust_agg, full_aggregated, best_ensemble,
         individual_predictions, base_model_predictions_train, base_model_predictions_test,
         y_true_train, y_true_test, meta_model_type) = selection_func(
            train_data, test_data_new_copy, dataset, entity, iteration=i,
            trained_models_dict=trained_models,
            model_list_instances=algorithm_list_instances,
            population_size=args.get('population_size'),
            generations=args.get('generations'),
            thompson_iterations=args.get('thompson_iterations'),
            mc_simulations=args.get('mc_simulations'),
            noise_level=args.get('noise_level'),
            meta_model_type=args.get('meta_model_type', 'rf'),
            mutation_rate=args.get('mutation_rate'),
        )
        
        logger.info(f"[Window {i}] Updated best model: {best_thompson}")
        logger.info(f"[Window {i}] Updated best ensemble: {best_ensemble}")
        
        i += 1
    
    logger.info("=" * 80)
    logger.info(f"Online evaluation completed: {iterations} windows processed")
    logger.info("=" * 80)
    
    return (best_thompson, robust_agg, full_aggregated, best_ensemble,
            individual_predictions, base_model_predictions_train, base_model_predictions_test,
            y_true_train, y_true_test, meta_model_type)


def run_single_shot_evaluation(
    train_data,
    test_data,
    dataset: str,
    entity: str,
    trained_models: Dict[str, Any],
    algorithm_list_instances: List[str],
    selection_func: Callable,
    anomaly_list: List[str],
    args: Dict[str, Any]
) -> Tuple:
    """
    Run single-shot (non-online) evaluation on the full test set.
    
    This is equivalent to iterations=1, processing the entire test data at once.
    
    Parameters
    ----------
    train_data : Dataset
        Training data
    test_data : Dataset
        Test data
    dataset : str
        Dataset name
    entity : str
        Entity name
    trained_models : dict
        Dictionary of trained models
    algorithm_list_instances : list
        List of algorithm instance names
    selection_func : callable
        Model selection function
    anomaly_list : list
        List of anomaly types to inject
    args : dict
        Command-line arguments
        
    Returns
    -------
    tuple : Results (10-item tuple from selection function)
    """
    logger.info("Running single-shot evaluation (no online updates)")
    
    test_data_new = copy.deepcopy(test_data)
    test_data_new, _ = Inject(test_data_new, anomaly_list)
    
    results = selection_func(
        train_data, test_data_new, dataset, entity, iteration=0,
        trained_models_dict=trained_models,
        model_list_instances=algorithm_list_instances,
        population_size=args.get('population_size'),
        generations=args.get('generations'),
        thompson_iterations=args.get('thompson_iterations'),
        mc_simulations=args.get('mc_simulations'),
        noise_level=args.get('noise_level'),
        meta_model_type=args.get('meta_model_type', 'rf'),
        mutation_rate=args.get('mutation_rate'),
    )
    
    logger.info("Single-shot evaluation completed")
    return results
