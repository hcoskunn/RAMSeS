import math
import os
import textwrap
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Utils.pipeline_spec import (abbreviate_detector, combine_metrics,
                                 DEFAULT_DECISION_METRICS, DETECTOR_GROUPS,
                                 family_of, GROUP_LABELS, group_of,
                                 metrics_required)
from Utils.plot_labels import draw_abbreviation_key

from Metrics.metrics import prauc, f1_score, vus_score, vus_window
from Utils.model_selection_utils import evaluate_model, ScoringTimeout
from Explainability import ir
from Utils.paths import results_dir


def initialize_population(algorithm_list, population_size):
    """
    Initialize the population for the genetic algorithm.

    Args:
        algorithm_list (list): List of available algorithms.
        population_size (int): Desired size of the population.

    Returns:
        list: Initialized population of unique ensembles.
    """
    population = []
    unique_ensembles = set()

    # Only size>1 ensembles are kept, so the pool is the 2^n - n - 1 subsets of
    # size >= 2. Asking for more than exist does not run slowly, it never returns.
    target = min(population_size, 2 ** len(algorithm_list) - len(algorithm_list) - 1)

    while len(population) < target:
        ensemble_size = random.randint(1, len(algorithm_list))
        ensemble = random.sample(algorithm_list, k=ensemble_size)
        ensemble = tuple(sorted(ensemble))  # Canonical ordering and convert to tuple for set operations

        if ensemble not in unique_ensembles and len(ensemble) > 1:
            unique_ensembles.add(ensemble)
            population.append(list(ensemble))  # Convert back to list for the population

    logger.info(f"Initialized population with {len(population)} unique ensembles")
    return population


def inject_synthetic_anomalies(y_true):
    """
    Inject synthetic anomalies into the training labels.

    Args:
        y_true (np.ndarray): Array of true labels (1D or 2D with shape (1, N)).

    Returns:
        np.ndarray: Array of true labels with injected anomalies (same shape as input).
    """
    # Handle 2D arrays with batch dimension
    original_shape = y_true.shape
    if y_true.ndim == 2:
        y_true = y_true.flatten()
    
    num_anomalies = int(len(y_true) / 10)
    # num_anomalies = 1
    indices = np.random.choice(len(y_true), num_anomalies, replace=False)
    y_true[indices] = 1
    
    # Restore original shape
    if len(original_shape) == 2:
        y_true = y_true.reshape(original_shape)
    
    return y_true


def train_meta_model(base_model_predictions, y_true):
    """
    Train a logistic regression meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        LogisticRegression: Trained logistic regression meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = LogisticRegression()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Logistic Regression meta-model")
    return meta_model


def train_meta_model_rf(base_model_predictions, y_true):
    """
    Train a random forest meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        RandomForestClassifier: Trained random forest meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = RandomForestClassifier()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Random Forest meta-model")
    return meta_model


def train_meta_model_gbm(base_model_predictions, y_true):
    """
    Train a gradient boosting machine meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        GradientBoostingClassifier: Trained gradient boosting machine meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = GradientBoostingClassifier()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Gradient Boosting Machine meta-model")
    return meta_model


def train_meta_model_svm(base_model_predictions, y_true):
    """
    Train a support vector machine (SVM) meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        SVC: Trained SVM meta-model.
    """
    meta_model = SVC(probability=True)
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained SVM meta-model")
    return meta_model


def evaluate_model_consistently(data, model, model_name, is_ensemble=False):
    """
    Consistently evaluate a model or ensemble of models on the given data.

    Args:
        data: Dataset for evaluation.
        model: The model or ensemble of models to evaluate.
        model_name (str or list): Name of the model or list of model names for ensemble.
        is_ensemble (bool): Flag indicating if the model is an ensemble.

    Returns:
        tuple: True labels and predictions.
    """
    # Debug: Check raw labels before evaluation
    if hasattr(data, 'entities') and len(data.entities) > 0:
        raw_labels = data.entities[0].labels
        logger.info(f"RAW LABELS in data before evaluation: shape={raw_labels.shape}, unique={np.unique(raw_labels)}, sum={np.sum(raw_labels)}")
    
    y_true_agg_dict = {}
    base_model_predictions_dict = {}
    if is_ensemble:
        y_true_agg = None
        base_model_predictions = []

        for sub_model_name in model_name:
            sub_model = model.get(sub_model_name)
            if sub_model:
                try:
                    evaluation = evaluate_model(data, sub_model, sub_model_name)
                    y_true = evaluation['anomaly_labels'].flatten()
                    y_scores = evaluation['entity_scores'].flatten()
                    
                    # Validate shapes match
                    if len(y_true) != len(y_scores):
                        logger.error(f"Shape mismatch for {sub_model_name}: y_true={len(y_true)}, y_scores={len(y_scores)}")
                        logger.error(f"  Data shape: {data.entities[0].Y.shape}, labels: {data.entities[0].labels.shape}")
                        continue
                    
                    base_model_predictions.append(y_scores)
                    base_model_predictions_dict[sub_model_name] = y_scores
                    if y_true_agg is None:
                        y_true_agg = y_true
                        y_true_agg_dict[sub_model_name] = y_true
                except Exception as e:
                    logger.error(f"Inference failed for {sub_model_name}: {e}")
                    continue
                    
        if len(base_model_predictions) == 0:
            logger.error("No successful model evaluations in ensemble")
            return None, None, {}, {}
            
        base_model_predictions = np.array(base_model_predictions).T
        return y_true_agg, base_model_predictions, y_true_agg_dict, base_model_predictions_dict
    else:
        evaluation = evaluate_model(data, model, model_name)
        y_true = evaluation['anomaly_labels'].flatten()
        y_scores = evaluation['entity_scores'].flatten()

        return y_true, y_scores, y_true_agg_dict, base_model_predictions_dict


def evaluate_individual_models(algorithm_list, test_data, trained_models):
    """
    Evaluate individual models on the test data.

    Args:
        algorithm_list (list): List of algorithm names.
        test_data: Test dataset.
        trained_models (dict): Dictionary of trained models.

    Returns:
        dict: Predictions from individual models.
    """
    predictions = {}
    adjusted_y_pred_list = []
    F1_Score_list = []
    PR_AUC_Score_list = []
    for model_name in algorithm_list:
        model = trained_models.get(model_name)
        if model:
            try:
                y_true, y_scores, y_true_agg_dict, y_scores_dict = evaluate_model_consistently(test_data, model, model_name)
            except ScoringTimeout:
                continue
            
            # Debug: Check y_true
            logger.info(f"y_true shape: {np.array(y_true).shape}, unique values: {np.unique(y_true)}, sum: {np.sum(y_true)}")
            logger.info(f"y_scores shape: {np.array(y_scores).shape}, min: {np.min(y_scores)}, max: {np.max(y_scores)}, mean: {np.mean(y_scores)}")
            
            # Use range-based metric (segment-aware F1 + PR-AUC) — same as GAN/Borderline/MC
            # robustness tests, which previously produced non-zero F1 on injected data.
            # Strict standard F1 collapses to 0 on synthetic spikes for unsupervised base models,
            # making the single-model branch incomparable to the supervised GA ensemble.
            from Metrics.metrics import range_based_precision_recall_f1_auc
            _, _, best_f1, pr_auc, y_pred_binary = range_based_precision_recall_f1_auc(
                np.asarray(y_true).flatten(), np.asarray(y_scores).flatten()
            )
            
            logger.info(f"Model {model_name}: F1 score = {best_f1}, PR AUC = {pr_auc}")
            predictions[model_name] = (y_true, y_scores)
            adjusted_y_pred_list.append(y_pred_binary)  # Use binary predictions instead of scores
            F1_Score_list.append(best_f1)
            PR_AUC_Score_list.append(pr_auc)
            logger.info(f"First 10 scores for model {model_name}: {y_scores[:10]}")
            logger.info(f"First 10 true labels for model {model_name}: {y_true[:10]}")
    return predictions, adjusted_y_pred_list, F1_Score_list, PR_AUC_Score_list


def fitness_function(ensemble, train_data, test_data, trained_models,
                     individual_predictions,
                     base_model_predictions_train, algorithm_list,
                     base_model_predictions_test, y_true_train, y_true_test,
                     meta_model_type='svm', metric=DEFAULT_DECISION_METRICS,
                     vus_win=None):
    """
    Evaluate the fitness of an ensemble.

    Args:
        ensemble (list): List of model names in the ensemble.
        train_data: Training dataset.
        test_data: Test dataset.
        trained_models (dict): Dictionary of trained models.
        individual_predictions (list): Predictions from individual base models.
        meta_model_type (str): Type of meta-model to use ('lr', 'rf', 'gbm', 'svm').

    Returns:
        tuple: (best_f1, pr_auc, fitness, y_scores, y_true_test, meta_model).
        The trained meta_model is included so downstream explainability can
        attribute the exact meta-learner without retraining.
    """
    logger.info(f"Evaluating fitness for ensemble: {ensemble}")

    # Sort the ensemble to ensure canonical ordering
    ensemble = sorted(ensemble)

    # Evaluate ensemble on training data

    header_array_train = np.array(algorithm_list)

    # Determine which headers are in the ensemble
    desired_mask_train = np.isin(header_array_train, ensemble)

    # Filter the columns of data array based on the desired headers
    base_model_predictions_train = base_model_predictions_train[:, desired_mask_train]

    # -----

    # Convert the headers to a NumPy array for vectorized operations
    header_array_test = np.array(algorithm_list)

    # Determine which headers are in the ensemble
    desired_mask_test = np.isin(header_array_test, ensemble)

    # Filter the columns of data array based on the desired headers
    base_model_predictions_test = base_model_predictions_test[:, desired_mask_test]
    # Inject synthetic anomalies if the training labels have only one class
    if len(np.unique(y_true_train)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the training labels. Injecting synthetic anomalies.")
        y_true_train = inject_synthetic_anomalies(y_true_train)

    # Train the meta-model based on the specified type
    if meta_model_type == 'lr':
        meta_model = train_meta_model(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'rf':
        meta_model = train_meta_model_rf(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'gbm':
        meta_model = train_meta_model_gbm(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'svm':
        meta_model = train_meta_model_svm(base_model_predictions_train, y_true_train)
    else:
        raise ValueError(f"Unknown meta_model_type: {meta_model_type}")

    # Evaluate ensemble on test data
    # y_true_test, base_model_predictions_test, y_true_test_dict, base_model_predictions_test_dict = evaluate_model_consistently(
    #     test_data, trained_models, ensemble,
    #     is_ensemble=True)

    # Inject synthetic anomalies if the test labels have only one class
    if len(np.unique(y_true_test)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the test labels. Injecting synthetic anomalies.")
        y_true_test = inject_synthetic_anomalies(y_true_test)

    # Clean test predictions: replace inf/nan values before prediction
    base_model_predictions_test = np.array(base_model_predictions_test, dtype=np.float32)
    base_model_predictions_test = np.nan_to_num(base_model_predictions_test, 
                                                 nan=0.0, 
                                                 posinf=1.0, 
                                                 neginf=0.0)

    # Generate prediction scores using the meta-model
    y_scores = meta_model.predict_proba(base_model_predictions_test)[:, 1]

    # Convert probabilities to binary predictions using optimal threshold
    # We need to threshold y_scores for F1 calculation
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_temp_f1 = 0
    for threshold in thresholds:
        y_pred_binary = (y_scores >= threshold).astype(int)
        temp_f1, _, _, _, _, _, _ = f1_score(y_pred_binary, y_true_test)
        if temp_f1 > best_temp_f1:
            best_temp_f1 = temp_f1
            best_threshold = threshold
    
    # Use the best threshold for final predictions
    y_pred_binary = (y_scores >= best_threshold).astype(int)
    
    # Calculate evaluation metrics: F1 score and PR AUC
    # _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true_test, y_scores)
    best_f1, precision, recall, TP, TN, FP, FN = f1_score(y_pred_binary, y_true_test)
    # best_f1, precision, recall, TP, TN, FP, FN = f1_soft_score(y_scores, y_true_test)
    # best_f1 = get_composite_fscore_raw(y_scores, y_true_test)
    pr_auc = prauc(y_true_test, y_scores)

    needed = metrics_required(metric)
    vus = (vus_score(y_scores, y_true_test, vus_win)
           if 'vus' in needed and vus_win is not None else float('nan'))
    fitness = combine_metrics(metric, {'f1': best_f1, 'pr_auc': pr_auc, 'vus': vus})
    if np.isnan(fitness):
        fitness = best_f1
    logger.info(
        f"Evaluated fitness for ensemble {ensemble} with F1 score {best_f1} and PR AUC {pr_auc}, resulting in fitness {fitness}")
    # The trained meta-model is appended (6th element) so the combination-
    # explainability layer can attribute the EXACT meta-learner the GA built,
    # rather than retraining one. Existing callers index [0..4] and are unaffected.
    return best_f1, pr_auc, fitness, y_scores, y_true_test, meta_model


def score_ensemble(meta_model, ensemble, algorithm_list, base_model_predictions, y_true):
    """Score an already-trained meta-model on one set of base-model outputs.

    The threshold is swept on the rows being scored, matching what the single-model
    branch does, so the two sides of the final decision stay comparable.

    Returns:
        tuple: (best_f1, pr_auc, y_scores).
    """
    mask = np.isin(np.array(algorithm_list), ensemble)
    X = np.array(base_model_predictions, dtype=np.float32)[:, mask]
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    y_true = np.asarray(y_true).flatten()
    y_scores = meta_model.predict_proba(X)[:, 1]

    best_threshold = 0.5
    best_temp_f1 = 0
    for threshold in np.linspace(0.1, 0.9, 50):
        temp_f1, _, _, _, _, _, _ = f1_score((y_scores >= threshold).astype(int), y_true)
        if temp_f1 > best_temp_f1:
            best_temp_f1 = temp_f1
            best_threshold = threshold

    best_f1, _, _, _, _, _, _ = f1_score((y_scores >= best_threshold).astype(int), y_true)
    return best_f1, prauc(y_true, y_scores), y_scores


def selection(population, fitness_scores, num_selected):
    """
    Select the top ensembles based on fitness scores.

    Args:
        population (list): List of ensembles.
        fitness_scores (list): List of fitness scores corresponding to the population.
        num_selected (int): Number of ensembles to select.

    Returns:
        list: Selected top ensembles.
    """
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    selected = [population[i] for i in selected_indices]
    logger.info(f"Selected top {num_selected} ensembles with scores {fitness_scores}")
    return selected


def crossover(parent1, parent2):
    """
    Perform crossover between two parent ensembles to create a child ensemble.

    Args:
        parent1 (list): First parent ensemble.
        parent2 (list): Second parent ensemble.

    Returns:
        list: Child ensemble resulting from the crossover.
    """
    crossover_point1 = random.randint(1, len(parent1))
    crossover_point2 = random.randint(1, len(parent2))
    child = parent1[:crossover_point1] + parent2[crossover_point2:]
    child = list(set(child))
    child = sorted(child)
    logger.info(f"Crossover parents {parent1} and {parent2} to create child {child}")
    return child


def mutate(ensemble, mutation_rate, algorithm_list):
    """
    Perform mutation on an ensemble.

    Args:
        ensemble (list): Ensemble to mutate.
        mutation_rate (float): Mutation rate.
        algorithm_list (list): List of available algorithms.

    Returns:
        list: Mutated ensemble.
    """
    mutated_ensemble = ensemble.copy()
    for i in range(len(mutated_ensemble)):
        if random.random() < mutation_rate:
            available_models = [model for model in algorithm_list if model not in mutated_ensemble]
            if available_models:
                original_model = mutated_ensemble[i]
                mutated_ensemble[i] = random.choice(available_models)
                logger.info(f"Mutated model {original_model} to {mutated_ensemble[i]} in ensemble {ensemble}")
            else:
                logger.warning(f"No available models to mutate in the ensemble: {mutated_ensemble}")

    if random.random() < mutation_rate:
        if len(mutated_ensemble) > 3 and random.random() > 0.5:
            model_to_remove = random.choice(mutated_ensemble)
            mutated_ensemble.remove(model_to_remove)
            logger.info(f"Removed model {model_to_remove} from ensemble {ensemble}")
        else:
            possible_models = [model for model in algorithm_list if model not in mutated_ensemble]
            if possible_models:
                model_to_add = random.choice(possible_models)
                mutated_ensemble.append(model_to_add)
                logger.info(f"Added model {model_to_add} to ensemble {ensemble}")
            else:
                logger.warning(f"No available models to add to the ensemble: {mutated_ensemble}")

    mutated_ensemble = sorted(mutated_ensemble)  # Ensure canonical ordering after mutation
    return mutated_ensemble


def plot_scores_vs_true(data, F1_Score_list, PR_AUC_Score_list, adjusted_y_pred, list_ensemble, plot_name, plot_path):
    max_arg_f1 = np.argmax(np.array(F1_Score_list))
    max_arg_pr_auc = np.argmax(np.array(PR_AUC_Score_list))
    print(data.entities[0].labels)
    print(adjusted_y_pred[max_arg_f1])
    true_values = np.array(data.entities[0].labels)  # 1 for anomaly, 0 for normal
    print(10 * '=')
    print(true_values)
    predicted_values = np.array(
        adjusted_y_pred[max_arg_f1])  # True for predicted anomaly, False for no predicted anomaly

    # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
    predicted_int = predicted_values.astype(int)
    best_ = list_ensemble[max_arg_f1]
    # Identifying incorrect predictions
    incorrect_predictions = predicted_int != true_values
    misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
    total_anomalies = np.sum(true_values)  # Total number of real anomalies
    total_data = len(true_values)  # Total number of data points

    # Highlight incorrect predictions with a different marker
    print(incorrect_predictions)
    if incorrect_predictions.ndim == 2:
        pass
    else:

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, '.', label='True Values (Anomalies)', color='blue')  # Plot true values
        # plt.plot(predicted_int, 'x', label='Predicted Values (Anomalies)', color='red')  # Plot predicted values
        plt.scatter(np.where(incorrect_predictions)[0], predicted_int[incorrect_predictions], facecolors='none',
                    edgecolors='purple', s=100, label='Incorrect Predictions', linewidth=2)
        plt.title(
            f'True vs. Predicted Anomalies \n Misclassified Anomalies: {misclassified_count}, {best_}\n Total Anomalies: {total_anomalies} \n Total Data: {total_data}')
        plt.xlabel('Index')
        plt.ylabel('Anomaly Presence')
        plt.yticks([0, 1], ['No Anomaly', 'Anomaly'])  # Set y-ticks to be explicit about what 0 and 1 represent
        plt.legend()
        plt.grid(True)
        # Specify the directory
        directory = plot_path
        filename = plot_name
        full_path = os.path.join(directory, filename)

        # Get the directory path for the file
        output_dir = os.path.dirname(full_path)

        # Create all directories in the path if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

        # plt.show()


def plot_models_scores(algorithm_list, test_data, y_scores_list, dataset, entity, F1_Score_list_ind_curent,
                       PR_AUC_Score_list_ind_curent):
    data = test_data.entities[0].Y
    targets = test_data.entities[0].labels

    # Ensure unique algorithms and corresponding values
    unique_algorithms = []
    unique_y_scores_list = []
    unique_F1_Score_list = []
    unique_PR_AUC_Score_list = []

    seen = set()
    for i, algorithm in enumerate(algorithm_list):
        if algorithm not in seen:
            seen.add(algorithm)
            unique_algorithms.append(algorithm)
            unique_y_scores_list.append(y_scores_list[i])
            unique_F1_Score_list.append(F1_Score_list_ind_curent[i])
            unique_PR_AUC_Score_list.append(PR_AUC_Score_list_ind_curent[i])

    # Determine the number of rows needed
    num_algorithms = len(unique_algorithms)
    num_rows = 2 + num_algorithms  # 2 for original data and labels, rest for each algorithm

    # Plot the data
    fig, axes = plt.subplots(num_rows, 1, figsize=(18, 4 * num_rows), sharex=True)

    # First row: plot the data
    axes[0].plot(data.flatten(), label='Data', color='blue')
    axes[0].set_title('Data')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    # Second row: plot the labels with spikes
    axes[1].plot(targets, label='Labels', color='gray')
    spike_indices = np.where(targets == 1)[0]
    spike_values = np.ones_like(spike_indices)  # Set spikes at 1 for visibility
    axes[1].vlines(spike_indices, ymin=0, ymax=spike_values, color='red', label='Anomalies')
    axes[1].set_title('Labels')
    axes[1].set_ylabel('Label')
    axes[1].grid(True)

    # Loop over the unique y_scores_list and plot each under the original labels
    for i, algorithm in enumerate(unique_algorithms):
        y_scores = unique_y_scores_list[i]
        f1_score_value = unique_F1_Score_list[i]
        pr_auc_value = unique_PR_AUC_Score_list[i]

        # Plot the y_scores
        axes[i + 2].plot(y_scores, label=f'{algorithm} Scores', color='gray')

        # Identify spikes, true positives, false positives, and false negatives
        spike_indices = np.where(y_scores >= 0.5)[0]
        true_positive_indices = np.intersect1d(spike_indices, np.where(targets == 1)[0])
        false_positive_indices = np.setdiff1d(spike_indices, true_positive_indices)
        false_negative_indices = np.setdiff1d(np.where(targets == 1)[0], true_positive_indices)

        # Plot detected anomalies
        axes[i + 2].vlines(spike_indices, ymin=0, ymax=1, color='red', label='Detected Anomalies')

        # Highlight true positives with a different color
        # axes[i + 2].vlines(true_positive_indices, ymin=0, ymax=1, color='green', label='True Positives')

        # Highlight false positives with a different color
        # axes[i + 2].vlines(false_positive_indices, ymin=0, ymax=1, color='orange', label='False Positives')

        # Highlight false negatives with a different color
        # axes[i + 2].vlines(false_negative_indices, ymin=0, ymax=1, color='purple', label='False Negatives')

        axes[i + 2].set_title(f'{algorithm} Anomaly Scores, F1 Score = {f1_score_value}, PR AUC = {pr_auc_value}')
        axes[i + 2].set_ylabel('Score')
        axes[i + 2].grid(True)

    # Add legend to the last axis
    handles, labels = axes[1].get_legend_handles_labels()
    detected_handles, detected_labels = axes[2].get_legend_handles_labels()
    combined_handles = handles + detected_handles
    combined_labels = labels + detected_labels
    fig.legend(combined_handles, combined_labels, loc='upper right')

    axes[-1].set_xlabel('Time (index)')

    plt.tight_layout()
    directory = results_dir("Thomposon", dataset, entity)
    if not os.path.exists(directory):
        os.makedirs(directory)
    #plt.savefig(f'{directory}/performance_plot_nigg.png')
    # plt.show()


def genetic_algorithm(dataset, entity, train_data, test_data, algorithm_list, trained_models, meta_model_type,
                      population_size, generations, mutation_rate, explain: bool = False,
                      metric=DEFAULT_DECISION_METRICS, vus_win=None):
    """
    Run the genetic algorithm to find the best ensemble of models.

    The meta-learner is trained on train_data and every candidate subset is
    scored on test_data, which is also what the final ensemble-vs-single
    decision reads.

    Args:
        dataset (str): Dataset name.
        entity (str): Entity name.
        train_data: Training dataset.
        test_data: Test dataset.
        algorithm_list (list): List of algorithm names.
        trained_models (dict): Dictionary of trained models.
        meta_model_type (str): Type of meta-model to use ('lr', 'rf', 'gbm', 'svm').
        population_size (int): Size of the population.
        generations (int): Number of generations.
        mutation_rate (float): Mutation rate.

    Returns:
        tuple: Best ensemble, best F1 score, best PR AUC, and best fitness score.
    """
    # mevaluation_instance = Mevaluation()
    # mevaluation_instance.evaluate_model(train_data, test_data)  # Evaluate individual models before GA

    individual_predictions, adjusted_y_pred_ind, F1_Score_list_ind, PR_AUC_Score_list_ind = evaluate_individual_models(
        algorithm_list, test_data, trained_models)
    # Ensembles are bred from algorithm_list but scored from individual_predictions,
    # so a detector that never scored has to leave both.
    dropped = [m for m in algorithm_list if m not in individual_predictions]
    if dropped:
        algorithm_list = [m for m in algorithm_list if m in individual_predictions]
        logger.warning(f"⚠ Excluded from the GA pool (no score): {', '.join(dropped)}")
    logger.info(f"  ✓ Individual model evaluation complete ({len(algorithm_list)} models)")
    
    # Skip plotting to save time (but keep path definitions for later use)
    # logger.info(f"  → Plotting individual model scores...")
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_name = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_UMS_{date_time_string}.png'
    plot_path = results_dir("Outputs", "GA_Ens", dataset, entity)
    # plot_scores_vs_true(test_data, F1_Score_list_ind, PR_AUC_Score_list_ind, adjusted_y_pred_ind, algorithm_list,
    #                     plot_name, plot_path)
    # logger.info(f"  ✓ Plot saved to {plot_path}/{plot_name}")
    
    import time as time_module
    # individual_predictions = []
    evaluated_ensembles = {}  # HashMap to track evaluated ensembles and their scores
    file_name = results_dir("Outputs", "GA_Ens") + f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_{date_time_string}.txt'

    best_f1 = 0
    best_pr_auc = 0
    best_fitness = 0
    best_ensemble = None
    best_meta_model = None   # the trained meta-model of the best ensemble (for combination explainability)
    best_scores = None       # its continuous scores, for the final decision's VUS
    adjusted_y_pred_list = []
    F1_Score_list = []
    list_ensemble = []
    PR_AUC_Score_list = []
    fitness_list = []
    # Per-generation populations — used by the selection-explainability layer
    # (Axis 2: evolutionary survival rate). Snapshotted at the top of each loop
    # iteration, before fitness evaluation, so it reflects the population OF
    # that generation rather than the next one's offspring.
    generation_populations: List[List[List[str]]] = []
    # Per-generation fitness for the convergence figure, recorded after each
    # generation is scored.
    generation_fitness: List[Dict[str, float]] = []

    # Prepare training data for meta-model
    logger.info(f"  → Evaluating all {len(algorithm_list)} models on TRAINING data (for meta-model training)...")
    start_train = time_module.time()
    y_true_train, base_model_predictions_train, y_true_train_dict, base_model_predictions_train_dict = evaluate_model_consistently(
        train_data,
        trained_models,
        algorithm_list,
        is_ensemble=True)
    logger.info(f"  ✓ Training data evaluation complete in {time_module.time() - start_train:.2f}s")

    usable_detectors = [m for m in algorithm_list
                        if m in base_model_predictions_train_dict
                        and m in individual_predictions]
    if not usable_detectors:
        raise ValueError("No detector produced usable predictions on both splits.")
    if len(usable_detectors) < len(algorithm_list):
        dropped = [m for m in algorithm_list if m not in usable_detectors]
        logger.warning(
            f"  ⚠ {len(dropped)} detector(s) produced no usable predictions and are "
            f"excluded from the GA: {', '.join(dropped)}")
        base_model_predictions_train = np.array(
            [base_model_predictions_train_dict[m] for m in usable_detectors]).T
        algorithm_list = usable_detectors

    logger.info(f"  → Initializing GA population (size={population_size})...")
    start_init = time_module.time()
    population = initialize_population(algorithm_list, population_size)
    logger.info(f"  ✓ Population initialized in {time_module.time() - start_init:.2f}s")
    print(population)

    # Reuse test predictions from individual_predictions instead of re-computing
    logger.info(f"  → Reusing test predictions from individual evaluation (no re-computation)...")
    start_reuse = time_module.time()
    y_true_test = None
    base_model_predictions_test_list = []
    for model_name in algorithm_list:
        if model_name in individual_predictions:
            y_true, y_scores = individual_predictions[model_name]
            base_model_predictions_test_list.append(y_scores)
            if y_true_test is None:
                y_true_test = y_true
    base_model_predictions_test = np.array(base_model_predictions_test_list).T
    logger.info(f"  ✓ Test predictions reused in {time_module.time() - start_reuse:.4f}s (vs ~{start_train - start_train:.2f}s if recomputed)")
    
    logger.info(f"  ✓ Meta-model preparation complete. Starting GA generations...")
    print("y_true_train mine")
    print(y_true_train)
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}")
        print(f"Generation {generation + 1}")

        if explain:
            generation_populations.append([list(ind) for ind in population if ind is not None])

        fitness_results = []
        for ensemble in population:
            if ensemble is not None:  # Ensure ensemble is not None
                ensemble_key = tuple(sorted(ensemble))  # Create a unique key for the ensemble
                if ensemble_key not in evaluated_ensembles:
                    fitness_result = fitness_function(ensemble, train_data, test_data, trained_models,
                                                      individual_predictions,
                                                      base_model_predictions_train, algorithm_list,
                                                      base_model_predictions_test, y_true_train, y_true_test,
                                                      meta_model_type=meta_model_type,
                                                      metric=metric, vus_win=vus_win)
                    evaluated_ensembles[ensemble_key] = fitness_result


                else:
                    fitness_result = evaluated_ensembles[ensemble_key]
                adjusted_y_pred = fitness_result[3]
                adjusted_y_pred_list.append(adjusted_y_pred)
                F1_Score_list.append(fitness_result[0])
                list_ensemble.append(ensemble_key)
                PR_AUC_Score_list.append(fitness_result[1])
                fitness_list.append(fitness_result[2])
                fitness_results.append(fitness_result)

        fitness_scores = [result[2] for result in fitness_results]
        f1_scores = [result[0] for result in fitness_results]
        pr_aucs = [result[1] for result in fitness_results]

        if explain:
            finite = [s for s in fitness_scores if not np.isnan(s)]
            generation_fitness.append({
                'generation': generation + 1,
                'best': max(finite) if finite else float('nan'),
                'mean': float(np.mean(finite)) if finite else float('nan'),
                'worst': min(finite) if finite else float('nan'),
                'n_evaluated': len(finite),
            })

        print(f"Fitness Scores: {fitness_scores}")

        selected = selection(population, fitness_scores, max(1, population_size // 2))
        new_population = selected.copy()

        while len(new_population) < population_size:
            if len(selected) > 1:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
            else:
                child = selected[0]
            child = mutate(child, mutation_rate, algorithm_list)
            new_population.append(child)

        

        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_f1 = f1_scores[best_idx]
            best_pr_auc = pr_aucs[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_ensemble = population[best_idx]
            # Capture the meta-model that achieved this best fitness (index 5 of the
            # fitness tuple) for the combination-explainability layer. Aligned with
            # f1_scores / fitness_scores, which are all derived from fitness_results.
            best_meta_model = fitness_results[best_idx][5]
            best_scores = fitness_results[best_idx][3]
        population = new_population

        logger.info(f"End of Generation {generation + 1}, Population: {population}")
        print(f"End of Generation {generation + 1}, Population: {population}")

    misclassified_ens = []
    for predicts in adjusted_y_pred_list:
        true_values = np.array(test_data.entities[0].labels)  # 1 for anomaly, 0 for normal

        predicted_values = np.array(predicts)  # True for predicted anomaly, False for no predicted anomaly

        # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
        predicted_int = predicted_values.astype(int)

        # Identifying incorrect predictions
        incorrect_predictions = predicted_int != true_values
        misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
        misclassified_ens.append(misclassified_count)
    directory = plot_path
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"GA_Differeneces_results_{dataset}_{entity}.txt")
    #with open(output_file, 'w') as f:
     #   f.write("All Ensembles:\n")
      #  f.write(f" Evaluated Ensembles: {evaluated_ensembles.keys()}\n")
       # f.write("All Differences:\n")
        #f.write(f"{misclassified_ens}")
    ensemble_names = [name for name in evaluated_ensembles.keys()]
    f1_scores = [result[0] for result in evaluated_ensembles.values()]
    pr_auc_scores = [result[1] for result in evaluated_ensembles.values()]
    flat_ensemble_names = ['_'.join(names) for names in ensemble_names]
    plot_name = results_dir("Outputs", "GA_Ens") + f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_ensemble_{date_time_string}.png'
    plot_models_scores(list_ensemble, test_data, adjusted_y_pred_list, dataset, entity, F1_Score_list,
                       PR_AUC_Score_list)
    plot_scores_vs_true(test_data, F1_Score_list, PR_AUC_Score_list, adjusted_y_pred_list, list_ensemble, plot_name,
                        plot_path)
    # Plot for F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(flat_ensemble_names, f1_scores, marker='o', linestyle='-', color='b')
    plt.title('F1 Scores of Ensembles')
    plt.xlabel('Ensemble Name')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    # Specify the directory
    directory = plot_path
    filename = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_F1_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    # plt.show()
    #
    # # Plot for PR_AUC scores
    plt.figure(figsize=(10, 5))
    plt.plot(flat_ensemble_names, pr_auc_scores, marker='o', linestyle='-', color='r')
    plt.title('PR_AUC Scores of Ensembles')
    plt.xlabel('Ensemble Name')
    plt.ylabel('PR AUC Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Specify the directory
    directory = plot_path
    filename = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_PR_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    # plt.show()
    logger.info(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    print(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    # Sort evaluated_ensembles by fitness score before writing to the file
    sorted_ensembles = sorted(evaluated_ensembles.items(), key=lambda x: x[1][2], reverse=True)
    # Save the results to a text file

   # with open(file_name, "w") as f:
    #    for ensemble, result in sorted_ensembles:
     #       f.write(f"Ensemble: {list(ensemble)}, f1 : {result[0]}, PR_AUC: {result[1]}, Fitness Score: {result[2]}\n")

    if explain and best_ensemble:
        explain_ga_selection(best_ensemble, evaluated_ensembles, generation_populations,
                             algorithm_list, population_size,
                             dataset, entity, explain=True,
                             base_fit=base_model_predictions_train, y_fit=y_true_train,
                             base_eval=base_model_predictions_test, y_eval=y_true_test,
                             meta_model_type=meta_model_type,
                             metric=metric, vus_win=vus_win,
                             generation_fitness=generation_fitness)

        explain_ga_combination(best_ensemble, algorithm_list,
                               base_model_predictions_train, base_model_predictions_test,
                               y_true_train, y_true_test, meta_model_type,
                               dataset, entity, meta_model=best_meta_model, explain=True,
                               metric=metric, vus_win=vus_win)

    return best_ensemble, best_f1, best_pr_auc, best_fitness, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type, best_scores

# Usage
# Assuming train_data and test_data are already loaded and preprocessed
# algorithm_list = ['LOF', 'NN', 'RNN']
# trained_models = {'LOF': lof_model, 'NN': nn_model, 'RNN': rnn_model}
# best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(train_data, test_data, algorithm_list, trained_models, meta_model_type='lr')
# You can change meta_model_type to 'rf', 'gbm', or 'svm' for Random Forest, Gradient Boosting Machine, and SVM respectively


# ════════════════════════════════════════════════════════════════════════════
#  GA Ensemble Selection Explainability
#
#  Two analytical axes per candidate detector:
#    1. Utility    — mean marginal contribution across evaluated subsets.
#    2. Stability  — evolutionary survival rate per generation:
#                    P(d_j, g) = (#individuals in gen g containing d_j) /
#                                 population_size.
# ════════════════════════════════════════════════════════════════════════════


def _conditional_mean_fitness(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    present: Tuple[str, ...] = (),
    absent: Tuple[str, ...] = (),
) -> Tuple[float, int, float]:
    """Mean of subset fitness over evaluated subsets that contain every detector
    in `present` and none in `absent`. Returns (mean, count, variance). When
    count == 0 the mean is NaN; the variance needs two subsets."""
    present_set = set(present)
    absent_set = set(absent)
    vals = []
    for key, result in evaluated_ensembles.items():
        members = set(key)
        if present_set.issubset(members) and absent_set.isdisjoint(members):
            vals.append(float(result[2]))   # result = (f1, pr_auc, fitness, ...)
    if not vals:
        return float('nan'), 0, float('nan')
    var = float(np.var(vals, ddof=1)) if len(vals) > 1 else float('nan')
    return float(np.mean(vals)), len(vals), var


def measure_refit_noise(
    ensemble: List[str],
    algorithm_list: List[str],
    base_fit: np.ndarray,
    y_fit: np.ndarray,
    base_eval: np.ndarray,
    y_eval: np.ndarray,
    repeats: int = 30,
    meta_model_type: str = 'rf',
    metric: str = DEFAULT_DECISION_METRICS,
    vus_win: Optional[int] = None,
) -> Dict[str, float]:
    """
    Spread of the fitness when the meta-learner is refitted on unchanged data.

    The meta-learner is not seeded, so the same subset scores differently on
    every refit. Nothing else varies across the repeats, so the spread is the
    floor below which a fitness difference says nothing about the detectors.

    Returns {'eps', 'sigma', 'repeats'}, with eps at two sigma. NaN when the
    ensemble is empty or every repeat failed.
    """
    nan = float('nan')
    empty = {'eps': nan, 'sigma': nan, 'repeats': 0}
    if not ensemble:
        return empty
    evals = []
    for _ in range(max(1, repeats)):
        try:
            res = fitness_function(ensemble, None, None, None, None,
                                   base_fit, algorithm_list, base_eval,
                                   y_fit, y_eval, meta_model_type=meta_model_type,
                                   metric=metric, vus_win=vus_win)
        except Exception:
            continue
        if not np.isnan(res[2]):
            evals.append(float(res[2]))
    if not evals:
        return empty
    sigma = float(np.std(evals, ddof=1)) if len(evals) > 1 else 0.0
    return {'eps': 2.0 * sigma, 'sigma': sigma, 'repeats': len(evals)}


def compute_solo_fitness(
    detectors: List[str],
    algorithm_list: List[str],
    base_eval: np.ndarray,
    y_eval: np.ndarray,
    metric: str = DEFAULT_DECISION_METRICS,
    vus_win: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    What each detector scores on its own, with no meta-learner.

    The detector's own scores are thresholded directly, which is the baseline a
    reader actually has: run this detector and nothing else. The F1 and PR-AUC
    definitions are the ones the ensemble's fitness uses, so the two are
    comparable; only the threshold grid differs, since raw detector scores are
    not probabilities and do not share the meta-learner's [0, 1] range.

    Returns {detector: {'f1', 'fitness'}}, skipping any that fail to evaluate.
    """
    out: Dict[str, Dict[str, float]] = {}
    y = np.asarray(y_eval).flatten()
    needed = metrics_required(metric)
    for d in detectors:
        if d not in algorithm_list:
            continue
        try:
            scores = np.asarray(base_eval, dtype=float)[:, algorithm_list.index(d)]
            lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            best_f1 = 0.0
            for t in np.linspace(lo, hi, 50):
                f1 = f1_score((scores >= t).astype(int), y)[0]
                if f1 > best_f1:
                    best_f1 = float(f1)
            pr_auc = prauc(y, scores)
            vus = (vus_score(scores, y, vus_win)
                   if 'vus' in needed and vus_win is not None else float('nan'))
            fit = combine_metrics(metric, {'f1': best_f1, 'pr_auc': pr_auc,
                                           'vus': vus})
            if np.isnan(fit):
                fit = best_f1
        except Exception:
            continue
        if np.isnan(fit):
            continue
        out[d] = {'f1': best_f1, 'fitness': float(fit)}
    return out


def compute_near_best(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    best_ensemble: List[str],
    algorithm_list: List[str],
    sigma: float,
    kappa: float = math.sqrt(2.0),
    min_ensembles: int = 3,
) -> Dict[str, Any]:
    """
    The ensembles the run could not rank apart from the one it reported, and
    what they say about each detector.

    An ensemble is near-best when its fitness is within `kappa * sigma` of the
    highest found. Each fitness was measured once, so the difference between
    two of them carries sqrt(2) times the fitting noise of one — hence the
    default kappa.

    A detector's share of the near-best ensembles is read against the baseline,
    the share it would reach by chance: those ensembles have some mean size, so
    a detector with no preference either way appears in mean_size / |pool| of
    them. Comparing against a flat half instead would read every large ensemble
    as evidence for every detector in it.

    `disagrees` is the only field the explanation acts on: the near-best
    ensembles point one way and the reported one goes the other.

    Undefined when the fitting noise is unknown or fewer than `min_ensembles`
    ensembles clear the cutoff; `detectors` is then empty and `defined` False.
    """
    nan = float('nan')
    n_pool = len(algorithm_list)
    out: Dict[str, Any] = {
        'defined': False, 'n_near_best': 0,
        'n_evaluated': len(evaluated_ensembles),
        'best_fitness': nan, 'cutoff': nan, 'sigma': float(sigma),
        'kappa': float(kappa), 'mean_size': nan, 'baseline': nan,
        'detectors': {},
    }
    if not evaluated_ensembles or not n_pool or np.isnan(sigma):
        return out

    scored = [(set(key), float(res[2])) for key, res in evaluated_ensembles.items()
              if not np.isnan(float(res[2]))]
    if not scored:
        return out

    best_fitness = max(f for _, f in scored)
    cutoff = best_fitness - float(kappa) * float(sigma)
    near = [members for members, f in scored if f >= cutoff]
    out['best_fitness'] = best_fitness
    out['cutoff'] = cutoff
    out['n_near_best'] = len(near)
    if len(near) < min_ensembles:
        return out

    mean_size = float(np.mean([len(m) for m in near]))
    baseline = mean_size / n_pool
    chosen = set(best_ensemble)
    detectors: Dict[str, Dict[str, Any]] = {}
    for d in algorithm_list:
        count = sum(1 for m in near if d in m)
        share = count / len(near)
        says_in = share > baseline
        detectors[d] = {'count': count, 'share': share, 'says_in': says_in,
                        'disagrees': says_in != (d in chosen)}
    out.update(defined=True, mean_size=mean_size, baseline=baseline,
               detectors=detectors)
    return out


def compute_mean_marginal_contribution(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    algorithm_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Axis 1b — mean marginal contribution across evaluated subsets.

    For each detector d:
        contribution[d] = E[fitness | d present] − E[fitness | d absent]
    over the distinct subsets the GA evaluated. Returns a dict per detector:
        {'contribution', 'se', 'e_present', 'e_absent', 'n_present', 'n_absent'}.
    Missing means are NaN with the corresponding count = 0.

    `se` is the standard error of that difference. It is what says whether two
    detectors' utilities are separable at all, which is why the utility axis
    carries a middle class and the stability axis does not.
    """
    out: Dict[str, Dict[str, float]] = {}
    for d in algorithm_list:
        e_p, n_p, v_p = _conditional_mean_fitness(evaluated_ensembles, present=(d,))
        e_a, n_a, v_a = _conditional_mean_fitness(evaluated_ensembles, absent=(d,))
        contrib = (e_p - e_a) if (n_p > 0 and n_a > 0) else float('nan')
        se = float('nan')
        if n_p > 1 and n_a > 1 and not (np.isnan(v_p) or np.isnan(v_a)):
            se = float(np.sqrt(v_p / n_p + v_a / n_a))
        out[d] = {
            'contribution': contrib,
            'se': se,
            'e_present': e_p,
            'e_absent': e_a,
            'n_present': n_p,
            'n_absent': n_a,
        }
    return out




def compute_survival_rates(
    generation_populations: List[List[List[str]]],
    algorithm_list: List[str],
    population_size: int,
) -> Dict[str, List[float]]:
    """
    Axis 2 — survival rate per detector per generation:
        P(d, g) = (count of individuals in generation g containing d) / population_size
    Returns {detector: [P(d, g) for g in generations]}.
    """
    out: Dict[str, List[float]] = {d: [] for d in algorithm_list}
    denom = float(population_size) if population_size > 0 else 1.0
    for pop in generation_populations:
        for d in algorithm_list:
            count = sum(1 for ind in pop if d in ind)
            out[d].append(count / denom)
    return out


# ── Functional archetypes (intersection of the axes) ────────────────────────

# Each detector is labelled by the (Utility, Stability) high/low pair as a
# 2-letter H/L code, e.g. "HL" = high utility, low stability. A detector with no
# utility data is "Unclassified".
ARCHETYPE_UNCLASSIFIED = "Unclassified"

ARCHETYPE_ORDER = [
    "HH", "HL", "LH", "LL",
    ARCHETYPE_UNCLASSIFIED,
]

# The banded scheme splits utility into three levels, so its codes are a
# superset of the two-level ones and need their own display order.
BANDED_ARCHETYPE_ORDER = [
    "HH", "HL", "MH", "ML", "LH", "LL",
    ARCHETYPE_UNCLASSIFIED,
]


def _assign_archetype(u_high: bool, s_high: bool, util_nan: bool) -> str:
    """
    Label the (utility, stability) high/low pair as a 2-letter H/L code (e.g.
    "HL"). A detector with no utility data is "Unclassified".
    """
    if util_nan:
        return ARCHETYPE_UNCLASSIFIED
    return ("H" if u_high else "L") + ("H" if s_high else "L")


def _assign_banded_archetype(u_level: str, s_high: bool, util_nan: bool) -> str:
    """As `_assign_archetype`, but utility carries a middle level: "ML" = middle
    utility, low stability."""
    if util_nan:
        return ARCHETYPE_UNCLASSIFIED
    return u_level + ("H" if s_high else "L")


def _zero_anchored_axis(ax, which: str, values: List[float]) -> None:
    """Start `which` axis at 0 and tick it in equal, round steps.

    Utility and stability live on unrelated scales (survival is a rate in
    [0, 1]; marginal contribution is a fitness difference that can be a
    thousandth of that), so a shared step would flatten one of them — each axis
    gets its own. What they share is the origin: these are magnitudes, and an
    axis that starts at 0.0008 makes a negligible spread look like a wide one.

    The step is the 1/2/5 decade multiple that puts roughly five ticks across
    the data, the same ladder matplotlib's own locator walks; picking it here
    rather than letting the locator choose is what pins the origin.

    Negative values are respected — a detector can genuinely hurt fitness — so
    the range is anchored at zero rather than forced non-negative.
    """
    finite = [float(v) for v in values if v == v and abs(float(v)) != float("inf")]
    if not finite:
        return
    lo, hi = min(finite + [0.0]), max(finite + [0.0])
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
    raw = span / 5.0
    decade = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = next((m * decade for m in (1, 2, 5) if raw <= m * decade), 10 * decade)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    # Guard against a float-accumulation overrun on the final tick.
    n = int(round((end - start) / step)) + 1
    ticks = [start + i * step for i in range(max(n, 2))]
    if which == "x":
        ax.set_xlim(start, ticks[-1])
        ax.set_xticks(ticks)
    else:
        ax.set_ylim(start, ticks[-1])
        ax.set_yticks(ticks)


def _finite_median(values: List[float]) -> float:
    """Median over the finite (non-NaN) values; NaN if none are finite."""
    finite = [v for v in values if not np.isnan(v)]
    return float(np.median(finite)) if finite else float('nan')


def _finite_mean_sd(values: List[float]) -> Tuple[float, float]:
    """(mean, sample sd) over the finite values. The sd is NaN with fewer than
    two of them, which leaves the middle band undefined rather than zero-width."""
    finite = [v for v in values if not np.isnan(v)]
    if not finite:
        return float('nan'), float('nan')
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float('nan')
    return float(np.mean(finite)), sd


def classify_detector_archetypes(
    mean_marginal: Dict[str, Dict[str, float]],
    survival: Dict[str, List[float]],
    algorithm_list: List[str],
    abs_utility: float = 0.0,
    abs_stability: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Classify each detector into a functional archetype from the intersection of
    the two active axes (Utility × Stability). Reports three schemes side by
    side: relative (median split), absolute (fixed cutoff) and banded.

    Banded is the one the explanation reads. Utility carries a middle level
    because it is the noisier axis: its standard error is about 38% of the
    spread it has to resolve, against 12% for stability, so there is a range of
    utilities the pool does not separate. Stability has no such range and stays
    two-level — forcing a band onto it leaves most of the pool classified on
    neither axis.

    Axis scalars (per detector):
      utility        = mean_marginal[d]['contribution']
      stability_mean = mean(survival[d])  (the Stability axis)

    A detector is "stable-high" when its mean survival is above the threshold.
    (stability_trend = P(last) − P(first) is still reported for context, but does
    not affect the classification.)

    Returns {detector: {utility, stability_mean, stability_trend,
                        "relative": {u_high,s_high,archetype},
                        "absolute": {u_high,s_high,archetype},
                        "banded":   {u_level,s_high,archetype}}}.
    """
    util = {d: mean_marginal.get(d, {}).get('contribution', float('nan')) for d in algorithm_list}
    stab_mean, stab_trend = {}, {}
    for d in algorithm_list:
        ys = survival.get(d, [])
        stab_mean[d] = float(np.mean(ys)) if ys else float('nan')
        stab_trend[d] = (ys[-1] - ys[0]) if ys else float('nan')

    med_u = _finite_median(list(util.values()))
    med_s = _finite_median(list(stab_mean.values()))
    mean_u, sd_u = _finite_mean_sd(list(util.values()))
    mean_s, _ = _finite_mean_sd(list(stab_mean.values()))

    out: Dict[str, Dict[str, Any]] = {}
    for d in algorithm_list:
        u, sm, st = util[d], stab_mean[d], stab_trend[d]
        util_nan = bool(np.isnan(u))

        schemes: Dict[str, Dict[str, Any]] = {}
        for scheme, (tu, ts) in (
            ("relative", (med_u, med_s)),
            ("absolute", (abs_utility, abs_stability)),
        ):
            u_high = (not np.isnan(u)) and (not np.isnan(tu)) and (u > tu)
            # Stability depends only on the mean survival rate (no trend gate).
            s_high = (not np.isnan(sm)) and (not np.isnan(ts)) and (sm > ts)
            schemes[scheme] = {
                "u_high": u_high, "s_high": s_high,
                "archetype": _assign_archetype(u_high, s_high, util_nan),
            }

        # A detector exactly on a band edge stays in the middle: the edges are
        # estimates, so the level that claims less is the right one to give.
        if util_nan or np.isnan(mean_u) or np.isnan(sd_u):
            u_level = "M"
        elif u > mean_u + sd_u:
            u_level = "H"
        elif u < mean_u - sd_u:
            u_level = "L"
        else:
            u_level = "M"
        s_high_b = (not np.isnan(sm)) and (not np.isnan(mean_s)) and (sm > mean_s)

        out[d] = {
            "utility": u,
            "stability_mean": sm, "stability_trend": st,
            "relative": schemes["relative"], "absolute": schemes["absolute"],
            "banded": {
                "u_level": u_level, "s_high": s_high_b,
                "archetype": _assign_banded_archetype(u_level, s_high_b,
                                                      util_nan),
            },
        }
    return out


# ── Plot helpers ────────────────────────────────────────────────────────────

def _ga_plot_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_ga_utility(
    mean_marginal: Dict[str, Dict[str, float]],
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    The utility axis with its uncertainty and its class boundaries.

    Detectors are sorted by mean marginal contribution, each bar carries its
    standard error, and the middle band (mean ± sd across detectors) is shaded.
    Read together the three say why the axis has a middle class: the error bars
    are large against the spread the band has to cut, so a detector inside it is
    not separable from the pool rather than merely average.

    Saves to results/GA_Ens/{dataset}/{entity}/ga_selection_utility_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    raw = {d: mean_marginal.get(d, {}).get('contribution', float('nan'))
           for d in algorithm_list}
    ordered = sorted(algorithm_list,
                     key=lambda d: (np.isnan(raw[d]),
                                    -(0.0 if np.isnan(raw[d]) else raw[d])))
    vals = [raw[d] for d in ordered]
    errs = [mean_marginal.get(d, {}).get('se', float('nan')) for d in ordered]
    chosen = set(best_ensemble)
    mean_u, sd_u = _finite_mean_sd(list(raw.values()))

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(algorithm_list) + 4), 5))
    x = np.arange(len(ordered))
    colours = ["#cccccc" if np.isnan(v) else
               ("#2ca02c" if v >= 0 else "#d62728") for v in vals]
    ax.bar(x, [0.0 if np.isnan(v) else v for v in vals], color=colours,
           yerr=[0.0 if np.isnan(e) else e for e in errs],
           error_kw=dict(ecolor="#333333", elinewidth=0.9, capsize=2.5))
    if not (np.isnan(mean_u) or np.isnan(sd_u)):
        ax.axhspan(mean_u - sd_u, mean_u + sd_u, color="#4c72b0", alpha=0.10,
                   zorder=0)
        for edge in (mean_u - sd_u, mean_u + sd_u):
            ax.axhline(edge, color="#4c72b0", linestyle="--", linewidth=0.9,
                       alpha=0.8)
        ax.text(0.995, mean_u, "middle band  (mean ± sd)", transform=
                ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                color="#4c72b0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}*" if d in chosen else d for d in ordered],
                       rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("E[fit | present] − E[fit | absent]")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_utility_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()




def _plot_ga_survival_impl(
    survival_rates: Dict[str, List[float]],
    bold_set: set,
    title: Optional[str],
    save_path: str,
) -> None:
    """Shared rendering for survival-rate plots. `bold_set` controls which
    detectors receive the bold/opaque/marker treatment."""
    _ga_plot_rcparams()
    detectors = list(survival_rates.keys())
    if not detectors:
        return
    G = len(next(iter(survival_rates.values())))
    colour_map = {d: plt.cm.tab20(i / max(len(detectors), 1)) for i, d in enumerate(detectors)}

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * G + 4), 5))
    x = np.arange(G)
    for d in detectors:
        ys = survival_rates[d]
        bold = d in bold_set
        ax.plot(x, ys, label=d, color=colour_map[d],
                linewidth=2.0 if bold else 1.0,
                alpha=1.0 if bold else 0.35,
                marker="o" if bold else None,
                markersize=3.5 if bold else 0)

    ax.set_xlabel("Generation")
    ax.set_ylabel("P(d, g)  —  survival rate")
    ax.set_ylim(-0.02, 1.05)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_survival(
    survival_rates: Dict[str, List[float]],
    best_ensemble: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Produce two survival-rate line plots for Axis 2.

    (a) Ensemble-highlighted version — detectors in best_ensemble are bold/opaque;
        all others are faded. Emphasises which detectors the GA converged on.
        → ga_selection_survival_{dataset}_{entity}.png

    (b) All-bold version — every detector is drawn equally bold and opaque,
        so their trajectories can be compared without any pre-selection bias.
        → ga_selection_survival_all_{dataset}_{entity}.png
    """
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    in_best = set(best_ensemble or [])

    _plot_ga_survival_impl(
        survival_rates,
        bold_set=in_best,
        title=None,
        save_path=f"{directory}/ga_selection_survival_{dataset}_{entity}.png",
    )
    _plot_ga_survival_impl(
        survival_rates,
        bold_set=set(survival_rates.keys()),   # every detector bold
        title=None,
        save_path=f"{directory}/ga_selection_survival_all_{dataset}_{entity}.png",
    )


def _spread_annotations(fig, annotations, max_passes: int = 80) -> None:
    """Nudge overlapping point labels apart, in place.

    The archetype scatter names every point, and points cluster: two detectors
    with near-identical utility and stability sit on top of each other, and so
    did their names. Dropping labels is not an option — placing every detector
    is the figure's whole job — so they are moved instead.

    Pairwise repulsion in DISPLAY coordinates, which is the only space where
    "these two labels overlap" is a question at all: the axes carry different
    units (a fitness difference against a survival rate), so data coordinates
    cannot say whether two boxes collide.

    Each annotation was placed with `textcoords="offset points"`, so the nudge
    is applied to that offset — the label moves and the point does not. The
    vertical push is the larger one because a detector name is far wider than
    it is tall, so vertical separation buys more clearance per point moved.

    Deliberately not `adjustText`: a dependency for one figure, where a few
    passes of this converge. Stops as soon as a pass finds no collision, so the
    common case of a well-spread pool costs one draw.
    """
    if len(annotations) < 2:
        return
    fig.canvas.draw()                 # extents are undefined before a draw
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer=renderer) for a in annotations]
    shift = [np.zeros(2) for _ in annotations]

    for _ in range(max_passes):
        collided = False
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                a = boxes[i].translated(*shift[i])
                b = boxes[j].translated(*shift[j])
                if not a.overlaps(b):
                    continue
                collided = True
                dx = ((a.x0 + a.x1) - (b.x0 + b.x1)) / 2.0
                dy = ((a.y0 + a.y1) - (b.y0 + b.y1)) / 2.0
                norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                # A dead-on vertical tie gets an arbitrary but deterministic
                # nudge, or the pair would push along a zero-length vector.
                step = np.array([dx / norm * 1.2,
                                 (dy / norm * 2.4) if abs(dy) > 1e-9 else 2.4])
                shift[i] += step
                shift[j] -= step
        if not collided:
            break

    px_to_points = 72.0 / fig.dpi
    for ann, (dx, dy) in zip(annotations, shift):
        if not dx and not dy:
            continue
        ox, oy = ann.xyann
        ann.xyann = (ox + dx * px_to_points, oy + dy * px_to_points)
        # Far enough that the label no longer reads as belonging to its point:
        # reveal the leader line. The patch is created with every annotation and
        # left transparent, because matplotlib positions `arrow_patch` itself
        # and only does so for an annotation that was BUILT with `arrowprops` —
        # attaching one afterwards leaves it unplaced.
        if abs(dx) + abs(dy) > 14 and ann.arrow_patch is not None:
            ann.arrow_patch.set_alpha(0.6)


ARCHETYPE_COLOURS = {
    "HH": "#2ca02c", "MH": "#1f77b4", "LH": "#17becf",
    "HL": "#ff7f0e", "ML": "#9467bd", "LL": "#d62728",
    ARCHETYPE_UNCLASSIFIED: "#888888",
}


def _archetype_axes(ax, util, stab, algorithm_list, mean_u, sd_u, mean_s,
                    band: bool) -> None:
    """Shared frame for the two utility x stability scatters.

    The axes CROSS AT (0, 0) rather than at the corner of a box. Utility is a
    fitness difference, so a detector that lowers mean fitness is genuinely
    negative, and the crossing point moves right to give it somewhere to sit.
    """
    edges = []
    if not (np.isnan(mean_u) or np.isnan(sd_u)):
        edges = [mean_u - sd_u, mean_u + sd_u]
        if band:
            ax.axvspan(edges[0], edges[1], color="#4c72b0", alpha=0.10,
                       zorder=0)
        # Grey without the shading, so the cuts read the same as the stability
        # line above and do not compete with the blue an archetype may carry.
        for edge in edges:
            ax.axvline(edge, color="#4c72b0" if band else "grey",
                       linestyle="--", linewidth=0.9,
                       alpha=0.8 if band else 0.7)
    if not np.isnan(mean_s):
        ax.axhline(mean_s, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.7)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.1)
        ax.spines[spine].set_color("black")
    # With the vertical spine inside the plot, the two "0" labels land on top of
    # each other at the crossing. The x axis keeps its own, since that is the one
    # a reader is checking a sign against.
    if min(util.values(), default=0.0) < 0:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: "" if abs(v) < 1e-12 else f"{v:g}"))
        ax.yaxis.set_label_coords(-0.02, 0.5)
    ax.set_xlabel("Utility  (mean marginal contribution)")
    ax.set_ylabel("Stability  (mean survival rate)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _zero_anchored_axis(ax, "x", [util[d] for d in algorithm_list] + edges)
    _zero_anchored_axis(ax, "y", [stab[d] for d in algorithm_list] + [mean_s])


def _scatter_common(archetypes, algorithm_list):
    util = {d: archetypes[d]["utility"] for d in algorithm_list}
    stab = {d: archetypes[d]["stability_mean"] for d in algorithm_list}
    mean_u, sd_u = _finite_mean_sd(list(util.values()))
    mean_s, _ = _finite_mean_sd(list(stab.values()))
    unclassified = sorted(
        {d for d in algorithm_list
         if archetypes[d].get("banded", {}).get("archetype")
         == ARCHETYPE_UNCLASSIFIED})
    return util, stab, mean_u, sd_u, mean_s, unclassified


def _finish_scatter(fig, ax, annotations, algorithm_list, unclassified,
                    handles, labels, dataset, entity, stem) -> None:
    # After the limits are final: a nudge computed against one set of axis
    # limits is wrong once `_zero_anchored_axis` moves them.
    _spread_annotations(fig, annotations)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    if unclassified:
        fig.text(0.5, -0.02, "Unclassified (no marginal-contribution data): "
                 + ", ".join(unclassified), ha="center", fontsize=9, alpha=0.8)
    draw_abbreviation_key(fig, algorithm_list,
                          y=-0.06 if unclassified else -0.02)
    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/{stem}_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_archetypes(
    archetypes: Dict[str, Dict[str, Any]],
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Utility x stability, coloured by archetype.

      x = utility (mean marginal contribution), y = stability (mean survival)
      colour       = the detector's archetype
      filled point = in the chosen ensemble, hollow = not
      dashed lines = the two utility cuts and the stability cut

    The colour already carries the archetype, so the middle band is marked by
    its two edges rather than shaded. `plot_ga_bands` is the same scatter with
    the band filled in and without the per-archetype colour.

    Saves to ga_selection_archetypes_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    util, stab, mean_u, sd_u, mean_s, unclassified = _scatter_common(
        archetypes, algorithm_list)
    chosen = set(best_ensemble)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    annotations = []
    present = []
    for d in algorithm_list:
        u, s = util[d], stab[d]
        if np.isnan(u) or np.isnan(s):
            continue
        code = archetypes[d].get("banded", {}).get("archetype",
                                                   ARCHETYPE_UNCLASSIFIED)
        colour = ARCHETYPE_COLOURS.get(code, "#888888")
        if code not in present:
            present.append(code)
        inside = d in chosen
        ax.scatter([u], [s], s=90, zorder=3,
                   color=colour if inside else "none",
                   edgecolors=colour, linewidths=1.5)
        annotations.append(
            ax.annotate(abbreviate_detector(d), (u, s),
                        textcoords="offset points", xytext=(5, 4), fontsize=8,
                        alpha=0.85,
                        arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                        color="grey", alpha=0.0,
                                        shrinkA=0, shrinkB=2)))
    _archetype_axes(ax, util, stab, algorithm_list, mean_u, sd_u, mean_s,
                    band=False)

    handles, labels = [], []
    for code in BANDED_ARCHETYPE_ORDER:
        if code not in present:
            continue
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markersize=9,
                              color=ARCHETYPE_COLOURS.get(code, "#888888")))
        labels.append(code)
    for filled, text in ((True, "in the chosen ensemble"),
                         (False, "not in the chosen ensemble")):
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markersize=9, color="#555555" if filled else "none",
                              markeredgecolor="#555555", markeredgewidth=1.5))
        labels.append(text)
    _finish_scatter(fig, ax, annotations, algorithm_list, unclassified,
                    handles, labels, dataset, entity, "ga_selection_archetypes")


def plot_ga_bands(
    archetypes: Dict[str, Dict[str, Any]],
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    The same scatter with the cuts drawn instead of the archetype colours.

    The shaded band is the middle utility class and the dashed line is the
    stability cut, so this is the figure for checking where a detector falls
    relative to the thresholds rather than which class it landed in.

    Saves to ga_selection_bands_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    util, stab, mean_u, sd_u, mean_s, unclassified = _scatter_common(
        archetypes, algorithm_list)
    chosen = set(best_ensemble)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    annotations = []
    for d in algorithm_list:
        u, s = util[d], stab[d]
        if np.isnan(u) or np.isnan(s):
            continue
        inside = d in chosen
        ax.scatter([u], [s], s=90, zorder=3,
                   color="#2ca02c" if inside else "none",
                   edgecolors="#2ca02c" if inside else "#d62728",
                   linewidths=1.5)
        annotations.append(
            ax.annotate(abbreviate_detector(d), (u, s),
                        textcoords="offset points", xytext=(5, 4), fontsize=8,
                        alpha=0.85,
                        arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                        color="grey", alpha=0.0,
                                        shrinkA=0, shrinkB=2)))
    _archetype_axes(ax, util, stab, algorithm_list, mean_u, sd_u, mean_s,
                    band=True)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=9,
                      color="#2ca02c"),
               Line2D([], [], marker="o", linestyle="none", markersize=9,
                      color="none", markeredgecolor="#d62728",
                      markeredgewidth=1.5)]
    labels = ["in the chosen ensemble", "not in the chosen ensemble"]
    _finish_scatter(fig, ax, annotations, algorithm_list, unclassified,
                    handles, labels, dataset, entity, "ga_selection_bands")


def plot_ga_profile(
    archetypes: Dict[str, Dict[str, Any]],
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    How many detectors landed in each archetype, and how many of those the
    algorithm kept.

    This is the figure that says whether the two axes predict the selection at
    all: the bars should fall from left to right, and the minority segment on
    the HH and LL bars is exactly the set of contradictions the explanation has
    to account for.

    Saves to ga_selection_profile_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    chosen = set(best_ensemble)
    codes, kept, left = [], [], []
    for code in BANDED_ARCHETYPE_ORDER:
        members = [d for d in algorithm_list
                   if archetypes.get(d, {}).get("banded", {}).get("archetype")
                   == code]
        if not members:
            continue
        codes.append(code)
        kept.append(sum(1 for d in members if d in chosen))
        left.append(sum(1 for d in members if d not in chosen))
    if not codes:
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(codes) + 2), 4.5))
    x = np.arange(len(codes))
    ax.bar(x, kept, color="#2ca02c", label="in the chosen ensemble")
    ax.bar(x, left, bottom=kept, color="#e6e6e6", edgecolor="#d62728",
           linewidth=0.8, label="not in the chosen ensemble")
    tallest = max(k + l for k, l in zip(kept, left))
    for i, (k, l) in enumerate(zip(kept, left)):
        total = k + l
        ax.text(i, total + tallest * 0.03, f"{100.0 * k / total:.0f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, tallest * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(codes)
    ax.set_xlabel("Archetype  (utility level, stability level)")
    ax.set_ylabel("Detectors")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    # Below the axes: the tallest bar can stand anywhere, so no corner is safe.
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.18))

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_profile_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_plateau(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    near_best: Dict[str, Any],
    dataset: str,
    entity: str,
) -> None:
    """
    Every evaluated ensemble's fitness, best first, with the near-best cutoff.

    How flat the top of this curve is decides whether the reported ensemble was
    a clear winner or one of many the run could not rank apart, which is the
    whole basis of the fallback.

    Saves to ga_selection_plateau_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    vals = sorted((float(r[2]) for r in evaluated_ensembles.values()
                   if not np.isnan(float(r[2]))), reverse=True)
    if not vals:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(vals) + 1)
    ax.plot(x, vals, color="#4c72b0", linewidth=1.4)
    cutoff = near_best.get("cutoff", float("nan"))
    n_near = near_best.get("n_near_best", 0)
    if not np.isnan(cutoff):
        ax.axhline(cutoff, color="#d62728", linestyle="--", linewidth=1.0)
        ax.fill_between(x, cutoff, vals, where=[v >= cutoff for v in vals],
                        color="#d62728", alpha=0.12, interpolate=True)
        # The flatter the plateau the closer the cutoff sits to the best, so
        # the room above it is made rather than assumed: the label always goes
        # over the line, and the axes are extended when that would not fit.
        lo, hi = ax.get_ylim()
        needed = lo + (cutoff - lo) / 0.88
        if needed > hi:
            ax.set_ylim(lo, needed)
        ax.annotate(f"{n_near} near-best of {len(vals)}",
                    xy=(max(n_near, 1), cutoff), xytext=(14, 6),
                    textcoords="offset points", fontsize=9, color="#d62728",
                    va="bottom")
    ax.set_xlabel("Evaluated ensembles, best first")
    ax.set_ylabel("Fitness")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_plateau_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_convergence(
    generation_fitness: List[Dict[str, float]],
    near_best: Dict[str, Any],
    dataset: str,
    entity: str,
) -> None:
    """
    Best-so-far and population-mean fitness per generation.

    The plateau figure ranks the subsets the run evaluated; this one says when
    they were found, and so whether the budget or the pool limited the answer.

    Saves to ga_selection_convergence_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    rows = [g for g in (generation_fitness or [])
            if not np.isnan(float(g.get('best', float('nan'))))]
    if len(rows) < 2:
        return

    gens = [int(g['generation']) for g in rows]
    best = [float(g['best']) for g in rows]
    mean = [float(g['mean']) for g in rows]
    running = list(np.maximum.accumulate(best))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, running, color="#2ca02c", linewidth=1.8, label="best so far")
    ax.plot(gens, best, color="#4c72b0", linewidth=1.0, linestyle="--",
            marker="o", markersize=3, label="best in generation")
    ax.plot(gens, mean, color="#9467bd", linewidth=1.2, label="population mean")

    sigma = near_best.get("sigma", float("nan"))
    if not np.isnan(sigma) and sigma > 0:
        # The band the run cannot see inside: this close to the final answer,
        # the refitting noise alone could produce the difference.
        top = running[-1]
        ax.axhspan(top - math.sqrt(2.0) * sigma, top, color="#d62728",
                   alpha=0.10, zorder=0)
        ax.axhline(top - math.sqrt(2.0) * sigma, color="#d62728",
                   linestyle="--", linewidth=0.9, alpha=0.8)

    # The FIRST generation to reach the final best is the last one that changed
    # the outcome; everything after it is budget spent for nothing.
    settled = min(i for i, v in enumerate(running) if v == running[-1])
    if settled < len(gens) - 1:
        ax.axvline(gens[settled], color="grey", linestyle=":", linewidth=1.0)
        ax.annotate(f"final best reached at generation {gens[settled]}",
                    xy=(gens[settled], running[-1]), xytext=(6, -14),
                    textcoords="offset points", fontsize=9, color="grey")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_convergence_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_solo(
    solo: Dict[str, Dict[str, float]],
    best_ensemble: List[str],
    best_fitness: float,
    near_best: Dict[str, Any],
    dataset: str,
    entity: str,
    top_n: int = 12,
) -> None:
    """
    Each detector's standalone fitness against the chosen ensemble's.

    Every other figure in this stage compares detectors with each other; this
    one compares the ensemble with the baseline of running one detector.

    Saves to ga_selection_solo_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    rows = sorted(((d, v['fitness']) for d, v in (solo or {}).items()),
                  key=lambda kv: -kv[1])[:max(1, top_n)]
    if not rows or np.isnan(best_fitness):
        return

    chosen = set(best_ensemble)
    names = [d for d, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    colours = ["#2ca02c" if d in chosen else "#bdbdbd" for d in names]

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.34 * len(names) + 1.8)))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colours)
    ax.axvline(best_fitness, color="#d62728", linestyle="--", linewidth=1.2)

    sigma = near_best.get("sigma", float("nan"))
    if not np.isnan(sigma) and sigma > 0:
        ax.axvspan(best_fitness - math.sqrt(2.0) * sigma, best_fitness,
                   color="#d62728", alpha=0.10, zorder=0)

    gain = best_fitness - vals[-1]
    ax.annotate(f"ensemble {best_fitness:.4f}  ({gain:+.4f} vs best single)",
                xy=(best_fitness, len(names) - 0.4), xytext=(-6, 0),
                textcoords="offset points", fontsize=9, color="#d62728",
                ha="right", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Fitness of the detector on its own")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    handles = [Patch(facecolor="#2ca02c", label="in the chosen ensemble"),
               Patch(facecolor="#bdbdbd", label="not in it")]
    # Below the axes: the bars run to the right edge, so no corner inside the
    # plot is reliably free of them.
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.12 - 1.2 / max(4, len(names))))

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_solo_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_composition(
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Which detector groups the chosen ensemble drew from, against the pool.

    A group the pool offered and the ensemble passed over entirely is what the
    archetype figures cannot show, being keyed on detectors rather than kinds.

    Saves to ga_selection_composition_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    chosen = set(best_ensemble)
    groups: Dict[str, List[int]] = {}
    for d in algorithm_list:
        g = group_of(family_of(d))
        if g is None:
            continue
        counts = groups.setdefault(g, [0, 0])
        counts[0] += 1
        if d in chosen:
            counts[1] += 1
    if not groups:
        return

    order = [g for g in DETECTOR_GROUPS if g in groups]
    pool = [groups[g][0] for g in order]
    kept = [groups[g][1] for g in order]
    labels = [GROUP_LABELS.get(g, g) for g in order]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(order) + 2), 4.5))
    x = np.arange(len(order))
    ax.bar(x, pool, color="#e6e6e6", edgecolor="#9e9e9e", linewidth=0.8,
           label="in the pool")
    ax.bar(x, kept, color="#4c72b0", label="in the chosen ensemble")
    for i, (p, k) in enumerate(zip(pool, kept)):
        ax.text(i, p + max(pool) * 0.03, f"{k}/{p}", ha="center", fontsize=9)
    ax.set_ylim(0, max(pool) * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Detectors")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_composition_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


# ── Orchestrator + report ───────────────────────────────────────────────────

def explain_ga_selection(
    best_ensemble: List[str],
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    generation_populations: List[List[List[str]]],
    algorithm_list: List[str],
    population_size: int,
    dataset: str,
    entity: str,
    explain: bool = False,
    base_fit: Optional[np.ndarray] = None,
    y_fit: Optional[np.ndarray] = None,
    base_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    meta_model_type: str = 'rf',
    metric: str = DEFAULT_DECISION_METRICS,
    vus_win: Optional[int] = None,
    generation_fitness: Optional[List[Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    GA-ensemble selection explainability: explain *why* each detector ended up
    in best_ensemble, along two analytical axes (utility, stability). Produces
    a structured text report and the stage's figures under
        results/GA_Ens/{dataset}/{entity}/

    Returns a dict with the computed structures when explain=True; None otherwise.
    """
    if not explain:
        return None
    if not best_ensemble:
        return None

    mean_marginal = compute_mean_marginal_contribution(evaluated_ensembles, algorithm_list)
    survival = compute_survival_rates(generation_populations, algorithm_list, population_size)
    archetypes = classify_detector_archetypes(
        mean_marginal, survival, algorithm_list)

    noise: Dict[str, float] = {}
    if base_fit is not None and base_eval is not None:
        noise = measure_refit_noise(
            best_ensemble, algorithm_list, base_fit, y_fit, base_eval, y_eval,
            meta_model_type=meta_model_type,
            metric=metric, vus_win=vus_win)
    sigma = noise.get('sigma', float('nan'))
    # Without the fitting noise there is no cutoff, so a contradiction between
    # the archetype and the chosen ensemble has nothing to answer it. That is a
    # silent loss otherwise, since every other part of the stage still works.
    if np.isnan(sigma):
        logger.warning("  ⚠ Fitting noise could not be measured; the near-best "
                       "ensembles are undefined and any contradiction between "
                       "an archetype and the chosen ensemble will go "
                       "unexplained.")

    near_best = compute_near_best(evaluated_ensembles, best_ensemble,
                                  algorithm_list, sigma)

    solo: Dict[str, Dict[str, float]] = {}
    if base_eval is not None and y_eval is not None:
        solo = compute_solo_fitness(
            algorithm_list, algorithm_list, base_eval, y_eval,
            metric=metric, vus_win=vus_win)

    best_key = tuple(sorted(best_ensemble))
    best_fitness = float(evaluated_ensembles.get(best_key, (0, 0, float('nan')))[2])

    plot_ga_utility(mean_marginal, best_ensemble, algorithm_list, dataset, entity)
    plot_ga_survival(survival, best_ensemble, dataset, entity)
    plot_ga_archetypes(archetypes, best_ensemble, algorithm_list, dataset, entity)
    plot_ga_bands(archetypes, best_ensemble, algorithm_list, dataset, entity)
    plot_ga_profile(archetypes, best_ensemble, algorithm_list, dataset, entity)
    plot_ga_plateau(evaluated_ensembles, near_best, dataset, entity)
    plot_ga_convergence(generation_fitness or [], near_best, dataset, entity)
    plot_ga_solo(solo, best_ensemble, best_fitness, near_best, dataset, entity)
    plot_ga_composition(best_ensemble, algorithm_list, dataset, entity)

    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory, f"ga_selection_explainability_{dataset}_{entity}.txt")

    n_subsets = len(evaluated_ensembles)
    n_generations = len(generation_populations)

    with open(report_path, "w") as f:
        f.write("=== GA Ensemble Selection Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Best ensemble  : {list(best_ensemble)}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Generations    : {n_generations}\n")
        f.write(f"Distinct subsets evaluated: {n_subsets}\n\n")

        # ── Axis 1 ────────────────────────────────────────────────────────
        f.write("--- Axis 1: Utility ---\n")
        f.write("Mean marginal contribution across all evaluated subsets:\n")
        f.write(f"      {'detector':<14} {'E[fit|p]-E[fit|a]':>18} {'se':>10} "
                f"{'E[fit|p]':>10} {'E[fit|a]':>10} {'#p':>5} {'#a':>5}\n")
        f.write("      " + "-" * 79 + "\n")
        for d in algorithm_list:
            mm = mean_marginal[d]
            c = f"{mm['contribution']:+.4f}" if not np.isnan(mm['contribution']) else "N/A"
            se = f"{mm['se']:.4f}" if not np.isnan(mm.get('se', float('nan'))) else "N/A"
            ep = f"{mm['e_present']:.4f}" if not np.isnan(mm['e_present']) else "N/A"
            ea = f"{mm['e_absent']:.4f}" if not np.isnan(mm['e_absent']) else "N/A"
            f.write(f"      {d:<14} {c:>18} {se:>10} {ep:>10} {ea:>10} "
                    f"{mm['n_present']:>5d} {mm['n_absent']:>5d}\n")

        # ── Axis 2 ────────────────────────────────────────────────────────
        f.write("\n--- Axis 2: Stability (Evolutionary Survival) ---\n")
        f.write("P(d, g) = (#individuals in generation g containing d) / population_size\n")
        f.write(f"      {'detector':<14} {'mean P':>8} {'first':>8} "
                f"{'last':>8} {'trend':>10}\n")
        f.write("      " + "-" * 52 + "\n")
        for d in algorithm_list:
            ys = survival[d]
            if not ys:
                f.write(f"      {d:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}\n")
                continue
            f.write(f"      {d:<14} {np.mean(ys):>8.3f} {ys[0]:>8.3f} "
                    f"{ys[-1]:>8.3f} {(ys[-1] - ys[0]):>+10.3f}\n")

        # ── Functional archetypes ──────────────────────────────────────────
        f.write("\n--- Functional Archetypes (axis intersections) ---\n")
        f.write("Utility = mean marginal contribution, cut at mean +- sd across "
                "detectors (H / M / L).\n")
        f.write("Stability = mean survival rate, cut at the mean across "
                "detectors (H / L).\n")
        f.write("Trend P_last - P_first is shown for context but does not "
                "affect classification.\n")
        f.write("Archetype = the (U,S) pair as a 2-letter code, e.g. "
                "ML = middle utility, low stability.\n\n")
        f.write(f"      {'detector':<14} {'util':>9} {'stab':>7} "
                f"{'trend':>8}  {'archetype':<12} {'in ensemble'}\n")
        f.write("      " + "-" * 72 + "\n")

        def _num(v, fmt="{:+.4f}"):
            return fmt.format(v) if not np.isnan(v) else "N/A"

        chosen = set(best_ensemble)
        for d in algorithm_list:
            a = archetypes[d]
            f.write(
                f"      {d:<14} {_num(a['utility']):>9} "
                f"{_num(a['stability_mean'], '{:.3f}'):>7} "
                f"{_num(a['stability_trend'], '{:+.3f}'):>8}  "
                f"{a['banded']['archetype']:<12} "
                f"{'yes' if d in chosen else 'no'}\n"
            )

        tally: Dict[str, int] = {}
        for d in algorithm_list:
            name = archetypes[d]["banded"]["archetype"]
            tally[name] = tally.get(name, 0) + 1
        ordered = [(nm, tally[nm]) for nm in BANDED_ARCHETYPE_ORDER if nm in tally]
        f.write("\n  Tally: "
                + ", ".join(f"{nm}: {ct}" for nm, ct in ordered) + "\n")

        # ── Near-best ensembles ────────────────────────────────────────────
        f.write("\n--- Near-best ensembles ---\n")
        sigma = noise.get('sigma', float('nan'))
        f.write(f"Fitting noise (sd over {noise.get('repeats', 0)} refits of the "
                f"chosen ensemble): {_num(sigma, '{:.4f}')}\n")
        if near_best.get('defined'):
            f.write(f"Cutoff = best - sqrt(2) * fitting noise = "
                    f"{near_best['cutoff']:.4f}  "
                    f"(best {near_best['best_fitness']:.4f})\n")
            f.write(f"{near_best['n_near_best']} of {near_best['n_evaluated']} "
                    f"evaluated ensembles are near-best; mean size "
                    f"{near_best['mean_size']:.1f}, baseline "
                    f"{near_best['baseline']:.3f}\n\n")
            f.write(f"      {'detector':<14} {'in near-best':>13} {'share':>8} "
                    f"{'baseline':>10}  {'disagrees with the ensemble'}\n")
            f.write("      " + "-" * 76 + "\n")
            for d in algorithm_list:
                st = near_best['detectors'].get(d)
                if not st:
                    continue
                frac = f"{st['count']}/{near_best['n_near_best']}"
                f.write(f"      {d:<14} {frac:>13} "
                        f"{st['share']:>8.3f} {near_best['baseline']:>10.3f}  "
                        f"{'yes' if st['disagrees'] else 'no'}\n")
        else:
            f.write("Too few ensembles cleared the cutoff to compare against.\n")

        # The rule the IR applies, written down at run time rather than left
        # to be re-derived: the archetype expects HH in and LL out.
        f.write("\n--- Contradictions between archetype and decision ---\n")
        contradictions = []
        for d in algorithm_list:
            code = archetypes[d]["banded"]["archetype"]
            inside = d in chosen
            if code == "HH" and not inside:
                contradictions.append((d, code, "excluded"))
            elif code == "LL" and inside:
                contradictions.append((d, code, "included"))
        covered = [c for c in contradictions
                   if near_best.get("defined")
                   and near_best["detectors"].get(c[0], {}).get("disagrees")]
        n_classified = sum(1 for d in algorithm_list
                           if archetypes[d]["banded"]["archetype"] in ("HH", "LL"))
        f.write(f"Decisive detectors (HH or LL): {n_classified} of "
                f"{len(algorithm_list)}\n")
        f.write(f"Contradictions: {len(contradictions)}  "
                f"(covered by the near-best ensembles: {len(covered)}, "
                f"unexplained: {len(contradictions) - len(covered)})\n")
        if contradictions:
            f.write(f"\n      {'detector':<14} {'archetype':>10} "
                    f"{'decision':>10}  {'covered'}\n")
            f.write("      " + "-" * 48 + "\n")
            for d, code, decision in contradictions:
                is_cov = any(c[0] == d for c in covered)
                f.write(f"      {d:<14} {code:>10} {decision:>10}  "
                        f"{'yes' if is_cov else 'no'}\n")

        # ── Search convergence ─────────────────────────────────────────────
        if generation_fitness:
            f.write("\n--- Search convergence ---\n")
            f.write(f"      {'generation':>10} {'best':>10} {'mean':>10} "
                    f"{'worst':>10} {'best so far':>12}\n")
            f.write("      " + "-" * 56 + "\n")
            running = float('-inf')
            for g in generation_fitness:
                running = max(running, float(g['best']))
                f.write(f"      {int(g['generation']):>10d} "
                        f"{_num(g['best'], '{:.4f}'):>10} "
                        f"{_num(g['mean'], '{:.4f}'):>10} "
                        f"{_num(g['worst'], '{:.4f}'):>10} "
                        f"{running:>12.4f}\n")
            bests = [float(g['best']) for g in generation_fitness]
            peak = max(bests)
            settled = min(i for i, v in enumerate(
                np.maximum.accumulate(bests)) if v == peak)
            f.write(f"Best fitness last improved at generation "
                    f"{int(generation_fitness[settled]['generation'])} of "
                    f"{len(generation_fitness)}.\n")

        # ── Solo detectors ─────────────────────────────────────────────────
        if solo:
            f.write("\n--- Each detector on its own ---\n")
            f.write("The detector's own scores thresholded directly, with no "
                    "meta-learner: the baseline of running one detector and "
                    "nothing else.\n")
            ranked = sorted(solo.items(), key=lambda kv: -kv[1]['fitness'])
            f.write(f"\n      {'detector':<14} {'fitness':>10} {'F1':>10}"
                    f"  {'in ensemble'}\n")
            f.write("      " + "-" * 50 + "\n")
            for d, v in ranked:
                f.write(f"      {d:<14} {v['fitness']:>10.4f} {v['f1']:>10.4f}"
                        f"  {'yes' if d in chosen else 'no'}\n")
            top_d, top_v = ranked[0]
            if not np.isnan(best_fitness):
                f.write(f"\nBest alone: {top_d} at {top_v['fitness']:.4f}.  "
                        f"Chosen ensemble: {best_fitness:.4f}  "
                        f"({best_fitness - top_v['fitness']:+.4f}).\n")

            # ── Naive baseline explanation ─────────────────────────────────
            #
            # The explanation anyone would give without this stage: the
            # ensemble holds the detectors that score best alone. Scored the
            # same way the archetypes are, so the two error counts compare.
            f.write("\n--- Naive baseline explanation ---\n")
            f.write("Predicting that the ensemble holds the k best detectors "
                    "by standalone fitness, k being the ensemble size, and "
                    "counting where that disagrees with the chosen ensemble.\n")
            k = len(chosen)
            predicted = {d for d, _ in ranked[:k]}
            scored = [d for d in algorithm_list if d in solo]
            naive_err = sum(1 for d in scored
                            if (d in predicted) != (d in chosen))
            decisive = [d for d in scored
                        if archetypes[d]["banded"]["archetype"] in ("HH", "LL")]
            naive_dec = sum(1 for d in decisive
                            if (d in predicted) != (d in chosen))
            arch_dec = sum(1 for d in decisive
                           if (archetypes[d]["banded"]["archetype"] == "HH")
                           != (d in chosen))
            f.write(f"\n      over all {len(scored)} scored detectors      : "
                    f"naive baseline {naive_err} wrong\n")
            f.write(f"      over the {len(decisive)} decisive (HH or LL) : "
                    f"naive baseline {naive_dec} wrong, "
                    f"archetypes {arch_dec} wrong\n")
            f.write("The second line is the like-for-like comparison: the "
                    "archetypes commit only on the decisive detectors, so the "
                    "baseline is scored on those same detectors.\n")

        # ── Composition ────────────────────────────────────────────────────
        f.write("\n--- Ensemble composition by detector group ---\n")
        f.write(f"      {'group':<20} {'in pool':>8} {'chosen':>8} {'share':>8}\n")
        f.write("      " + "-" * 46 + "\n")
        for g in DETECTOR_GROUPS:
            members = [d for d in algorithm_list if group_of(family_of(d)) == g]
            if not members:
                continue
            took = sum(1 for d in members if d in chosen)
            f.write(f"      {GROUP_LABELS.get(g, g):<20} {len(members):>8d} "
                    f"{took:>8d} {took / len(members):>8.2f}\n")

    result = {
        "best_ensemble": list(best_ensemble),
        "mean_marginal": mean_marginal,
        "survival": survival,
        "archetypes": archetypes,
        "n_subsets_evaluated": n_subsets,
        "n_generations": n_generations,
        "noise": noise,
        "near_best": near_best,
        "solo": solo,
        "best_fitness": best_fitness,
        "generation_fitness": list(generation_fitness or []),
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        ir.write_stage_ir(ir.build_ga_selection_ir(dataset, entity, result),
                           dataset, entity, "ir_ga_selection")
    except Exception as e:
        logger.error(f"GA selection IR emission failed (non-fatal): {e}")

    return result


# ════════════════════════════════════════════════════════════════════════════
#  GA Ensemble Combination Explainability
#
#  Explains *how the meta-learner combines* the chosen detectors, by attributing
#  its output to the per-detector score columns via two methods, then merging
#  their rankings with a Markov-chain rank aggregation:
#    • SHAP — exact interventional Shapley (single median baseline), label-free.
#    • PFI  — permutation feature importance measured as fitness drop, label-based.
# ════════════════════════════════════════════════════════════════════════════


def _best_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Maximum F1 over the same threshold grid fitness_function uses."""
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()
    best = 0.0
    for t in np.linspace(0.1, 0.9, 50):
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_pred, y_true)[0]
        if f1 > best:
            best = f1
    return float(best)


def compute_meta_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    baseline_row: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """
    The per-row Shapley matrix, shape (n_rows, n_features).

    Exact interventional Shapley values of the meta-learner over its detector
    features, using a SINGLE baseline row. For instance x and subset S of
    features, F(S) marginalises the absent features to the baseline:
    z_j = x_j if j in S else baseline_row[j].
    phi_i(x) = Σ_{S ⊆ F\\{i}} w(|S|) (F(S∪i) − F(S)),  w(s)=s!(d−s−1)!/d!.

    Because d = ensemble size is small, all 2^d subset predictions are
    enumerated exactly (cheap).

    Returned per row rather than pre-aggregated because the callers want
    several different summaries of the same numbers — mean|phi| and mean phi
    at minimum — and the 2^d enumeration is by far the most expensive thing
    this stage does. Collapsing inside meant it ran twice for identical work.
    """
    d, n = int(n_features), X_explain.shape[0]
    if d == 0 or n == 0:
        return np.full((n, d), np.nan, dtype=float)
    if d == 1:
        # Single feature carries the entire deviation from baseline.
        Z0 = np.tile(np.asarray(baseline_row, dtype=float), (n, 1))
        phi = (np.asarray(predict_fn(X_explain.copy()), float)
               - np.asarray(predict_fn(Z0), float))
        return phi.reshape(n, 1)

    import math
    # Cache F(S) for every subset mask (bitmask over feature indices).
    pred_cache: Dict[int, np.ndarray] = {}
    for mask in range(1 << d):
        Z = np.tile(np.asarray(baseline_row, dtype=float), (n, 1))
        for j in range(d):
            if mask & (1 << j):
                Z[:, j] = X_explain[:, j]
        pred_cache[mask] = np.nan_to_num(np.asarray(predict_fn(Z), dtype=float),
                                         nan=0.0, posinf=1.0, neginf=0.0)

    fact = math.factorial
    weight = {s: fact(s) * fact(d - s - 1) / fact(d) for s in range(d)}
    out = np.zeros((n, d), dtype=float)
    for i in range(d):
        phi = np.zeros(n, dtype=float)
        others = [j for j in range(d) if j != i]
        # Enumerate every subset S of the other features.
        for sub in range(1 << (d - 1)):
            mask = 0
            s = 0
            for b, j in enumerate(others):
                if sub & (1 << b):
                    mask |= (1 << j)
                    s += 1
            phi += weight[s] * (pred_cache[mask | (1 << i)] - pred_cache[mask])
        out[:, i] = phi
    return out


def compute_meta_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    baseline_row: np.ndarray,
    feature_names: List[str],
    mode: str = "abs",
) -> Dict[str, float]:
    """
    Global SHAP importance per feature, aggregated over the explained rows:
      mode="abs"    → mean of |phi_i(x)|  (magnitude of influence)
      mode="signed" → mean of  phi_i(x)   (net direction of influence)

    A thin aggregation over compute_meta_shap_values. NOTE on mode="signed":
    Shapley efficiency makes the signed means decompose
    (mean prediction over the explained rows) − (prediction at the baseline
    row), so they measure how far the explained set sits from that one
    reference row, not which way a feature pushes, and the signed mean can
    carry the wrong sign outright. The sign comes from compute_meta_ale
    instead; this mode is retained only for the report's superseded
    comparison table.
    """
    d = len(feature_names)
    if d == 0 or X_explain.shape[0] == 0:
        return {f: float('nan') for f in feature_names}
    phi = compute_meta_shap_values(predict_fn, X_explain, baseline_row, d)
    agg = np.abs(phi).mean(axis=0) if mode == "abs" else phi.mean(axis=0)
    return {f: float(agg[i]) for i, f in enumerate(feature_names)}


def score_fn_for(metric=DEFAULT_DECISION_METRICS, vus_win=None):
    """The (y_true, y_scores) -> float scorer matching the GA's fitness.

    Kept beside `combine_metrics` so a weighted spec changes one function
    rather than every caller.
    """
    needed = metrics_required(metric)

    def score(y_true, y_scores):
        values = {}
        if 'f1' in needed:
            values['f1'] = _best_threshold_f1(y_true, y_scores)
        if 'pr_auc' in needed:
            values['pr_auc'] = float(prauc(np.asarray(y_true).flatten(),
                                           np.asarray(y_scores).flatten()))
        if 'vus' in needed:
            values['vus'] = (vus_score(y_scores, y_true, vus_win)
                             if vus_win is not None else float('nan'))
        value = combine_metrics(metric, values)
        return _best_threshold_f1(y_true, y_scores) if np.isnan(value) else value

    return score


def compute_meta_pfi(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    score_fn: Callable[[np.ndarray, np.ndarray], float] = _best_threshold_f1,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Permutation feature importance of the meta-learner: importance of feature i =
    baseline_score − mean over n_repeats of the score after shuffling column i.
    score_fn defaults to best-threshold F1 (matches GA fitness); injectable.
    """
    d = len(feature_names)
    n = X.shape[0]
    if d == 0 or n == 0:
        return {f: float('nan') for f in feature_names}
    y = np.asarray(y).flatten()
    base_scores = np.nan_to_num(np.asarray(predict_fn(X), float),
                                nan=0.0, posinf=1.0, neginf=0.0)
    baseline = score_fn(y, base_scores)
    rng = np.random.RandomState(random_state)
    out: Dict[str, float] = {}
    for i in range(d):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, i] = Xp[rng.permutation(n), i]
            sp = np.nan_to_num(np.asarray(predict_fn(Xp), float),
                               nan=0.0, posinf=1.0, neginf=0.0)
            drops.append(baseline - score_fn(y, sp))
        out[feature_names[i]] = float(np.mean(drops))
    return out


# The sign is the sign of ALE's net accumulated effect. It is ALWAYS reported
# when there is a net effect to take the sign of — withholding it turned a
# measured negative into a blank, which reads as missing data rather than as the
# weakly-supported finding it is. The two gates below no longer decide WHETHER a
# sign is shown, only whether it is shown as well supported; each has a reason
# rather than a round number:
#
# ALE_CONSISTENCY_MIN. Consistency is |net| / total. If P is the movement in the
# dominant direction and N against it, then consistency = (P−N)/(P+N), so the
# dominant share is (1+consistency)/2. A threshold of 0.6 therefore means "at
# least 80% of everything the meta-learner did went one way" — the number is
# readable as a statement, which a bare cut-off is not. Observed on real runs:
# SKAB/7 gave 1.00, 0.78, 0.78, 0.63, 0.47 and SMD/machine-1-6 gave 0.88, 0.71,
# 0.58, 0.29, 0.19, 0.18, 0.14, so 0.6 falls in a gap on both rather than
# splitting a cluster.
#
# ALE_MAGNITUDE_FLOOR. Consistency alone would label noise: a detector that
# barely moves the model can still move it consistently (SKAB/7's NN_3 scored
# consistency 1.00 on a total of 0.012, and a synthetic noise column scored 0.61
# on 0.066). The floor is relative to the strongest detector in the same
# ensemble because the units are that meta-learner's own probability scale.
ALE_N_BINS = 10
ALE_CONSISTENCY_MIN = 0.6
ALE_MAGNITUDE_FLOOR = 0.05      # as a fraction of the largest total variation


def compute_meta_ale(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_names: List[str],
    n_bins: int = ALE_N_BINS,
) -> Dict[str, Dict[str, Any]]:
    """
    Accumulated Local Effects of the meta-learner, per detector column.

    For each feature the observed range is cut into `n_bins` quantile bins. In
    each bin, ONLY the rows whose value falls in that bin are used, and only
    that feature is moved — from the bin's lower edge to its upper edge. Every
    other detector keeps the value it actually had.

    That locality is the whole point. It makes the measurement interventional,
    so it isolates this detector's own influence rather than everything
    correlated with it — while never asking the model about a combination that
    does not occur, which is what breaks PDP on correlated score columns.

    Per feature:
      deltas          the per-bin local effects, in probability units
      edges           the bin boundaries (len(deltas) + 1)
      curve           the accumulated effect at each edge (len(edges)); the
                      ALE curve, starting at 0
      net             sum(deltas) — where the curve ends; its SIGN is the
                      detector's sign
      total_variation sum(|deltas|) — how much the model moved in total; this
                      is the magnitude, and unlike `net` it cannot cancel
      consistency     |net| / total_variation in [0, 1]; 1 means every bin
                      pushed the same way, 0 means the ups exactly cancel the
                      downs and NO single sign is honest

    A constant (or near-constant) column has fewer than two distinct bin edges;
    everything is NaN for it and the caller reports no sign.
    """
    d, n = len(feature_names), X.shape[0]
    out: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(feature_names):
        blank = {"deltas": [], "edges": [], "curve": [],
                 "net": float("nan"), "total_variation": float("nan"),
                 "consistency": float("nan"), "n_bins": 0}
        if d == 0 or n == 0:
            out[name] = blank
            continue
        column = np.asarray(X[:, i], dtype=float)
        edges = np.unique(np.quantile(column, np.linspace(0.0, 1.0, int(n_bins) + 1)))
        if edges.size < 2:
            out[name] = blank
            continue

        deltas: List[float] = []
        for k in range(1, edges.size):
            lo_e, hi_e = float(edges[k - 1]), float(edges[k])
            # The lowest bin is closed on both sides so the minimum row is used.
            rows = ((column > lo_e) & (column <= hi_e) if k > 1
                    else (column >= lo_e) & (column <= hi_e))
            if not rows.any():
                deltas.append(0.0)
                continue
            lo = X[rows].astype(float).copy()
            hi = lo.copy()
            lo[:, i] = lo_e
            hi[:, i] = hi_e
            p_lo = np.nan_to_num(np.asarray(predict_fn(lo), float),
                                 nan=0.0, posinf=1.0, neginf=0.0)
            p_hi = np.nan_to_num(np.asarray(predict_fn(hi), float),
                                 nan=0.0, posinf=1.0, neginf=0.0)
            deltas.append(float(np.mean(p_hi - p_lo)))

        arr = np.asarray(deltas, dtype=float)
        net, tv = float(arr.sum()), float(np.abs(arr).sum())
        out[name] = {
            "deltas": [float(v) for v in arr],
            "edges": [float(v) for v in edges],
            "curve": [0.0] + [float(v) for v in np.cumsum(arr)],
            "net": net,
            "total_variation": tv,
            "consistency": (abs(net) / tv) if tv > 0 else float("nan"),
            "n_bins": int(arr.size),
        }
    return out


def ale_signs(
    ale: Dict[str, Dict[str, Any]],
    feature_names: List[str],
) -> Dict[str, str]:
    """
    The sign of the net accumulated effect, per detector: "positive" /
    "negative" / "not_available".

    The sign is read off `net` and nothing else. A detector whose curve wanders
    still ends somewhere, and where it ends is a measurement; how well that end
    point is supported is a separate question, answered by `ale_sign_support`.
    Folding the two together suppressed real negatives and left them looking
    like missing data.

    "not_available" is reserved for the two cases with genuinely no sign to
    take: ALE could not be computed at all (a constant column has no bins to
    walk), or the net effect is exactly zero, where "positive" and "negative"
    are equally wrong. A detector the meta-learner ignores outright lands in the
    latter.
    """
    out: Dict[str, str] = {}
    for name in feature_names:
        rec = ale.get(name) or {}
        net = rec.get("net", float("nan"))
        if not rec.get("n_bins") or net != net or net == 0.0:
            out[name] = "not_available"
        else:
            out[name] = "positive" if net > 0 else "negative"
    return out


def ale_sign_support(
    ale: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    consistency_min: float = ALE_CONSISTENCY_MIN,
    magnitude_floor: float = ALE_MAGNITUDE_FLOOR,
) -> Dict[str, List[str]]:
    """
    Why a reported sign should be read with care, per detector: a list holding
    any of "low_consistency" and "weak_influence", empty when neither applies.

    low_consistency  the curve turns — the sign is where it happened to end up,
                     not a property of the whole range.
    weak_influence   the detector barely moves the meta-learner relative to the
                     strongest one here, so the sign is of a small effect.

    Detectors with no sign at all get an empty list: there is nothing to qualify.
    """
    tvs = [(ale.get(f) or {}).get("total_variation", float("nan")) for f in feature_names]
    finite = [v for v in tvs if v == v]
    floor = magnitude_floor * (max(finite) if finite else 0.0)

    out: Dict[str, List[str]] = {}
    signed = ale_signs(ale, feature_names)
    for name in feature_names:
        rec = ale.get(name) or {}
        reasons: List[str] = []
        if signed.get(name) in ("positive", "negative"):
            tv = rec.get("total_variation", float("nan"))
            cons = rec.get("consistency", float("nan"))
            if cons != cons or cons < consistency_min:
                reasons.append("low_consistency")
            if tv != tv or tv < floor:
                reasons.append("weak_influence")
        out[name] = reasons
    return out


def markov_aggregate_importances(
    importances_by_method: Dict[str, Dict[str, float]],
    feature_names: List[str],
    smoothing: float = 0.1,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Markov-chain rank aggregation over each method's importance ranking — the same
    family as enhanced_markov_chain_rank_aggregator_text in
    Model_Selection/rank_aggregation.py, reimplemented here (numpy-only) so the
    Metrics explainability layer stays free of that module's heavy deps and exposes
    a per-feature stationary score.

    Each method induces a best-first ranking (descending importance; NaN last). A
    pairwise preference matrix C[i,j] counts the methods that rank i above j. The
    chain moves preferentially toward the better item: P[i,j] ∝ sigmoid(C[j,i] −
    C[i,j]) for i≠j, with a small Laplace `smoothing` added to every off-diagonal so
    the chain is irreducible/aperiodic (a unique stationary distribution exists);
    rows are then normalised. The stationary distribution π (left eigenvector of P
    for eigenvalue 1) is the per-feature Markov score — higher π = stronger consensus
    importance. Returns ({feature: π}, final_ranking_descending).
    """
    n = len(feature_names)
    if n == 0:
        return {}, []
    if n == 1:
        return {feature_names[0]: 1.0}, list(feature_names)

    idx = {f: i for i, f in enumerate(feature_names)}

    # Pairwise preference counts from each method's best-first ranking.
    C = np.zeros((n, n), dtype=float)
    for imp in importances_by_method.values():
        order = sorted(
            feature_names,
            key=lambda f: (imp.get(f, float('nan'))
                           if not np.isnan(imp.get(f, float('nan'))) else -np.inf),
            reverse=True,
        )
        for a in range(n):
            for b in range(a + 1, n):
                C[idx[order[a]], idx[order[b]]] += 1.0  # order[a] preferred over order[b]

    # Transition matrix: move toward the better item; Laplace smoothing → ergodic.
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = 1.0 / (1.0 + np.exp(-(C[j, i] - C[i, j]))) + smoothing
    P /= P.sum(axis=1, keepdims=True)

    # Stationary distribution: left eigenvector of P for the eigenvalue closest to 1.
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(vecs[:, k].real)
    total = pi.sum()
    pi = pi / total if total > 0 else np.full(n, 1.0 / n)

    scores = {f: float(pi[idx[f]]) for f in feature_names}
    final_ranking = sorted(feature_names, key=lambda f: scores[f], reverse=True)
    return scores, final_ranking


# Points closer than this are one tie. When the points are Markov scores from
# markov_aggregate_importances, exact ties are common — three measures feed the
# chain, so C[j,i] - C[i,j] takes only five values — and np.linalg.eig returns
# those ties a few ulp apart. An exact `!=` promoted a 1.1e-16 eigenvector
# wobble into a real rank difference, letting the solver rather than the data
# decide which feature led. (Ties were the NORM at two measures, where the
# difference took three values; the third source made them rarer, not rare.)
_RANK_TIE_ATOL = 1e-9


def _competition_ranks(points: Dict[str, float], order: List[str]) -> Dict[str, int]:
    """
    Standard competition ranking ("1224"): features with equal points share the
    smallest rank in their group (so two features tied for 2nd are both rank 2 and
    the next is rank 4). `order` must be the points-descending feature order.
    Equality is within `_RANK_TIE_ATOL`, compared against the running block's
    points rather than the previous feature so a drift of near-equal values
    cannot chain a long run into one rank.
    """
    ranks: Dict[str, int] = {}
    block_pts = None
    rank = 0
    for i, f in enumerate(order):
        pts = points[f]
        if block_pts is None or abs(float(pts) - float(block_pts)) > _RANK_TIE_ATOL:
            rank = i + 1
            block_pts = pts
        ranks[f] = rank
    return ranks


def plot_ga_combination(
    shap_abs: Dict[str, float],
    pfi_imp: Dict[str, float],
    ale_total: Dict[str, float],
    markov_scores: Dict[str, float],
    final_ranking: List[str],
    feature_names: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Two-panel summary of the meta-learner weighting.
      Left  — grouped horizontal bars per detector: mean|SHAP|, PFI and total
              |ALE|, each normalised to its own max so the three are comparable.
              All three are magnitudes, all three feed the ranking on the right,
              and none can be negative — the panel answers "how much", only.
      Right — Markov final ranking: horizontal bars of the stationary-probability
              score, winner on top.

    The sign is deliberately absent here. It lives in the companion ALE-curve
    figure, which can show a detector whose effect changes sign across its range
    — something a single bar cannot represent without reading as "unimportant".

    Saves to ga_combination_importance_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()

    def _norm(dct):
        vals = [dct.get(f, float('nan')) for f in feature_names]
        m = np.nanmax(np.abs(vals)) if np.any(~np.isnan(vals)) else 0.0
        m = m if m > 0 else 1.0
        return [0.0 if np.isnan(v) else v / m for v in vals]

    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    height = max(4, 0.6 * len(feature_names) + 2)

    # ── Figure 1: the three magnitude measures ──────────────────────────────
    fig, ax_imp = plt.subplots(figsize=(9, height))
    y = np.arange(len(feature_names))
    h = 0.27
    ax_imp.barh(y - h, _norm(shap_abs), height=h, label="mean|SHAP|", color="#1f77b4")
    ax_imp.barh(y, _norm(pfi_imp), height=h, label="PFI", color="#ff7f0e")
    ax_imp.barh(y + h, _norm(ale_total), height=h, label="total |ALE|", color="#2ca02c")
    ax_imp.axvline(0, color="black", linewidth=0.6)
    ax_imp.set_yticks(y)
    ax_imp.set_yticklabels(list(feature_names))
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance (normalised to each method's max |·|)")
    ax_imp.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    # Outside the axes. In the corner it sat on top of whichever detector had
    # the shortest bars, and which detector that is changes per entity.
    ax_imp.legend(loc="upper left", frameon=False,
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout(pad=1.2)
    plt.savefig(f"{directory}/ga_combination_importance_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 2: the consensus ranking those three feed ────────────────────
    #
    # Its own figure rather than a second panel: side by side, the ranking's
    # tick labels had to be written either into the gutter (where they reached
    # the left panel) or on the outside edge (where they read as belonging to
    # nothing). Alone, they go back where a bar chart's labels belong.
    ranked = list(final_ranking)
    pts = [markov_scores.get(f, 0.0) for f in ranked]
    yb = np.arange(len(ranked))
    # Competition ranks so tied Markov scores share a rank number on the labels.
    rk = _competition_ranks(markov_scores, ranked)
    fig, ax_markov = plt.subplots(figsize=(9, height))
    ax_markov.barh(yb, pts, color="#2ca02c")
    ax_markov.set_yticks(yb)
    ax_markov.set_yticklabels([f"{rk[f]}. {f}" for f in ranked])
    ax_markov.invert_yaxis()
    ax_markov.set_xlabel("Markov score (stationary prob.)")
    ax_markov.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(pad=1.2)
    plt.savefig(f"{directory}/ga_combination_ranking_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


_ALE_SIGN_COLOUR = {"positive": "#2ca02c", "negative": "#c92a2a",
                         "mixed": "#7f7f7f", "not_available": "#bbbbbb"}

# What a support reason is called on the figure. Spelled out rather than shown
# as a flag, because the panel is where a reader decides how much to trust the
# sign in its title.
_ALE_SUPPORT_LABEL = {"low_consistency": "low consistency",
                      "weak_influence": "weak influence"}


def plot_ga_combination_agreement(
    ranks: Dict[str, Dict[str, int]],
    feature_names: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Where the three attribution methods disagree about a detector's rank.

    The Markov chain merges them into one order, and a merged order looks
    equally confident whether its sources agreed or not.

    Saves to ga_combination_agreement_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    methods = [m for m in ("SHAP_abs", "PFI", "ALE", "Markov") if m in ranks]
    if len(methods) < 2 or not feature_names:
        return
    labels = {"SHAP_abs": "mean |SHAP|", "PFI": "PFI", "ALE": "ALE",
              "Markov": "Markov"}

    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(methods) + 2),
                                    max(4, 0.28 * n + 1.8)))
    x = np.arange(len(methods))
    # Coloured by the merged rank, so a line can be followed across without a
    # legend of 17 names.
    cmap = plt.get_cmap("viridis")
    final = ranks.get("Markov") or ranks[methods[0]]
    for f_ in feature_names:
        ys = [ranks[m][f_] for m in methods]
        ax.plot(x, ys, color=cmap((final[f_] - 1) / max(1, n - 1)),
                linewidth=1.3, marker="o", markersize=3.5, alpha=0.85)
    for f_ in feature_names:
        ax.annotate(abbreviate_detector(f_), xy=(0, ranks[methods[0]][f_]),
                    xytext=(-6, 0), textcoords="offset points", fontsize=8,
                    ha="right", va="center")
        ax.annotate(abbreviate_detector(f_),
                    xy=(len(methods) - 1, ranks[methods[-1]][f_]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8,
                    ha="left", va="center")

    spread = max(max(ranks[m][f_] for m in methods)
                 - min(ranks[m][f_] for m in methods) for f_ in feature_names)
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in methods])
    ax.set_ylabel("Rank  (1 = most important)")
    ax.set_ylim(n + 0.6, 0.4)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.annotate(f"maximum rank spread: {spread}",
                xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                fontsize=9, color="#555555")

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_combination_agreement_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_combination_ale(
    ale: Dict[str, Dict[str, Any]],
    signs: Dict[str, str],
    feature_names: List[str],
    dataset: str,
    entity: str,
    support: Optional[Dict[str, List[str]]] = None,
    mark_bins: bool = False,
) -> None:
    """
    One ALE curve per detector: how the meta-learner's predicted anomaly
    probability accumulates as that detector sweeps its own observed range.

    Small multiples with a SHARED Y AXIS and independent x axes. Detectors emit
    scores on their own scales, so one shared x would put unrelated values on
    top of each other; the shared y is what makes the magnitudes comparable,
    which is the comparison that matters.

    Reading it: a curve that climbs means higher scores from this detector push
    the ensemble toward flagging an anomaly; falling means toward normal; one
    that rises and then falls ends with a sign that only describes where it
    happened to finish — that case is drawn dashed and labelled.

    mark_bins : bool
        Draw the quantile bin edges the curve is built from, as vertical rules
        under the curve. Every vertex IS a bin edge, so this adds no data — it
        makes the resolution visible, which is what tells a reader whether a
        turn in the curve is structure or one coarse bin. Saved as a separate
        figure so the plain curve stays uncluttered.

    Saves to ga_combination_ale{_bins}_{dataset}_{entity}.png.
    """
    usable = [f for f in feature_names if (ale.get(f) or {}).get("n_bins")]
    if not usable:
        return
    support = support or {}
    _ga_plot_rcparams()

    ncols = min(3, len(usable))
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharey=True,
                             figsize=(4.2 * ncols, 3.1 * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]

    any_weak = False
    for ax, name in zip(flat, usable):
        rec = ale[name]
        edges, curve = rec["edges"], rec["curve"]
        colour = _ALE_SIGN_COLOUR.get(signs.get(name), "#7f7f7f")
        reasons = list(support.get(name) or [])
        any_weak = any_weak or bool(reasons)
        if mark_bins:
            # Navy rather than grey: the grid is already grey dashes, so grey
            # rules read as part of it instead of as the bin structure.
            for edge in edges:
                ax.axvline(edge, color="#1f3d7a", linewidth=0.7, alpha=0.55,
                           zorder=0)
        ax.plot(edges, curve, marker="o", markersize=3, linewidth=1.6,
                color=colour, linestyle="--" if reasons else "-")
        ax.axhline(0, color="black", linewidth=0.7)
        # The qualifier belongs beside the sign, not in a corner: the title is
        # the only part of the panel a reader is guaranteed to take away.
        qualifier = (f" ({', '.join(_ALE_SUPPORT_LABEL.get(r, r) for r in reasons)})"
                     if reasons else "")
        sign = signs.get(name, "not_available")
        label = "no sign (no net effect)" if sign == "not_available" else sign
        # Wrapped, not truncated. A panel is ~4.2in wide and the longest title
        # this can produce — "SpectralResidual_1 — negative (low consistency,
        # weak influence)" — is 74 characters, which ran off both sides at
        # fontsize 11. Wrapping keeps every word and grows downward, and
        # `constrained_layout`/`tight_layout` reserves the extra height.
        ax.set_title("\n".join(textwrap.wrap(f"{name} — {label}{qualifier}", 34)),
                     fontsize=10)
        ax.set_xlabel(f"{name} score" + (f" ({rec['n_bins']} bins)" if mark_bins else ""))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        cons = rec["consistency"]
        ax.text(0.02, 0.02,
                f"net {rec['net']:+.4f}\ntotal {rec['total_variation']:.4f}\n"
                f"consistency {'N/A' if cons != cons else f'{cons:.2f}'}",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                color="dimgrey")
    for ax in flat[len(usable):]:
        ax.set_visible(False)
    # One figure-level label: the y axis is shared, so repeating it per row just
    # crowds the left edge with the same sentence.
    ylabel = "Accumulated effect on P(anomaly)"
    if hasattr(fig, "supylabel"):
        fig.supylabel(ylabel)
    else:
        axes[0][0].set_ylabel(ylabel)

    skipped = [f for f in feature_names if f not in usable]
    note = ("Curve rising = higher scores push toward anomaly; falling = toward "
            "normal. The sign is where the curve ends.")
    if any_weak:
        note += ("  Dashed = the sign is weakly supported, for the reason named "
                 "in the panel title.")
    if mark_bins:
        note += ("  Vertical rules are the quantile bin edges; the curve has one "
                 "vertex per edge and no detail between them.")
    if skipped:
        note += f"  Not shown (no ALE could be computed): {', '.join(skipped)}."
    # WRAPPED to a fixed width, and it must stay that way: the note sits
    # outside the axes, `bbox_inches="tight"` grows the image to contain it, so
    # an unwrapped note makes the bins figure save wider than the plain one and
    # the card then scales the two to different sizes.
    fig.text(0.5, -0.01, textwrap.fill(note, 108),
             ha="center", va="top", fontsize=8, color="dimgrey")

    plt.tight_layout(pad=1.2)
    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    stem = "ga_combination_ale_bins" if mark_bins else "ga_combination_ale"
    plt.savefig(f"{directory}/{stem}_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def explain_ga_combination(
    best_ensemble: List[str],
    algorithm_list: List[str],
    base_model_predictions_train: np.ndarray,
    base_model_predictions_test: np.ndarray,
    y_true_train: np.ndarray,
    y_true_test: np.ndarray,
    meta_model_type: str,
    dataset: str,
    entity: str,
    meta_model: Any = None,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_explain: int = 200,
    explain: bool = False,
    metric: str = DEFAULT_DECISION_METRICS,
    vus_win: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Combination-layer explainability: attribute the best-ensemble meta-learner's
    output to its detector score-columns via SHAP and PFI, then merge the three
    magnitude rankings (mean|SHAP| and PFI) with a Markov-chain rank aggregation;
    ALE supplies each detector's sign; signed SHAP is superseded.
    Writes a report + plot under results/GA_Ens/{dataset}/{entity}/ and returns a
    dict (None if explain=False).

    The meta-learner is, in priority order: an injected predict_fn (tests); the
    GA's actual captured meta_model; or — only as a defensive fallback — a freshly
    trained model of meta_model_type. The captured model is validated against the
    feature count and the fallback is used on any mismatch.
    """
    if not explain or not best_ensemble:
        return None

    feature_names = [a for a in algorithm_list if a in best_ensemble]
    d = len(feature_names)
    if d == 0:
        return None

    header = np.array(algorithm_list)
    mask = np.isin(header, best_ensemble)
    X_train_f = np.asarray(base_model_predictions_train, dtype=float)[:, mask]
    X_test_f = np.asarray(base_model_predictions_test, dtype=float)[:, mask]
    X_train_f = np.nan_to_num(X_train_f, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_f = np.nan_to_num(X_test_f, nan=0.0, posinf=1.0, neginf=0.0)

    used_source = "injected"
    if predict_fn is None:
        candidate = meta_model
        # Validate the captured model's expected feature width when discoverable.
        ok = candidate is not None
        n_in = getattr(candidate, "n_features_in_", None)
        if ok and n_in is not None and n_in != d:
            ok = False
        if ok:
            predict_fn = lambda Z: candidate.predict_proba(Z)[:, 1]
            used_source = "captured"
        else:
            # Defensive fallback: retrain a meta-model of the requested type.
            yt = np.asarray(y_true_train).flatten()
            if len(np.unique(yt)) < 2:
                yt = inject_synthetic_anomalies(yt)
            trainer = {
                'lr': train_meta_model, 'rf': train_meta_model_rf,
                'gbm': train_meta_model_gbm, 'svm': train_meta_model_svm,
            }.get(meta_model_type, train_meta_model_rf)
            model = trainer(X_train_f, yt)
            predict_fn = lambda Z: model.predict_proba(Z)[:, 1]
            used_source = "retrained_fallback"

    # Subsample explained rows for SHAP speed (deterministic).
    n_test = X_test_f.shape[0]
    if n_test > max_explain:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_test, size=max_explain, replace=False)
        X_explain = X_test_f[idx]
    else:
        X_explain = X_test_f
    # Median, not mean: detector scores are heavy-tailed, so a column mean lands in
    # the upper quartile of every detector at once — a combination no real row has.
    # From the test rows the values are explained on, not the training split.
    baseline_row = (np.median(X_test_f, axis=0) if X_test_f.shape[0] > 0
                    else np.zeros(d))

    # One 2^d enumeration, both summaries. These used to be two identical passes.
    phi = compute_meta_shap_values(predict_fn, X_explain, baseline_row, d)
    shap_abs = {f: float(np.abs(phi[:, i]).mean()) for i, f in enumerate(feature_names)}
    shap_signed = {f: float(phi[:, i].mean()) for i, f in enumerate(feature_names)}
    pfi_imp = compute_meta_pfi(predict_fn, X_test_f, y_true_test, feature_names,
                               score_fn=score_fn_for(metric, vus_win))
    # ALE on the FULL test set, matching PFI rather than SHAP's subsample: at
    # ~2*n*d predictions it is an order of magnitude cheaper than the SHAP pass,
    # so subsampling would buy nothing.
    ale = compute_meta_ale(predict_fn, X_test_f, feature_names, n_bins=ALE_N_BINS)
    ale_total = {f: ale[f]["total_variation"] for f in feature_names}
    ale_net = {f: ale[f]["net"] for f in feature_names}
    ale_consistency = {f: ale[f]["consistency"] for f in feature_names}
    signs = ale_signs(ale, feature_names)
    sign_support = ale_sign_support(ale, feature_names)

    # Three magnitude measures feed the chain. ALE contributes its TOTAL
    # VARIATION, not its net: the net cancels for a detector whose effect
    # changes sign across its range, which is exactly how signed SHAP came to
    # rank a genuinely influential detector as unimportant.
    markov_scores, final_ranking = markov_aggregate_importances(
        {"SHAP_abs": shap_abs, "PFI": pfi_imp, "ALE": ale_total}, feature_names)

    baseline_f1 = _best_threshold_f1(
        y_true_test, np.nan_to_num(np.asarray(predict_fn(X_test_f), float),
                                   nan=0.0, posinf=1.0, neginf=0.0))

    plot_ga_combination(shap_abs, pfi_imp, ale_total, markov_scores, final_ranking,
                        feature_names, dataset, entity)
    plot_ga_combination_ale(ale, signs, feature_names, dataset, entity,
                            support=sign_support)
    plot_ga_combination_ale(ale, signs, feature_names, dataset, entity,
                            support=sign_support, mark_bins=True)

    # Per-method ranks (1 = most important / most positive).
    def _ranks(imp):
        order = sorted(feature_names,
                       key=lambda f: (imp[f] if not np.isnan(imp[f]) else -np.inf),
                       reverse=True)
        return {f: i + 1 for i, f in enumerate(order)}
    shap_abs_rank, shap_signed_rank = _ranks(shap_abs), _ranks(shap_signed)
    pfi_rank, ale_rank = _ranks(pfi_imp), _ranks(ale_total)

    plot_ga_combination_agreement(
        {"SHAP_abs": shap_abs_rank, "PFI": pfi_rank, "ALE": ale_rank,
         "Markov": _ranks(markov_scores)},
        feature_names, dataset, entity)

    directory = results_dir("GA_Ens", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory, f"ga_combination_explainability_{dataset}_{entity}.txt")
    with open(report_path, "w") as f:
        f.write("=== GA Ensemble Combination Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Best ensemble : {list(best_ensemble)}\n")
        f.write(f"Meta-learner  : {meta_model_type}  (model source: {used_source})\n")
        f.write(f"Features (detector score columns): {d}\n")
        f.write(f"Baseline meta-learner F1 (best threshold): {baseline_f1:.4f}\n\n")

        f.write("--- SHAP |.| (mean |SHAP|: magnitude of contribution; label-free) ---\n")
        f.write(f"      {'detector':<14} {'mean|SHAP|':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: shap_abs_rank[x]):
            v = shap_abs[f_]
            vs = f"{v:.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {shap_abs_rank[f_]:>6}\n")

        f.write("\n--- PFI (fitness drop when the detector's column is shuffled; label-based) ---\n")
        f.write(f"      {'detector':<14} {'drop':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: pfi_rank[x]):
            v = pfi_imp[f_]
            vs = f"{v:+.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {pfi_rank[f_]:>6}\n")

        f.write("\n--- ALE (accumulated local effects; magnitude, sign and "
                "consistency; label-free) ---\n")
        f.write(f"      {'detector':<14} {'total |ALE|':>12} {'rank':>6} "
                f"{'net':>12} {'consist.':>9} {'sign':>14} {'support':>32}\n")
        f.write("      " + "-" * 104 + "\n")
        for f_ in sorted(feature_names, key=lambda x: ale_rank[x]):
            tv, nt = ale_total[f_], ale_net[f_]
            cs = ale_consistency[f_]
            reasons = sign_support.get(f_) or []
            if signs[f_] not in ("positive", "negative"):
                sup = "-"
            elif reasons:
                sup = ", ".join(_ALE_SUPPORT_LABEL.get(r, r) for r in reasons)
            else:
                sup = "strong"
            f.write(f"      {f_:<14} "
                    f"{('N/A' if np.isnan(tv) else f'{tv:.6f}'):>12} {ale_rank[f_]:>6} "
                    f"{('N/A' if np.isnan(nt) else f'{nt:+.6f}'):>12} "
                    f"{('N/A' if np.isnan(cs) else f'{cs:.2f}'):>9} "
                    f"{signs[f_]:>14} {sup:>32}\n")
        f.write(f"      (sign = sign of net, always reported when net is non-zero; "
                f"support is 'strong' at consistency >= {ALE_CONSISTENCY_MIN} and "
                f"total |ALE| >= {ALE_MAGNITUDE_FLOOR:g} x the largest; "
                f"{ALE_N_BINS} bins)\n")

        # Kept for provenance only. This was the sign until ALE
        # replaced it; nothing downstream reads it any more. It is retained so
        # the change can be audited side by side — see the closing Note.
        f.write("\n--- SHAP signed (SUPERSEDED BY ALE; retained for comparison) ---\n")
        f.write(f"      {'detector':<14} {'mean SHAP':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: shap_signed_rank[x]):
            v = shap_signed[f_]
            vs = f"{v:+.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {shap_signed_rank[f_]:>6}\n")

        f.write("\n--- Markov aggregation (SHAP |.| + PFI + ALE) ---\n")
        f.write(f"      {'detector':<14} {'|SHAP| rk':>9} {'PFI rk':>7} {'ALE rk':>7} "
                f"{'sign':>14} {'Markov π':>10}\n")
        f.write("      " + "-" * 72 + "\n")
        for f_ in final_ranking:
            # A trailing * flags a sign the ALE table qualifies, so this summary
            # never reads as more certain than the section it summarises.
            mark = "*" if (sign_support.get(f_) or []) else ""
            f.write(f"      {f_:<14} {shap_abs_rank[f_]:>9} {pfi_rank[f_]:>7} "
                    f"{ale_rank[f_]:>7} {signs[f_] + mark:>14} "
                    f"{markov_scores[f_]:>10.4f}\n")
        if any(sign_support.get(f_) for f_ in final_ranking):
            f.write("      (* weakly supported sign — see the support column above)\n")
        # Final ranking with ties shown as equals (e.g. "1.A > 2.B = C > 4.D").
        ranks = _competition_ranks(markov_scores, final_ranking)
        groups: List[Tuple[int, List[str]]] = []
        for f_ in final_ranking:
            r = ranks[f_]
            if groups and groups[-1][0] == r:
                groups[-1][1].append(f_)
            else:
                groups.append((r, [f_]))
        f.write("\nFinal ranking (Markov): "
                + " > ".join(f"{r}.{' = '.join(fs)}" for r, fs in groups) + "\n")

        # ── Method agreement ───────────────────────────────────────────────
        markov_rank = _ranks(markov_scores)
        f.write("\n--- Agreement between the attribution methods ---\n")
        f.write("Rank spread is the widest gap between the three methods for "
                "one detector, so a large value means the merged order is "
                "resolving a disagreement rather than reporting a consensus.\n")
        f.write(f"\n      {'detector':<14} {'|SHAP|':>8} {'PFI':>6} {'ALE':>6} "
                f"{'Markov':>8} {'spread':>8}\n")
        f.write("      " + "-" * 54 + "\n")
        for f_ in sorted(feature_names, key=lambda x: markov_rank[x]):
            trio = [shap_abs_rank[f_], pfi_rank[f_], ale_rank[f_]]
            f.write(f"      {f_:<14} {shap_abs_rank[f_]:>8} {pfi_rank[f_]:>6} "
                    f"{ale_rank[f_]:>6} {markov_rank[f_]:>8} "
                    f"{max(trio) - min(trio):>8}\n")

        f.write("\nNote: three magnitude measures feed the ranking. mean|SHAP| = the size of the "
                "detector's influence on the meta-learner's output (label-free, measured on a "
                "fixed 200-row sample whereas PFI and ALE use every row); PFI = the fitness drop "
                "when its column is shuffled (label-based); total |ALE| = how far the output moves "
                "in total as the detector sweeps its own observed range (label-free). A "
                "Markov-chain rank aggregation over the pairwise preferences of all three gives "
                "π, each detector's stationary probability (higher = stronger consensus).\n"
                "\nThe sign is the sign of ALE's NET accumulated effect — where the curve "
                "ends — and is reported whenever there is a non-zero net to take the sign of. "
                "How well that sign is supported is reported separately: consistency = "
                "|net| / total, so 1.0 means every bin pushed the same way and 0.0 means the "
                "ups cancel the downs, and the magnitude gate asks whether the detector moves "
                "the meta-learner enough for the question to arise. A sign failing either is "
                "still shown, marked as weakly supported — withholding it hid measured "
                "negatives behind what looked like missing data.\n"
                "\nSigned SHAP used to supply the sign and no longer does. It is measured "
                "against one synthetic 'every detector at its median' row; when the detector "
                "columns are correlated that row does not occur in the data, the meta-learner "
                "extrapolates there, and the resulting sign can be wrong outright — a contrarian "
                "detector scored a confident positive in testing. Its table above is retained "
                "only so the two can be compared.\n")

    result = {
        "best_ensemble": list(best_ensemble),
        "feature_names": feature_names,
        "meta_model_type": meta_model_type,
        "model_source": used_source,
        "baseline_f1": baseline_f1,
        "shap_importance": shap_abs,
        # Superseded by ALE for the sign; kept for the report's comparison
        # table only. The IR no longer reads it.
        "shap_signed_importance": shap_signed,
        "pfi_importance": pfi_imp,
        "ale_total_variation": ale_total,
        "ale_net": ale_net,
        "ale_consistency": ale_consistency,
        "ale_sign": signs,
        # Per detector, why its sign should be read with care ([] = no caveat).
        "ale_sign_support": sign_support,
        "ale_curves": {f: {"edges": ale[f]["edges"], "curve": ale[f]["curve"]}
                       for f in feature_names},
        "ale_n_bins": ALE_N_BINS,
        "ale_consistency_min": ALE_CONSISTENCY_MIN,
        "ale_magnitude_floor": ALE_MAGNITUDE_FLOOR,
        "markov_scores": markov_scores,
        "final_ranking": final_ranking,
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        ir.write_stage_ir(ir.build_ga_combination_ir(dataset, entity, result),
                           dataset, entity, "ir_ga_combination")
    except Exception as e:
        logger.error(f"GA combination IR emission failed (non-fatal): {e}")

    return result
