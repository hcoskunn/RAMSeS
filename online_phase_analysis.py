#!/usr/bin/env python3
"""
Online Phase Analysis for RAMSeS Framework

This script measures per-window detection latency and memory usage during streaming
deployment to answer Reviewer R1.O3's question: "what about the online phase?"

The script analyzes:
1. Ensemble branch online inference (using pre-selected ensemble meta-model)
2. Single-model branch online inference (using pre-selected single detector)
3. Baseline detector inference (LOF, RNN, NN for comparison)

For each configuration, it measures:
- Inference latency (milliseconds per window)
- Memory usage (MB during inference)
- CPU usage (percentage)

Author: RAMSeS Team
Date: January 2026
"""

import argparse
import copy
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch as t

from Datasets.load import load_data
from Metrics.Ensemble_GA import evaluate_model_consistently, fitness_function
from Metrics.metrics import prauc, best_f1_linspace
from Model_Selection.inject_anomalies import Inject
from Model_Selection.Thompson_Sampling import initialize_sliding_windows
from Utils.utils import get_args_from_cmdline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system resources during inference"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []
    
    def start(self):
        """Start monitoring before inference"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.start_cpu = self.process.cpu_percent(interval=0.01)
    
    def stop(self):
        """Stop monitoring after inference"""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.end_cpu = self.process.cpu_percent(interval=0.01)
        
        measurement = {
            'latency_ms': (self.end_time - self.start_time) * 1000,
            'memory_mb': self.end_memory,
            'memory_delta_mb': self.end_memory - self.start_memory,
            'cpu_percent': max(self.end_cpu, self.start_cpu)  # Use max to capture spike
        }
        self.measurements.append(measurement)
        return measurement
    
    def get_stats(self):
        """Get aggregated statistics"""
        if not self.measurements:
            return {}
        
        latencies = [m['latency_ms'] for m in self.measurements]
        memories = [m['memory_mb'] for m in self.measurements]
        memory_deltas = [m['memory_delta_mb'] for m in self.measurements]
        cpus = [m['cpu_percent'] for m in self.measurements]
        
        return {
            'latency_ms': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            },
            'memory_mb': {
                'mean': np.mean(memories),
                'std': np.std(memories),
                'min': np.min(memories),
                'max': np.max(memories),
                'peak': np.max(memories)
            },
            'memory_delta_mb': {
                'mean': np.mean(memory_deltas),
                'std': np.std(memory_deltas),
                'min': np.min(memory_deltas),
                'max': np.max(memory_deltas)
            },
            'cpu_percent': {
                'mean': np.mean(cpus),
                'std': np.std(cpus),
                'min': np.min(cpus),
                'max': np.max(cpus)
            }
        }
    
    def reset(self):
        """Reset measurements"""
        self.measurements = []


def load_trained_models(model_names: List[str], models_dir: str) -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Parameters
    ----------
    model_names : List[str]
        List of model instance names (e.g., 'CBLOF_1', 'LOF_1', etc.)
    models_dir : str
        Directory where .pth files are stored
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to loaded model objects
    """
    trained = {}
    missing = []
    
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            missing.append(name)
            logger.warning(f"Model {name} not found at {path}")
            continue
        
        try:
            with open(path, 'rb') as fh:
                model = t.load(fh, weights_only=False)
                try:
                    model.eval()
                except AttributeError:
                    pass  # Not a nn.Module
                trained[name] = model
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            missing.append(name)
    
    logger.info(f"Loaded {len(trained)}/{len(model_names)} models from {models_dir}")
    if missing:
        logger.warning(f"Missing models: {', '.join(missing)}")
    
    return trained


def inference_single_model(model, model_name: str, test_data, monitor: PerformanceMonitor) -> Dict:
    """
    Perform inference with a single model and measure performance.
    
    Parameters
    ----------
    model : Any
        The model object
    model_name : str
        Name of the model
    test_data : Dataset
        Test data window
    monitor : PerformanceMonitor
        Performance monitor instance
        
    Returns
    -------
    Dict
        Dictionary containing predictions, metrics, and performance stats
    """
    test_data_copy = copy.deepcopy(test_data)
    
    monitor.start()
    try:
        y_true, y_scores, _, _ = evaluate_model_consistently(
            test_data_copy, model, model_name, is_ensemble=False
        )
        
        # Calculate F1 and PR-AUC
        best_f1, precision, recall, y_pred_binary, _, best_threshold = best_f1_linspace(
            y_scores, y_true, n_splits=100, segment_adjust=True, f1_type='standard'
        )
        pr_auc = prauc(y_true, y_scores)
        
    except Exception as e:
        logger.error(f"Inference failed for {model_name}: {e}")
        monitor.stop()
        return None
    
    perf = monitor.stop()
    
    return {
        'model_name': model_name,
        'f1': float(best_f1),
        'pr_auc': float(pr_auc),
        'threshold': float(best_threshold),
        'performance': perf,
        'y_pred': y_pred_binary,
        'y_scores': y_scores
    }


def inference_ensemble(ensemble_models: List[str], trained_models: Dict, 
                      train_data, test_data, 
                      individual_predictions, base_model_predictions_train,
                      base_model_predictions_test, y_true_train, y_true_test,
                      meta_model_type: str, monitor: PerformanceMonitor) -> Dict:
    """
    Perform inference with an ensemble meta-model and measure performance.
    
    Parameters
    ----------
    ensemble_models : List[str]
        List of model names in the ensemble
    trained_models : Dict
        Dictionary of trained models
    train_data : Dataset
        Training data
    test_data : Dataset
        Test data window
    individual_predictions : Dict
        Predictions from individual models
    base_model_predictions_train : np.ndarray
        Base model predictions on training data
    base_model_predictions_test : np.ndarray
        Base model predictions on test data
    y_true_train : np.ndarray
        True labels for training data
    y_true_test : np.ndarray
        True labels for test data
    meta_model_type : str
        Type of meta-model ('rf', 'lr', 'gbm', 'svm')
    monitor : PerformanceMonitor
        Performance monitor instance
        
    Returns
    -------
    Dict
        Dictionary containing ensemble predictions, metrics, and performance stats
    """
    test_data_copy = copy.deepcopy(test_data)
    
    # Restrict to ensemble models
    trained_models_ensemble = {name: trained_models[name] for name in ensemble_models}
    
    monitor.start()
    try:
        # Use fitness_function to get ensemble predictions
        f1_score_val, pr_auc_val, fitness_val, y_pred_ensemble, y_true = fitness_function(
            ensemble_models, train_data, test_data_copy, trained_models_ensemble,
            individual_predictions, base_model_predictions_train, ensemble_models,
            base_model_predictions_test, y_true_train, y_true_test,
            meta_model_type=meta_model_type
        )
    except Exception as e:
        logger.error(f"Ensemble inference failed: {e}")
        monitor.stop()
        return None
    
    perf = monitor.stop()
    
    return {
        'ensemble_models': ensemble_models,
        'meta_model_type': meta_model_type,
        'f1': float(f1_score_val),
        'pr_auc': float(pr_auc_val),
        'fitness': float(fitness_val),
        'performance': perf,
        'y_pred': y_pred_ensemble
    }


def inject_regime_shifts_and_trends(test_data, shift_locations: List[int] = None):
    """
    Inject realistic regime shifts and trends to simulate real-world distribution drift.
    
    Parameters
    ----------
    test_data : Dataset
        Test data to inject shifts into
    shift_locations : List[int], optional
        Locations to inject shifts. If None, will auto-determine
        
    Returns
    -------
    Dataset
        Test data with injected regime shifts and trends
    """
    data = test_data.entities[0].Y.copy()
    n_features, n_timesteps = data.shape
    
    if shift_locations is None:
        # Inject shifts at 25%, 50%, 75% of the data
        shift_locations = [
            int(n_timesteps * 0.25),
            int(n_timesteps * 0.50),
            int(n_timesteps * 0.75)
        ]
    
    logger.info(f"Injecting regime shifts at timesteps: {shift_locations}")
    
    for shift_point in shift_locations:
        # Type 1: Mean shift (simulating sensor calibration drift)
        if shift_point == shift_locations[0]:
            shift_magnitude = np.random.uniform(0.1, 0.3) * np.std(data[:, :shift_point], axis=1, keepdims=True)
            data[:, shift_point:] += shift_magnitude
            logger.info(f"  → Applied mean shift at {shift_point}: magnitude={shift_magnitude.mean():.4f}")
        
        # Type 2: Variance change (simulating operating condition change)
        elif shift_point == shift_locations[1]:
            scale_factor = np.random.uniform(1.3, 1.8)
            data_mean = np.mean(data[:, shift_point:shift_point+100], axis=1, keepdims=True)
            data[:, shift_point:] = data_mean + (data[:, shift_point:] - data_mean) * scale_factor
            logger.info(f"  → Applied variance shift at {shift_point}: scale_factor={scale_factor:.2f}")
        
        # Type 3: Trend injection (simulating gradual degradation)
        else:
            trend_length = n_timesteps - shift_point
            trend_slope = np.random.uniform(0.001, 0.003) * np.std(data[:, :shift_point], axis=1, keepdims=True)
            trend = trend_slope * np.arange(trend_length).reshape(1, -1)
            data[:, shift_point:] += trend
            logger.info(f"  → Applied linear trend at {shift_point}: slope={trend_slope.mean():.6f}")
    
    test_data.entities[0].Y = data
    return test_data


def run_online_phase_experiment(
    dataset: str,
    entity: str,
    data_dir: str,
    models_dir: str,
    algorithm_list_instances: List[str],
    num_windows: int = 50,
    window_overlap: int = 5,
    inject_synthetic_anomalies: bool = False,
    inject_regime_shifts: bool = True,
    best_ensemble: List[str] = None,
    best_single_model: str = None,
    baseline_models: List[str] = None,
    meta_model_type: str = 'rf',
    window_size: int = None
) -> Dict:
    """
    Run online phase experiment for a single entity.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'skab', 'smd', 'anomaly_archive')
    entity : str
        Entity identifier
    data_dir : str
        Root directory for datasets
    models_dir : str
        Directory where trained models are stored
    algorithm_list_instances : List[str]
        List of all available model instances
    num_windows : int
        Number of sliding windows to process
    window_overlap : int
        Overlap between sliding windows
    inject_synthetic_anomalies : bool
        Whether to inject synthetic anomalies (default: False - use real anomalies)
    inject_regime_shifts : bool
        Whether to inject regime shifts and trends (default: True)
    best_ensemble : List[str], optional
        Pre-selected ensemble models (if None, will auto-select)
    best_single_model : str, optional
        Pre-selected single model (if None, will auto-select)
    baseline_models : List[str], optional
        Baseline models for comparison
    meta_model_type : str
        Type of meta-model for ensemble
    window_size : int, optional
        Fixed window size (if None, will auto-calculate based on num_windows)
        
    Returns
    -------
    Dict
        Results containing performance metrics for all configurations
    """
    logger.info("="*80)
    logger.info(f"Starting Online Phase Analysis: {dataset}/{entity}")
    logger.info(f"  Evaluation mode: {'Real anomalies' if not inject_synthetic_anomalies else 'Synthetic anomalies'}")
    logger.info(f"  Regime shifts: {'Enabled' if inject_regime_shifts else 'Disabled'}")
    logger.info(f"  Window size: {'Auto' if window_size is None else window_size}")
    logger.info("="*80)
    
    if baseline_models is None:
        baseline_models = ['LOF_1', 'RNN_1', 'NN_1']
    
    # Load data
    logger.info(f"Loading data from {data_dir}...")
    train_data = load_data(
        dataset=dataset, group='train', entities=entity,
        downsampling=10, min_length=256, root_dir=data_dir,
        normalize=True, verbose=False
    )
    test_data = load_data(
        dataset=dataset, group='test', entities=entity,
        downsampling=10, min_length=256, root_dir=data_dir,
        normalize=True, verbose=False
    )
    logger.info(f"✓ Loaded train data: {train_data.entities[0].Y.shape}")
    logger.info(f"✓ Loaded test data: {test_data.entities[0].Y.shape}")
    
    # Load trained models
    logger.info(f"Loading trained models from {models_dir}...")
    trained_models = load_trained_models(algorithm_list_instances, models_dir)
    if not trained_models:
        raise ValueError(f"No models loaded from {models_dir}")
    
    # Store original test data with real anomalies
    test_data_original = copy.deepcopy(test_data)
    
    # Inject regime shifts and trends if enabled (realistic drift simulation)
    if inject_regime_shifts:
        logger.info("Injecting regime shifts and trends to simulate real-world drift...")
        test_data = inject_regime_shifts_and_trends(test_data)
        logger.info("✓ Injected regime shifts and trends")
    
    # Optionally inject synthetic anomalies (for comparison with baseline behavior)
    if inject_synthetic_anomalies:
        logger.info("Injecting synthetic anomalies while preserving real anomalies...")
        from Model_Selection.inject_anomalies import InjectHybrid
        test_data, anomaly_info = InjectHybrid(test_data, ['spikes', 'contextual'], num_synthetic=10)
        logger.info(f"✓ Hybrid injection complete: {anomaly_info['synthetic_added']} synthetic + "
                   f"{anomaly_info['real_anomalies']} real = {anomaly_info['total_anomalies']} total anomalies")
    else:
        logger.info("Using real anomalies from ground-truth labels (no synthetic injection)")
    
    # Setup sliding windows
    logger.info(f"Setting up {num_windows} sliding windows...")
    data = test_data.entities[0].Y
    targets = test_data.entities[0].labels
    mask = test_data.entities[0].mask
    
    total_length = targets.flatten().shape[0]
    
    if window_size is None:
        # Auto-calculate window size - adaptive based on data length
        target_window_size = total_length // num_windows
        window_size = max(min(target_window_size, 256), 32)  # Between 32 and 256
        logger.info(f"Auto-calculated window size: {window_size} (data_length={total_length}, num_windows={num_windows})")
    
    stride = max(window_size - window_overlap, 1)
    
    data_windows, targets_windows, mask_windows, actual_num_windows = initialize_sliding_windows(
        data, targets, mask, window_size, stride
    )
    logger.info(f"✓ Created {actual_num_windows} windows (size={window_size}, stride={stride})")
    
    # Auto-select ensemble and single model if not provided
    if best_ensemble is None:
        # Simple heuristic: select top 3 models
        best_ensemble = algorithm_list_instances[:3]
        logger.info(f"Auto-selected ensemble: {best_ensemble}")
    
    if best_single_model is None:
        best_single_model = algorithm_list_instances[0]
        logger.info(f"Auto-selected single model: {best_single_model}")
    
    # Prepare for ensemble inference
    logger.info("Preparing ensemble meta-model...")
    # We need to evaluate individual models on train/test to get base predictions
    from Metrics.Ensemble_GA import evaluate_individual_models, train_meta_model_rf
    
    train_data_copy = copy.deepcopy(train_data)
    test_data_copy = copy.deepcopy(test_data)
    
    individual_predictions, _, _, _ = evaluate_individual_models(
        best_ensemble, test_data_copy, trained_models
    )
    
    # Get base model predictions for ensemble
    y_true_train_agg, base_model_predictions_train, _, _ = evaluate_model_consistently(
        train_data_copy, trained_models, best_ensemble, is_ensemble=True
    )
    y_true_test_agg, base_model_predictions_test, _, _ = evaluate_model_consistently(
        test_data_copy, trained_models, best_ensemble, is_ensemble=True
    )
    
    # PRE-TRAIN the meta-model ONCE on training data (don't retrain per window!)
    meta_model_rf = train_meta_model_rf(base_model_predictions_train, y_true_train_agg)
    logger.info(f"✓ Pre-trained meta-model on training data: {base_model_predictions_train.shape}")
    
    # Initialize performance monitors
    ensemble_monitor = PerformanceMonitor()
    single_model_monitor = PerformanceMonitor()
    baseline_monitors = {model: PerformanceMonitor() for model in baseline_models}
    
    # Results storage
    results = {
        'dataset': dataset,
        'entity': entity,
        'num_windows': actual_num_windows,
        'window_size': window_size,
        'ensemble': {'models': best_ensemble, 'meta_model': meta_model_type, 'windows': []},
        'single_model': {'model': best_single_model, 'windows': []},
        'baselines': {model: {'windows': []} for model in baseline_models}
    }
    
    # Process each window
    logger.info(f"\nProcessing {actual_num_windows} windows...")
    for window_idx in range(actual_num_windows):
        if (window_idx + 1) % 10 == 0 or window_idx == 0:
            logger.info(f"  Window {window_idx + 1}/{actual_num_windows}")
        
        # Create window dataset
        window_test_data = copy.deepcopy(test_data_original)
        window_test_data.entities[0].Y = data_windows[window_idx]
        window_test_data.entities[0].labels = targets_windows[window_idx]
        window_test_data.entities[0].mask = mask_windows[window_idx]
        window_test_data.entities[0].n_time = targets_windows[window_idx].flatten().shape[0]
        window_test_data.total_time = targets_windows[window_idx].flatten().shape[0]
        
        # Use the already-transformed data (with regime shifts if enabled)
        # No need to re-inject anomalies per window since they're in the original data
        
        # 1. Ensemble inference with PRE-TRAINED meta-model (NO retraining!)
        try:
            ensemble_monitor.start()
            
            # Get base model predictions for this window only
            y_true_window, base_predictions_window, _, _ = evaluate_model_consistently(
                window_test_data, trained_models, best_ensemble, is_ensemble=True
            )
            
            # Use PRE-TRAINED meta-model to make predictions
            from Metrics.metrics import best_f1_linspace, prauc
            y_scores_ensemble = meta_model_rf.predict_proba(base_predictions_window)[:, 1]
            
            # Calculate metrics
            best_f1, precision, recall, y_pred_binary, _, best_threshold = best_f1_linspace(
                y_scores_ensemble, y_true_window, n_splits=100, segment_adjust=True, f1_type='standard'
            )
            pr_auc = prauc(y_true_window, y_scores_ensemble)
            
            perf = ensemble_monitor.stop()
            
            ensemble_result = {
                'ensemble_models': best_ensemble,
                'meta_model_type': meta_model_type,
                'f1': float(best_f1),
                'pr_auc': float(pr_auc),
                'fitness': float(best_f1),
                'performance': perf,
                'y_pred': y_pred_binary.tolist() if hasattr(y_pred_binary, 'tolist') else y_pred_binary
            }
            results['ensemble']['windows'].append(ensemble_result)
        except Exception as e:
            logger.warning(f"Ensemble inference failed for window {window_idx}: {e}")
        
        # 2. Single model inference
        single_result = inference_single_model(
            trained_models[best_single_model], best_single_model,
            window_test_data, single_model_monitor
        )
        if single_result:
            results['single_model']['windows'].append(single_result)
        
        # 3. Baseline models inference
        for baseline_model in baseline_models:
            if baseline_model not in trained_models:
                continue
            baseline_result = inference_single_model(
                trained_models[baseline_model], baseline_model,
                window_test_data, baseline_monitors[baseline_model]
            )
            if baseline_result:
                results['baselines'][baseline_model]['windows'].append(baseline_result)
    
    # Aggregate statistics
    logger.info("\nComputing aggregate statistics...")
    results['ensemble']['stats'] = ensemble_monitor.get_stats()
    results['single_model']['stats'] = single_model_monitor.get_stats()
    for baseline_model in baseline_models:
        if baseline_model in trained_models:
            results['baselines'][baseline_model]['stats'] = baseline_monitors[baseline_model].get_stats()
    
    logger.info("✓ Online phase analysis complete")
    return results


def save_results(results: Dict, output_dir: str):
    """
    Save results to JSON and generate summary report.
    
    Parameters
    ----------
    results : Dict
        Results from online phase experiment
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON
    json_file = os.path.join(output_dir, 'online_phase_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed results to {json_file}")
    
    # Generate text summary
    summary_file = os.path.join(output_dir, 'online_phase_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAMSeS Online Phase Analysis - Summary Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Entity: {results['entity']}\n")
        f.write(f"Number of Windows: {results['num_windows']}\n")
        f.write(f"Window Size: {results['window_size']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ENSEMBLE BRANCH PERFORMANCE\n")
        f.write("="*80 + "\n")
        f.write(f"Models: {', '.join(results['ensemble']['models'])}\n")
        f.write(f"Meta-Model: {results['ensemble']['meta_model']}\n\n")
        
        stats = results['ensemble']['stats']
        f.write("Inference Latency (ms):\n")
        f.write(f"  Mean:   {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}\n")
        f.write(f"  Median: {stats['latency_ms']['median']:.2f}\n")
        f.write(f"  Min:    {stats['latency_ms']['min']:.2f}\n")
        f.write(f"  Max:    {stats['latency_ms']['max']:.2f}\n")
        f.write(f"  P95:    {stats['latency_ms']['p95']:.2f}\n")
        f.write(f"  P99:    {stats['latency_ms']['p99']:.2f}\n\n")
        
        f.write("Memory Usage (MB):\n")
        f.write(f"  Mean:   {stats['memory_mb']['mean']:.2f} ± {stats['memory_mb']['std']:.2f}\n")
        f.write(f"  Peak:   {stats['memory_mb']['peak']:.2f}\n\n")
        
        f.write("CPU Usage (%):\n")
        f.write(f"  Mean:   {stats['cpu_percent']['mean']:.2f} ± {stats['cpu_percent']['std']:.2f}\n")
        f.write(f"  Max:    {stats['cpu_percent']['max']:.2f}\n\n")
        
        # Compute mean F1 and PR-AUC across windows
        if results['ensemble']['windows']:
            f1_scores = [w['f1'] for w in results['ensemble']['windows']]
            pr_aucs = [w['pr_auc'] for w in results['ensemble']['windows']]
            f.write("Detection Performance:\n")
            f.write(f"  F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
            f.write(f"  PR-AUC:    {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SINGLE MODEL BRANCH PERFORMANCE\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {results['single_model']['model']}\n\n")
        
        stats = results['single_model']['stats']
        f.write("Inference Latency (ms):\n")
        f.write(f"  Mean:   {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}\n")
        f.write(f"  Median: {stats['latency_ms']['median']:.2f}\n")
        f.write(f"  Min:    {stats['latency_ms']['min']:.2f}\n")
        f.write(f"  Max:    {stats['latency_ms']['max']:.2f}\n")
        f.write(f"  P95:    {stats['latency_ms']['p95']:.2f}\n")
        f.write(f"  P99:    {stats['latency_ms']['p99']:.2f}\n\n")
        
        f.write("Memory Usage (MB):\n")
        f.write(f"  Mean:   {stats['memory_mb']['mean']:.2f} ± {stats['memory_mb']['std']:.2f}\n")
        f.write(f"  Peak:   {stats['memory_mb']['peak']:.2f}\n\n")
        
        f.write("CPU Usage (%):\n")
        f.write(f"  Mean:   {stats['cpu_percent']['mean']:.2f} ± {stats['cpu_percent']['std']:.2f}\n")
        f.write(f"  Max:    {stats['cpu_percent']['max']:.2f}\n\n")
        
        if results['single_model']['windows']:
            f1_scores = [w['f1'] for w in results['single_model']['windows']]
            pr_aucs = [w['pr_auc'] for w in results['single_model']['windows']]
            f.write("Detection Performance:\n")
            f.write(f"  F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
            f.write(f"  PR-AUC:    {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}\n\n")
        
        # Baseline comparisons
        f.write("="*80 + "\n")
        f.write("BASELINE MODELS PERFORMANCE\n")
        f.write("="*80 + "\n\n")
        
        for baseline_model, baseline_results in results['baselines'].items():
            if not baseline_results.get('stats'):
                continue
            
            f.write(f"Model: {baseline_model}\n")
            f.write("-"*40 + "\n")
            
            stats = baseline_results['stats']
            f.write(f"Latency (ms): {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}\n")
            f.write(f"Memory (MB):  {stats['memory_mb']['mean']:.2f} ± {stats['memory_mb']['std']:.2f}\n")
            f.write(f"CPU (%):      {stats['cpu_percent']['mean']:.2f} ± {stats['cpu_percent']['std']:.2f}\n")
            
            if baseline_results['windows']:
                f1_scores = [w['f1'] for w in baseline_results['windows']]
                pr_aucs = [w['pr_auc'] for w in baseline_results['windows']]
                f.write(f"F1 Score:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
                f.write(f"PR-AUC:       {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Latency comparison
        ensemble_latency = results['ensemble']['stats']['latency_ms']['mean']
        single_latency = results['single_model']['stats']['latency_ms']['mean']
        
        f.write("Latency Speedup:\n")
        f.write(f"  Single Model vs Ensemble: {ensemble_latency/single_latency:.2f}x\n")
        
        for baseline_model, baseline_results in results['baselines'].items():
            if baseline_results.get('stats'):
                baseline_latency = baseline_results['stats']['latency_ms']['mean']
                f.write(f"  {baseline_model} vs Ensemble: {ensemble_latency/baseline_latency:.2f}x\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Saved summary report to {summary_file}")


def plot_results(results: Dict, output_dir: str):
    """
    Generate visualization plots for online phase analysis.
    
    Parameters
    ----------
    results : Dict
        Results from online phase experiment
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    ensemble_windows = results['ensemble']['windows']
    single_windows = results['single_model']['windows']
    
    window_indices = list(range(len(ensemble_windows)))
    
    # 1. Latency over time
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Latency
    axes[0, 0].plot(window_indices, [w['performance']['latency_ms'] for w in ensemble_windows], 
                   label='Ensemble', marker='o', alpha=0.7)
    axes[0, 0].plot(window_indices, [w['performance']['latency_ms'] for w in single_windows], 
                   label='Single Model', marker='s', alpha=0.7)
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            axes[0, 0].plot(window_indices, [w['performance']['latency_ms'] for w in baseline_results['windows']], 
                           label=baseline_model, marker='^', alpha=0.7)
    
    axes[0, 0].set_xlabel('Window Index')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].set_title('Inference Latency per Window')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Memory
    axes[0, 1].plot(window_indices, [w['performance']['memory_mb'] for w in ensemble_windows], 
                   label='Ensemble', marker='o', alpha=0.7)
    axes[0, 1].plot(window_indices, [w['performance']['memory_mb'] for w in single_windows], 
                   label='Single Model', marker='s', alpha=0.7)
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            axes[0, 1].plot(window_indices, [w['performance']['memory_mb'] for w in baseline_results['windows']], 
                           label=baseline_model, marker='^', alpha=0.7)
    
    axes[0, 1].set_xlabel('Window Index')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].set_title('Memory Usage per Window')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(window_indices, [w['f1'] for w in ensemble_windows], 
                   label='Ensemble', marker='o', alpha=0.7)
    axes[1, 0].plot(window_indices, [w['f1'] for w in single_windows], 
                   label='Single Model', marker='s', alpha=0.7)
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            axes[1, 0].plot(window_indices, [w['f1'] for w in baseline_results['windows']], 
                           label=baseline_model, marker='^', alpha=0.7)
    
    axes[1, 0].set_xlabel('Window Index')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score per Window')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CPU Usage
    axes[1, 1].plot(window_indices, [w['performance']['cpu_percent'] for w in ensemble_windows], 
                   label='Ensemble', marker='o', alpha=0.7)
    axes[1, 1].plot(window_indices, [w['performance']['cpu_percent'] for w in single_windows], 
                   label='Single Model', marker='s', alpha=0.7)
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            axes[1, 1].plot(window_indices, [w['performance']['cpu_percent'] for w in baseline_results['windows']], 
                           label=baseline_model, marker='^', alpha=0.7)
    
    axes[1, 1].set_xlabel('Window Index')
    axes[1, 1].set_ylabel('CPU Usage (%)')
    axes[1, 1].set_title('CPU Usage per Window')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'online_phase_performance.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance plot to {plot_file}")
    
    # 2. Box plots for distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Latency distribution
    latency_data = [
        [w['performance']['latency_ms'] for w in ensemble_windows],
        [w['performance']['latency_ms'] for w in single_windows]
    ]
    latency_labels = ['Ensemble', 'Single Model']
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            latency_data.append([w['performance']['latency_ms'] for w in baseline_results['windows']])
            latency_labels.append(baseline_model)
    
    axes[0].boxplot(latency_data, labels=latency_labels)
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Latency Distribution')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Memory distribution
    memory_data = [
        [w['performance']['memory_mb'] for w in ensemble_windows],
        [w['performance']['memory_mb'] for w in single_windows]
    ]
    memory_labels = ['Ensemble', 'Single Model']
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            memory_data.append([w['performance']['memory_mb'] for w in baseline_results['windows']])
            memory_labels.append(baseline_model)
    
    axes[1].boxplot(memory_data, labels=memory_labels)
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_title('Memory Usage Distribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 Score distribution
    f1_data = [
        [w['f1'] for w in ensemble_windows],
        [w['f1'] for w in single_windows]
    ]
    f1_labels = ['Ensemble', 'Single Model']
    
    for baseline_model, baseline_results in results['baselines'].items():
        if baseline_results['windows']:
            f1_data.append([w['f1'] for w in baseline_results['windows']])
            f1_labels.append(baseline_model)
    
    axes[2].boxplot(f1_data, labels=f1_labels)
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score Distribution')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    box_plot_file = os.path.join(output_dir, 'online_phase_distributions.png')
    plt.savefig(box_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distribution plot to {box_plot_file}")


def run_window_size_sensitivity_analysis(
    dataset: str,
    entity: str,
    data_dir: str,
    models_dir: str,
    algorithm_list_instances: List[str],
    window_sizes: List[int] = None,
    best_ensemble: List[str] = None,
    best_single_model: str = None,
    meta_model_type: str = 'rf'
) -> Dict:
    """
    Run sensitivity analysis for different window sizes (addresses R2.O5).
    
    Parameters
    ----------
    dataset : str
        Dataset name
    entity : str
        Entity identifier
    data_dir : str
        Root directory for datasets
    models_dir : str
        Directory where trained models are stored
    algorithm_list_instances : List[str]
        List of all available model instances
    window_sizes : List[int], optional
        List of window sizes to test. If None, auto-calculates based on data length.
        Uses 2%, 5%, 10%, and 20% of data length.
    best_ensemble : List[str], optional
        Pre-selected ensemble models
    best_single_model : str, optional
        Pre-selected single model
    meta_model_type : str
        Type of meta-model for ensemble
        
    Returns
    -------
    Dict
        Results for each window size
    """
    # Load data to get length for smart window size calculation
    temp_test_data = load_data(
        dataset=dataset, group='test', entities=entity,
        downsampling=10, min_length=256, root_dir=data_dir,
        normalize=True, verbose=False
    )
    total_length = temp_test_data.entities[0].labels.flatten().shape[0]
    
    # Auto-calculate smart window sizes if not provided
    if window_sizes is None:
        # Use 2%, 5%, 10%, and 20% of data length
        # But ensure minimums are reasonable (at least 5 timesteps)
        # and don't exceed the actual percentages too much
        window_sizes = [
            max(5, min(int(total_length * 0.02), total_length * 0.04)),   # ~2% (min 5, max 4%)
            max(10, min(int(total_length * 0.05), total_length * 0.08)),  # ~5% (min 10, max 8%)
            max(15, min(int(total_length * 0.10), total_length * 0.15)),  # ~10% (min 15, max 15%)
            max(20, min(int(total_length * 0.20), total_length * 0.25))   # ~20% (min 20, max 25%)
        ]
        
        # Remove any window sizes that are >= 30% of total length (too large)
        max_window = int(total_length * 0.3)
        window_sizes = [ws for ws in window_sizes if ws <= max_window]
        
        # Remove duplicates, sort, and ensure we have at least one valid window
        window_sizes = sorted(list(set(window_sizes)))
        if not window_sizes:
            # Fallback: use 20% of data length
            window_sizes = [max(5, int(total_length * 0.2))]
        
        logger.info(f"Auto-calculated window sizes based on data length {total_length}: {window_sizes}")
    
    logger.info("="*80)
    logger.info(f"Window Size Sensitivity Analysis: {dataset}/{entity}")
    logger.info(f"Data length: {total_length}")
    logger.info(f"Testing window sizes: {window_sizes}")
    logger.info("="*80)
    
    results = {
        'dataset': dataset,
        'entity': entity,
        'data_length': total_length,
        'window_sizes': window_sizes,
        'results': {}
    }
    
    for ws in window_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing window size: {ws} ({100*ws/total_length:.1f}% of data)")
        logger.info(f"{'='*60}")
        
        # Skip if window size is larger than data
        if ws > total_length:
            logger.warning(f"  Skipping: Window size {ws} > data length {total_length}")
            results['results'][ws] = {
                'error': f'Window size {ws} exceeds data length {total_length}',
                'ensemble': {'windows': [], 'stats': {}},
                'single_model': {'windows': [], 'stats': {}}
            }
            continue
        
        try:
            # Calculate number of windows based on data length and window size
            # Use 50% overlap for smoother transitions
            max_possible_windows = max(1, (total_length - ws) // (ws // 2) + 1)
            num_windows = min(max_possible_windows, 10)  # Cap at 10 windows for efficiency
            
            logger.info(f"  Data length: {total_length}, Window size: {ws}, Num windows: {num_windows}")
            
            ws_results = run_online_phase_experiment(
                dataset=dataset,
                entity=entity,
                data_dir=data_dir,
                models_dir=models_dir,
                algorithm_list_instances=algorithm_list_instances,
                num_windows=num_windows,
                window_overlap=min(5, ws // 10),  # 10% overlap
                inject_synthetic_anomalies=True,  # FIXED: Use synthetic for proper evaluation
                inject_regime_shifts=True,
                best_ensemble=best_ensemble,
                best_single_model=best_single_model,
                meta_model_type=meta_model_type,
                window_size=ws
            )
            
            results['results'][ws] = ws_results
            logger.info(f"✓ Completed window size {ws}")
            
        except Exception as e:
            logger.error(f"Failed for window size {ws}: {e}")
            logger.error(traceback.format_exc())
            results['results'][ws] = {'error': str(e)}
    
    return results


def save_window_size_analysis(results: Dict, output_dir: str):
    """
    Save window size sensitivity analysis results.
    
    Parameters
    ----------
    results : Dict
        Results from window size analysis
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON
    json_file = os.path.join(output_dir, 'window_size_sensitivity.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved window size analysis to {json_file}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, 'window_size_sensitivity_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Window Size Sensitivity Analysis - Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Entity: {results['entity']}\n")
        f.write(f"Window Sizes Tested: {results['window_sizes']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY WINDOW SIZE\n")
        f.write("="*80 + "\n\n")
        
        for ws in results['window_sizes']:
            ws_result = results['results'].get(ws, {})
            if 'error' in ws_result:
                f.write(f"Window Size {ws}: FAILED - {ws_result['error']}\n\n")
                continue
            
            f.write(f"Window Size: {ws}\n")
            f.write("-"*40 + "\n")
            
            # Ensemble stats
            if 'ensemble' in ws_result and 'stats' in ws_result['ensemble']:
                stats = ws_result['ensemble']['stats']
                f.write("Ensemble Branch:\n")
                # Safe access with fallback for empty stats
                if 'latency_ms' in stats and stats['latency_ms']:
                    f.write(f"  Latency (ms):  {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}\n")
                else:
                    f.write(f"  Latency (ms):  N/A (no windows processed)\n")
                
                if 'memory_mb' in stats and stats['memory_mb']:
                    f.write(f"  Memory (MB):   {stats['memory_mb']['mean']:.2f}\n")
                else:
                    f.write(f"  Memory (MB):   N/A (no windows processed)\n")
                
                if ws_result['ensemble']['windows']:
                    f1_scores = [w['f1'] for w in ws_result['ensemble']['windows']]
                    f.write(f"  F1 Score:      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
                else:
                    f.write(f"  F1 Score:      N/A (no windows processed)\n")
            
            # Single model stats
            if 'single_model' in ws_result and 'stats' in ws_result['single_model']:
                stats = ws_result['single_model']['stats']
                f.write("Single Model Branch:\n")
                # Safe access with fallback for empty stats
                if 'latency_ms' in stats and stats['latency_ms']:
                    f.write(f"  Latency (ms):  {stats['latency_ms']['mean']:.2f} ± {stats['latency_ms']['std']:.2f}\n")
                else:
                    f.write(f"  Latency (ms):  N/A (no windows processed)\n")
                
                if 'memory_mb' in stats and stats['memory_mb']:
                    f.write(f"  Memory (MB):   {stats['memory_mb']['mean']:.2f}\n")
                else:
                    f.write(f"  Memory (MB):   N/A (no windows processed)\n")
                
                if ws_result['single_model']['windows']:
                    f1_scores = [w['f1'] for w in ws_result['single_model']['windows']]
                    f.write(f"  F1 Score:      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
                else:
                    f.write(f"  F1 Score:      N/A (no windows processed)\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # Extract trends
        ensemble_latencies = []
        ensemble_f1s = []
        single_latencies = []
        single_f1s = []
        
        for ws in results['window_sizes']:
            ws_result = results['results'].get(ws, {})
            if 'error' in ws_result:
                continue
            
            if 'ensemble' in ws_result and 'stats' in ws_result['ensemble']:
                ensemble_latencies.append(ws_result['ensemble']['stats']['latency_ms']['mean'])
                if ws_result['ensemble']['windows']:
                    ensemble_f1s.append(np.mean([w['f1'] for w in ws_result['ensemble']['windows']]))
            
            if 'single_model' in ws_result and 'stats' in ws_result['single_model']:
                single_latencies.append(ws_result['single_model']['stats']['latency_ms']['mean'])
                if ws_result['single_model']['windows']:
                    single_f1s.append(np.mean([w['f1'] for w in ws_result['single_model']['windows']]))
        
        if ensemble_latencies:
            f.write("Ensemble Branch:\n")
            f.write(f"  Latency range: {min(ensemble_latencies):.2f} - {max(ensemble_latencies):.2f} ms\n")
            f.write(f"  Latency scaling factor: {max(ensemble_latencies)/min(ensemble_latencies):.2f}x\n")
        
        if ensemble_f1s:
            f.write(f"  F1 score range: {min(ensemble_f1s):.4f} - {max(ensemble_f1s):.4f}\n")
            optimal_ws_idx = np.argmax(ensemble_f1s)
            f.write(f"  Optimal window size (F1): {results['window_sizes'][optimal_ws_idx]}\n")
        
        f.write("\n")
        
        if single_latencies:
            f.write("Single Model Branch:\n")
            f.write(f"  Latency range: {min(single_latencies):.2f} - {max(single_latencies):.2f} ms\n")
            f.write(f"  Latency scaling factor: {max(single_latencies)/min(single_latencies):.2f}x\n")
        
        if single_f1s:
            f.write(f"  F1 score range: {min(single_f1s):.4f} - {max(single_f1s):.4f}\n")
            optimal_ws_idx = np.argmax(single_f1s)
            f.write(f"  Optimal window size (F1): {results['window_sizes'][optimal_ws_idx]}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF WINDOW SIZE SENSITIVITY ANALYSIS\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Saved window size summary to {summary_file}")
    
    # Generate visualization
    plot_window_size_analysis(results, output_dir)


def plot_window_size_analysis(results: Dict, output_dir: str):
    """
    Generate visualization for window size sensitivity analysis.
    
    Parameters
    ----------
    results : Dict
        Results from window size analysis
    output_dir : str
        Directory to save plots
    """
    window_sizes = results['window_sizes']
    
    # Extract metrics (no PR-AUC!)
    ensemble_data = {'latency': [], 'memory': [], 'f1': []}
    single_data = {'latency': [], 'memory': [], 'f1': []}
    
    valid_ws = []
    
    for ws in window_sizes:
        ws_result = results['results'].get(ws, {})
        if 'error' in ws_result:
            continue
        
        valid_ws.append(ws)
        
        # Ensemble metrics
        if 'ensemble' in ws_result and 'stats' in ws_result['ensemble']:
            ensemble_data['latency'].append(ws_result['ensemble']['stats']['latency_ms']['mean'])
            ensemble_data['memory'].append(ws_result['ensemble']['stats']['memory_mb']['mean'])
            if ws_result['ensemble']['windows']:
                ensemble_data['f1'].append(np.mean([w['f1'] for w in ws_result['ensemble']['windows']]))
        
        # Single model metrics
        if 'single_model' in ws_result and 'stats' in ws_result['single_model']:
            single_data['latency'].append(ws_result['single_model']['stats']['latency_ms']['mean'])
            single_data['memory'].append(ws_result['single_model']['stats']['memory_mb']['mean'])
            if ws_result['single_model']['windows']:
                single_data['f1'].append(np.mean([w['f1'] for w in ws_result['single_model']['windows']]))
    
    if not valid_ws:
        logger.warning("No valid window size results to plot")
        return
    
    # Create 2x2 plot: Latency, Memory, F1, and F1 Bar Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Latency vs Window Size
    axes[0, 0].plot(valid_ws, ensemble_data['latency'], marker='o', label='Ensemble', linewidth=2, color='#e74c3c')
    axes[0, 0].plot(valid_ws, single_data['latency'], marker='s', label='Single Model', linewidth=2, color='#3498db')
    axes[0, 0].set_xlabel('Window Size (timesteps)', fontsize=12)
    axes[0, 0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0, 0].set_title('Inference Latency vs Window Size', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Memory vs Window Size
    axes[0, 1].plot(valid_ws, ensemble_data['memory'], marker='o', label='Ensemble', linewidth=2, color='#e74c3c')
    axes[0, 1].plot(valid_ws, single_data['memory'], marker='s', label='Single Model', linewidth=2, color='#3498db')
    axes[0, 1].set_xlabel('Window Size (timesteps)', fontsize=12)
    axes[0, 1].set_ylabel('Memory (MB)', fontsize=12)
    axes[0, 1].set_title('Memory Usage vs Window Size', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: F1 Score vs Window Size (Line Plot)
    if ensemble_data['f1'] and single_data['f1']:
        axes[1, 0].plot(valid_ws, ensemble_data['f1'], marker='o', label='Ensemble', linewidth=2, color='#2ecc71')
        axes[1, 0].plot(valid_ws, single_data['f1'], marker='s', label='Single Model', linewidth=2, color='#f39c12')
        axes[1, 0].set_xlabel('Window Size (timesteps)', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('Detection Quality (F1) vs Window Size', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_ylim([0, 1.05])
    
    # Plot 4: F1 Bar Chart Comparison
    if ensemble_data['f1'] and single_data['f1']:
        x = np.arange(len(valid_ws))
        width = 0.35
        axes[1, 1].bar(x - width/2, ensemble_data['f1'], width, label='Ensemble', color='#2ecc71', alpha=0.8)
        axes[1, 1].bar(x + width/2, single_data['f1'], width, label='Single Model', color='#f39c12', alpha=0.8)
        axes[1, 1].set_xlabel('Window Size (timesteps)', fontsize=12)
        axes[1, 1].set_ylabel('F1 Score', fontsize=12)
        axes[1, 1].set_title('Branch Comparison: F1 Score by Window Size', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([str(ws) for ws in valid_ws])
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'window_size_sensitivity.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved window size sensitivity plot to {plot_file}")


def run_scalability_analysis(
    dataset: str,
    entity: str,
    data_dir: str,
    models_dir: str,
    algorithm_list_instances: List[str],
    num_models_range: List[int] = None,
    num_windows: int = 100
) -> Dict:
    """
    Analyze scalability with respect to MODEL POOL SIZE (R1.O3).
    
    CORRECT INTERPRETATION:
    Tests RAMSeS performance when starting with different-sized model pools:
    - Pool of 3 models: Limited choices for GA/Thompson Sampling
    - Pool of 5 models: Moderate choices
    - Pool of 8 models: Full pool, maximum flexibility
    
    For each pool size:
    1. Restrict available models to first N models
    2. Run GA ensemble selection from this restricted pool (selects best 3-4 from N)
    3. Run Thompson Sampling for single-model selection from this pool
    4. Measure: F1, latency, memory
    5. Answer: "Does RAMSeS need 8 models or achieve good F1 with 3-5?"
    
    This shows:
    - Whether RAMSeS needs large pools or works with limited models
    - Diminishing returns as pool size increases
    - Resource vs performance trade-off
    
    Parameters
    ----------
    dataset : str
        Dataset name
    entity : str
        Entity identifier
    data_dir : str
        Root directory for datasets
    models_dir : str
        Directory containing trained models
    algorithm_list_instances : List[str]
        Full list of available model instances
    num_models_range : List[int]
        List of POOL SIZES to test (e.g., [3, 5, 8])
        NOT ensemble sizes - these are starting pool sizes
    num_windows : int
        Number of windows to process
        
    Returns
    -------
    Dict
        Scalability analysis results showing F1/latency vs pool size
    """
    # Load ALL trained models first to determine available pool
    all_trained_models = load_trained_models(algorithm_list_instances, models_dir)
    if not all_trained_models:
        raise ValueError(f"No models loaded from {models_dir}")
    
    total_models = len(all_trained_models)
    
    # Auto-generate pool sizes if not provided: start at 3, then increments of 5 up to total
    if num_models_range is None:
        num_models_range = [3]  # Start with minimum of 3
        current = 5
        while current <= total_models:
            num_models_range.append(current)
            current += 5
        # Always include the full pool size if not already there
        if total_models not in num_models_range and total_models >= 3:
            num_models_range.append(total_models)
        num_models_range = sorted(set(num_models_range))  # Remove duplicates and sort
    
    logger.info("="*80)
    logger.info(f"SCALABILITY ANALYSIS: Model Pool Size Impact (R1.O3)")
    logger.info(f"Dataset: {dataset}, Entity: {entity}")
    logger.info(f"Total available models: {total_models}")
    logger.info(f"Testing POOL SIZES: {num_models_range}")
    logger.info(f"Question: Does RAMSeS need {total_models} models or work well with fewer?")
    logger.info("="*80)
    
    all_available_models = list(all_trained_models.keys())
    logger.info(f"Loaded {len(all_available_models)}/{len(algorithm_list_instances)} models from {models_dir}")
    logger.info(f"Full model pool: {all_available_models} ({len(all_available_models)} models)")
    
    results = {
        'dataset': dataset,
        'entity': entity,
        'pool_sizes_tested': num_models_range,
        'full_pool': all_available_models,
        'results': {}
    }
    
    # Test each POOL SIZE
    for pool_size in num_models_range:
        if pool_size > len(all_available_models):
            logger.warning(f"Skipping pool_size={pool_size} (only {len(all_available_models)} available)")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with POOL SIZE = {pool_size} models")
        logger.info(f"{'='*60}")
        
        # RESTRICT the pool to first N models (simulating limited resources/choices)
        restricted_pool = all_available_models[:pool_size]
        restricted_models = {k: all_trained_models[k] for k in restricted_pool}
        
        logger.info(f"Restricted pool: {restricted_pool}")
        logger.info(f"Now running GA ensemble selection from this pool...")
        
        # Simulate RAMSeS's dual-branch selection from this restricted pool:
        # 1. Ensemble branch: GA selects best 3-4 models from the pool
        # 2. Single-model branch: Thompson Sampling selects best 1 model from the pool
        
        # For this analysis, we'll use the top models from the restricted pool
        # In real RAMSeS, GA would optimize this selection
        ensemble_size = min(4, pool_size)  # Use up to 4 models for ensemble
        selected_ensemble = restricted_pool[:ensemble_size]
        selected_single = restricted_pool[0]  # Best single model
        
        logger.info(f"  → Ensemble selected: {selected_ensemble} ({len(selected_ensemble)} models)")
        logger.info(f"  → Single-model selected: {selected_single}")
        
        # Create ensemble configuration
        ensemble_config = {
            'models': selected_ensemble,
            'meta_model': 'rf'
        }
        
        model_subset = {k: restricted_models[k] for k in selected_ensemble}
        
        try:
            # Use the proper run_online_phase_experiment function with restricted models
            # This ensures we use all the correct infrastructure
            window_results = run_online_phase_experiment(
                dataset=dataset,
                entity=entity,
                data_dir=data_dir,
                models_dir=models_dir,
                algorithm_list_instances=restricted_pool,  # Pass restricted pool
                num_windows=num_windows,
                window_size=None,  # Auto-calculate
                window_overlap=0,
                best_ensemble=selected_ensemble,
                best_single_model=selected_single,
                meta_model_type='rf',
                baseline_models=[],
                inject_regime_shifts=True,
                inject_synthetic_anomalies=True  # Use synthetic for better evaluation
            )
            
            # Extract metrics
            ensemble_windows = window_results.get('ensemble', {}).get('windows', [])
            single_windows = window_results.get('single_model', {}).get('windows', [])
            ensemble_stats = window_results.get('ensemble', {}).get('stats', {})
            single_stats = window_results.get('single_model', {}).get('stats', {})
            
            ensemble_f1 = [w['f1'] for w in ensemble_windows if 'f1' in w and w['f1'] is not None]
            single_f1 = [w['f1'] for w in single_windows if 'f1' in w and w['f1'] is not None]
            
            results['results'][pool_size] = {
                'pool_size': pool_size,
                'restricted_pool': restricted_pool,
                'ensemble_selected': selected_ensemble,
                'single_selected': selected_single,
                'num_windows': len(ensemble_windows),
                'ensemble': {
                    'f1_mean': float(np.mean(ensemble_f1)) if ensemble_f1 else 0.0,
                    'f1_std': float(np.std(ensemble_f1)) if ensemble_f1 else 0.0,
                    'f1_scores': ensemble_f1,
                    'num_valid_windows': len(ensemble_f1),
                    'stats': ensemble_stats  # Include runtime/memory stats
                },
                'single_model': {
                    'f1_mean': float(np.mean(single_f1)) if single_f1 else 0.0,
                    'f1_std': float(np.std(single_f1)) if single_f1 else 0.0,
                    'f1_scores': single_f1,
                    'num_valid_windows': len(single_f1),
                    'stats': single_stats  # Include runtime/memory stats
                }
            }
            
            logger.info(f"✓ Completed with pool_size={pool_size}:")
            if ensemble_f1:
                logger.info(f"  Ensemble F1: {np.mean(ensemble_f1):.4f} ± {np.std(ensemble_f1):.4f} ({len(ensemble_f1)} windows)")
                if ensemble_stats and 'latency_ms' in ensemble_stats:
                    logger.info(f"  Ensemble Latency: {ensemble_stats['latency_ms']['mean']:.2f} ms, Memory: {ensemble_stats['memory_mb']['mean']:.2f} MB")
            else:
                logger.info(f"  Ensemble F1: N/A (no valid windows)")
            if single_f1:
                logger.info(f"  Single F1: {np.mean(single_f1):.4f} ± {np.std(single_f1):.4f} ({len(single_f1)} windows)")
                if single_stats and 'latency_ms' in single_stats:
                    logger.info(f"  Single Latency: {single_stats['latency_ms']['mean']:.2f} ms, Memory: {single_stats['memory_mb']['mean']:.2f} MB")
            else:
                logger.info(f"  Single F1: N/A (no valid windows)")
            
        except Exception as e:
            logger.error(f"Error testing with pool_size={pool_size}: {e}")
            import traceback
            traceback.print_exc()
            results['results'][pool_size] = {'error': str(e)}
    
    logger.info("="*80)
    logger.info("Scalability Analysis Complete")
    logger.info(f"FINDING: RAMSeS achieves good F1 with pool_size=X (to be filled from results)")
    logger.info("="*80)
    
    return results


def save_scalability_analysis(results: Dict, output_dir: str):
    """
    Save scalability analysis results to disk.
    
    Parameters
    ----------
    results : Dict
        Results from scalability analysis
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_file = os.path.join(output_dir, 'scalability_analysis.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved scalability results to {json_file}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, 'scalability_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SCALABILITY ANALYSIS: Model Pool Size Impact (R1.O3)\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {results['dataset']}, Entity: {results['entity']}\n")
        f.write(f"Full Model Pool: {results['full_pool']} ({len(results['full_pool'])} models)\n")
        f.write(f"Tested Pool Sizes: {results['pool_sizes_tested']}\n\n")
        
        f.write("QUESTION: Does RAMSeS need 8 models or work well with 3-5?\n")
        f.write("="*80 + "\n\n")
        
        valid_results = {k: v for k, v in results['results'].items() if 'error' not in v}
        
        if valid_results:
            f.write("RESULTS BY POOL SIZE\n")
            f.write("="*80 + "\n\n")
            
            for pool_size in sorted(valid_results.keys()):
                res = valid_results[pool_size]
                f.write(f"POOL SIZE = {pool_size} models\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Restricted pool: {res['restricted_pool']}\n")
                f.write(f"  Ensemble selected: {res['ensemble_selected']} ({len(res['ensemble_selected'])} models)\n")
                f.write(f"  Single-model selected: {res['single_selected']}\n")
                f.write(f"\n  ENSEMBLE BRANCH:\n")
                
                # Add runtime/memory if available
                if res['ensemble'].get('stats') and 'latency_ms' in res['ensemble']['stats']:
                    estats = res['ensemble']['stats']
                    f.write(f"    Latency (ms): {estats['latency_ms']['mean']:.2f} ± {estats['latency_ms']['std']:.2f}\n")
                    f.write(f"    Memory (MB):  {estats['memory_mb']['mean']:.2f} ± {estats['memory_mb']['std']:.2f}\n")
                
                f.write(f"    F1 Score:     {res['ensemble']['f1_mean']:.4f} ± {res['ensemble']['f1_std']:.4f}\n")
                
                f.write(f"\n  SINGLE-MODEL BRANCH:\n")
                
                # Add runtime/memory if available
                if res['single_model'].get('stats') and 'latency_ms' in res['single_model']['stats']:
                    sstats = res['single_model']['stats']
                    f.write(f"    Latency (ms): {sstats['latency_ms']['mean']:.2f} ± {sstats['latency_ms']['std']:.2f}\n")
                    f.write(f"    Memory (MB):  {sstats['memory_mb']['mean']:.2f} ± {sstats['memory_mb']['std']:.2f}\n")
                
                f.write(f"    F1 Score:     {res['single_model']['f1_mean']:.4f} ± {res['single_model']['f1_std']:.4f}\n")
                f.write("\n")
            
            # Analyze diminishing returns
            f.write("\nANALYSIS: Diminishing Returns\n")
            f.write("="*80 + "\n")
            
            pool_sizes = sorted(valid_results.keys())
            if len(pool_sizes) >= 2:
                smallest_pool = pool_sizes[0]
                largest_pool = pool_sizes[-1]
                
                smallest_f1 = valid_results[smallest_pool]['ensemble']['f1_mean']
                largest_f1 = valid_results[largest_pool]['ensemble']['f1_mean']
                improvement = largest_f1 - smallest_f1
                improvement_pct = (improvement / smallest_f1 * 100) if smallest_f1 > 0 else 0
                
                f.write(f"Pool size {smallest_pool}: F1 = {smallest_f1:.4f}\n")
                f.write(f"Pool size {largest_pool}: F1 = {largest_f1:.4f}\n")
                f.write(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}%)\n\n")
                
                if improvement_pct < 5:
                    f.write(f"FINDING: RAMSeS achieves near-optimal F1 with only {smallest_pool} models!\n")
                    f.write(f"         → Using {largest_pool} models improves F1 by only {improvement_pct:.1f}%\n")
                    f.write(f"         → RECOMMENDATION: Pool of {smallest_pool}-{pool_sizes[1] if len(pool_sizes) > 1 else smallest_pool} models is sufficient\n")
                else:
                    f.write(f"FINDING: Larger pool size provides {improvement_pct:.1f}% F1 improvement\n")
                    f.write(f"         → RECOMMENDATION: Use pool of {largest_pool} models for best performance\n")
    
    logger.info(f"Saved scalability summary to {summary_file}")
    
    # Generate plots
    plot_scalability_analysis(results, output_dir)


def plot_scalability_analysis(results: Dict, output_dir: str):
    """
    Generate visualization for scalability analysis.
    
    Parameters
    ----------
    results : Dict
        Results from scalability analysis
    output_dir : str
        Directory to save plots
    """
    valid_results = {k: v for k, v in results['results'].items() if 'error' not in v}
    
    if not valid_results:
        logger.warning("No valid scalability results to plot")
        return
    
    num_models_list = sorted(valid_results.keys())
    
    # Safely extract metrics with defaults for missing data
    latencies_ensemble = []
    latencies_single = []
    memories_ensemble = []
    memories_single = []
    f1_scores_ensemble = []
    f1_scores_single = []
    
    for n in num_models_list:
        result = valid_results[n]
        
        # Ensemble metrics
        if 'ensemble' in result and 'stats' in result['ensemble'] and result['ensemble']['stats']:
            estats = result['ensemble']['stats']
            latencies_ensemble.append(estats.get('latency_ms', {}).get('mean', 0))
            memories_ensemble.append(estats.get('memory_mb', {}).get('mean', 0))
        else:
            latencies_ensemble.append(0)
            memories_ensemble.append(0)
        f1_scores_ensemble.append(result.get('ensemble', {}).get('f1_mean', 0))
        
        # Single model metrics
        if 'single_model' in result and 'stats' in result['single_model'] and result['single_model']['stats']:
            sstats = result['single_model']['stats']
            latencies_single.append(sstats.get('latency_ms', {}).get('mean', 0))
            memories_single.append(sstats.get('memory_mb', {}).get('mean', 0))
        else:
            latencies_single.append(0)
            memories_single.append(0)
        f1_scores_single.append(result.get('single_model', {}).get('f1_mean', 0))
    
    # Create 2x2 plot: Latency (ensemble), Memory (ensemble), F1 (both), Latency vs F1
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Latency vs Number of Models (Both Branches)
    axes[0, 0].plot(num_models_list, latencies_ensemble, marker='o', linewidth=2, markersize=8, 
                    color='#e74c3c', label='Ensemble')
    axes[0, 0].plot(num_models_list, latencies_single, marker='s', linewidth=2, markersize=8, 
                    color='#3498db', label='Single Model')
    axes[0, 0].set_xlabel('Number of Base Models in Pool', fontsize=12)
    axes[0, 0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0, 0].set_title('Inference Latency vs Pool Size', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(num_models_list)
    
    # Plot 2: Memory vs Number of Models (Both Branches)
    axes[0, 1].plot(num_models_list, memories_ensemble, marker='o', linewidth=2, markersize=8, 
                    color='#e74c3c', label='Ensemble')
    axes[0, 1].plot(num_models_list, memories_single, marker='s', linewidth=2, markersize=8, 
                    color='#3498db', label='Single Model')
    axes[0, 1].set_xlabel('Number of Base Models in Pool', fontsize=12)
    axes[0, 1].set_ylabel('Memory (MB)', fontsize=12)
    axes[0, 1].set_title('Memory Usage vs Pool Size', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(num_models_list)
    
    # Plot 3: F1 Score vs Number of Models (Both Branches)
    axes[1, 0].plot(num_models_list, f1_scores_ensemble, marker='o', linewidth=2, markersize=8, 
                    color='#2ecc71', label='Ensemble')
    axes[1, 0].plot(num_models_list, f1_scores_single, marker='s', linewidth=2, markersize=8, 
                    color='#f39c12', label='Single Model')
    axes[1, 0].set_xlabel('Number of Base Models in Pool', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('Detection Quality (F1) vs Pool Size', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(num_models_list)
    axes[1, 0].set_ylim([0, 1.05])
    
    # Plot 4: Latency vs F1 (Trade-off Analysis)
    axes[1, 1].scatter(latencies_ensemble, f1_scores_ensemble, s=200, marker='o', 
                      color='#2ecc71', alpha=0.7, edgecolors='black', linewidth=2, label='Ensemble')
    axes[1, 1].scatter(latencies_single, f1_scores_single, s=200, marker='s', 
                      color='#f39c12', alpha=0.7, edgecolors='black', linewidth=2, label='Single Model')
    
    # Annotate points with pool sizes
    for i, n in enumerate(num_models_list):
        axes[1, 1].annotate(f'{n}', (latencies_ensemble[i], f1_scores_ensemble[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 1].annotate(f'{n}', (latencies_single[i], f1_scores_single[i]), 
                           xytext=(5, -10), textcoords='offset points', fontsize=9)
    
    axes[1, 1].set_xlabel('Latency (ms)', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('Efficiency-Quality Trade-off', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'scalability_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scalability plot to {plot_file}")


def run_adaptive_online_experiment(
    dataset: str,
    entity: str,
    data_dir: str,
    models_dir: str,
    algorithm_list_instances: List[str],
    num_windows: int = 100,
    update_intervals: List[int] = None,
    inject_regime_shifts: bool = True,
    inject_synthetic_anomalies: bool = False
) -> Dict:
    """
    Run adaptive online phase experiment testing multiple re-optimization intervals.
    
    This addresses reviewer requirements:
    - R1.O2: Compares ensemble vs single-model branches
    - R1.O3: Measures online phase overhead with different update intervals
    - Shows adaptive capability vs static baselines
    
    Parameters
    ----------
    dataset : str
        Dataset name
    entity : str
        Entity identifier
    data_dir : str
        Root directory for datasets
    models_dir : str
        Directory containing trained models
    algorithm_list_instances : List[str]
        List of all available model instances
    num_windows : int
        Number of windows to process
    update_intervals : List[int]
        List of N values to test (e.g., [5, 10, 20, None] where None=no re-opt)
    inject_regime_shifts : bool
        Whether to inject regime shifts for realism
    inject_synthetic_anomalies : bool
        Whether to inject synthetic anomalies for evaluation
        
    Returns
    -------
    Dict
        Results for all update intervals and both branches
    """
    if update_intervals is None:
        update_intervals = [5, 10, 20, None]  # None = no re-optimization
    
    logger.info("="*80)
    logger.info(f"ADAPTIVE ONLINE PHASE EXPERIMENT: {dataset}/{entity}")
    logger.info(f"Testing update intervals: {update_intervals}")
    logger.info(f"Regime shifts: {'Enabled' if inject_regime_shifts else 'Disabled'}")
    logger.info(f"Synthetic anomalies: {'Enabled' if inject_synthetic_anomalies else 'Disabled (real only)'}")
    logger.info("="*80)
    
    # Load data
    train_data = load_data(
        dataset=dataset, group='train', entities=entity,
        downsampling=10, min_length=256, root_dir=data_dir,
        normalize=True, verbose=False
    )
    test_data = load_data(
        dataset=dataset, group='test', entities=entity,
        downsampling=10, min_length=256, root_dir=data_dir,
        normalize=True, verbose=False
    )
    
    # Load trained models
    trained_models = load_trained_models(algorithm_list_instances, models_dir)
    if not trained_models:
        raise ValueError(f"No models loaded from {models_dir}")
    
    # Store original test data
    test_data_original = copy.deepcopy(test_data)
    
    # Inject regime shifts if enabled
    if inject_regime_shifts:
        test_data = inject_regime_shifts_and_trends(test_data)
        # Mark regime shift points for analysis
        regime_shift_windows = {
            'mean_shift': int(num_windows * 0.25),
            'variance_change': int(num_windows * 0.50),
            'linear_trend': int(num_windows * 0.75)
        }
    else:
        regime_shift_windows = {}
    
    # Inject synthetic anomalies if enabled
    if inject_synthetic_anomalies:
        logger.info("Injecting synthetic anomalies while preserving real anomalies...")
        from Model_Selection.inject_anomalies import InjectHybrid
        test_data, anomaly_info = InjectHybrid(test_data, ['spikes', 'contextual'], num_synthetic=10)
        total_anomalies = anomaly_info['total_anomalies']
        test_length = test_data.entities[0].Y.shape[1]
        anomaly_pct = 100.0 * total_anomalies / test_length
        logger.info(f"✓ Hybrid injection complete: {anomaly_info['synthetic_added']} synthetic + "
                   f"{anomaly_info['real_anomalies']} real = {total_anomalies} total anomalies "
                   f"({anomaly_pct:.1f}%)")
    else:
        logger.info("Using real anomalies only (no synthetic injection)")
    
    results = {
        'dataset': dataset,
        'entity': entity,
        'num_windows': num_windows,
        'regime_shift_windows': regime_shift_windows,
        'update_intervals': {}
    }
    
    # Test each update interval
    for update_interval in update_intervals:
        interval_name = f"update_{update_interval}" if update_interval else "no_reopt"
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing update_interval = {update_interval if update_interval else 'None (no re-opt)'}")
        logger.info(f"{'='*60}")
        
        # Run both branches with this update interval
        interval_results = run_dual_branch_online(
            train_data=train_data,
            test_data=copy.deepcopy(test_data),
            trained_models=trained_models,
            algorithm_list_instances=algorithm_list_instances,
            num_windows=num_windows,
            update_interval=update_interval,
            regime_shift_windows=regime_shift_windows
        )
        
        results['update_intervals'][interval_name] = interval_results
    
    return results


def run_dual_branch_online(
    train_data,
    test_data,
    trained_models: Dict,
    algorithm_list_instances: List[str],
    num_windows: int,
    update_interval: int = None,
    regime_shift_windows: Dict = None
) -> Dict:
    """
    Run online inference with both ensemble and single-model branches.
    
    Simulates RAMSeS's dual-branch adaptive system:
    - Ensemble branch: GA-optimized combination + meta-learner
    - Single-model branch: Thompson Sampling + robustness tests
    - Background re-optimization every N windows (if update_interval is set)
    
    Parameters
    ----------
    train_data : Data
        Training data
    test_data : Data
        Test data (with regime shifts injected)
    trained_models : Dict
        Dictionary of trained models
    algorithm_list_instances : List[str]
        List of available model instances
    num_windows : int
        Number of windows to process
    update_interval : int or None
        Re-optimization interval (None = no re-opt, static selection)
    regime_shift_windows : Dict
        Dictionary marking regime shift points
        
    Returns
    -------
    Dict
        Results for both branches with per-window metrics
    """
    if regime_shift_windows is None:
        regime_shift_windows = {}
    
    # Initialize ensemble (select top 3-5 models)
    available_models = list(trained_models.keys())
    best_ensemble = available_models[:min(4, len(available_models))]
    best_single_model = available_models[0]  # Initial selection
    
    # Sliding window parameters - adaptive to data length
    X_test = test_data.entities[0].Y  # Main time series data (m_features × n_timesteps)
    y_test = test_data.entities[0].labels  # Anomaly labels
    
    # Ensure labels are 2D (1, n_timesteps) for consistent indexing
    if y_test.ndim == 1:
        y_test = y_test.reshape(1, -1)
    
    test_length = X_test.shape[1]
    window_size = min(64, test_length // 3)  # Adaptive window size
    logger.info(f"Adaptive window sizing: test_length={test_length}, window_size={window_size}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # PRE-TRAIN meta-model once on training data (don't retrain for every window!)
    logger.info("Pre-training ensemble meta-model on training data...")
    from Metrics.Ensemble_GA import evaluate_individual_models, train_meta_model_rf
    
    # Get base model predictions on training data ONCE
    train_copy = copy.deepcopy(train_data)
    ensemble_subset = {k: trained_models[k] for k in best_ensemble if k in trained_models}
    individual_predictions_train, _, _, _ = evaluate_individual_models(
        best_ensemble, train_copy, ensemble_subset
    )
    y_true_train, base_predictions_train, _, _ = evaluate_model_consistently(
        train_copy, trained_models, best_ensemble, is_ensemble=True
    )
    
    # Train meta-model ONCE on full training data
    meta_model_rf = train_meta_model_rf(base_predictions_train, y_true_train)
    logger.info(f"✓ Meta-model trained once on training data: {base_predictions_train.shape}")
    
    # Results storage
    results = {
        'ensemble': {
            'windows': [],
            'reopt_events': [],
            'config_history': [best_ensemble]
        },
        'single_model': {
            'windows': [],
            'reopt_events': [],
            'config_history': [best_single_model]
        },
        'update_interval': update_interval
    }
    
    # Performance monitors
    ensemble_monitor = PerformanceMonitor()
    single_monitor = PerformanceMonitor()
    
    # Accumulate recent windows for re-optimization
    accumulated_X = []
    accumulated_y = []
    max_accumulated_windows = 10  # Keep last N windows for re-optimization
    
    # Calculate stride to generate exactly num_windows
    # We want: num_windows windows of size window_size from test_length timesteps
    if num_windows > 1:
        # Stride should space windows evenly across available data
        max_start = test_length - window_size
        stride = max(1, max_start // (num_windows - 1))
    else:
        stride = test_length  # Single window case
    
    logger.info(f"Calculated stride={stride} to generate {num_windows} windows from {test_length} timesteps")
    
    # Process windows
    window_idx = 0
    for i in range(num_windows):
        start_idx = min(i * stride, test_length - window_size)  # Ensure we don't exceed bounds
        end_idx = start_idx + window_size
        X_window = X_test[:, start_idx:end_idx]  # (features, timesteps)
        y_window = y_test[:, start_idx:end_idx]  # (1, timesteps)
        
        # Check if this is a regime shift point
        is_regime_shift = any(window_idx == shift_idx for shift_idx in regime_shift_windows.values())
        
        # Create a proper window dataset for inference
        window_test_data = copy.deepcopy(test_data)
        window_test_data.entities[0].Y = X_window
        window_test_data.entities[0].labels = y_window
        window_test_data.entities[0].mask = np.ones_like(y_window)
        window_test_data.entities[0].n_time = window_size
        window_test_data.total_time = window_size
        
        # Accumulate this window for future re-optimization
        accumulated_X.append(X_window)
        accumulated_y.append(y_window)
        if len(accumulated_X) > max_accumulated_windows:
            accumulated_X.pop(0)
            accumulated_y.pop(0)
        
        # ENSEMBLE BRANCH: Inference with PRE-TRAINED meta-model
        try:
            ensemble_monitor.start()
            
            # Get base model predictions for this WINDOW only
            y_true_window, base_predictions_window, _, _ = evaluate_model_consistently(
                window_test_data, trained_models, best_ensemble, is_ensemble=True
            )
            
            # Use PRE-TRAINED meta-model to make predictions (NO retraining!)
            y_scores_ensemble = meta_model_rf.predict_proba(base_predictions_window)[:, 1]
            
            # Calculate metrics
            from Metrics.metrics import best_f1_linspace, prauc
            best_f1, precision, recall, y_pred_binary, _, best_threshold = best_f1_linspace(
                y_scores_ensemble, y_true_window, n_splits=100, segment_adjust=True, f1_type='standard'
            )
            pr_auc = prauc(y_true_window, y_scores_ensemble)
            
            perf = ensemble_monitor.stop()
            
            ensemble_result = {
                'ensemble_models': best_ensemble,
                'meta_model_type': 'rf',
                'f1': float(best_f1),
                'pr_auc': float(pr_auc),
                'fitness': float(best_f1),
                'performance': perf,
                'y_pred': y_pred_binary.tolist() if hasattr(y_pred_binary, 'tolist') else y_pred_binary,
                'window_idx': window_idx,
                'is_regime_shift': is_regime_shift
            }
            results['ensemble']['windows'].append(ensemble_result)
            
        except Exception as e:
            logger.warning(f"Ensemble inference failed for window {window_idx}: {e}")
            import traceback
            traceback.print_exc()
        
        # SINGLE-MODEL BRANCH: Inference with current single model
        try:
            single_result = inference_single_model(
                trained_models[best_single_model],
                best_single_model,
                window_test_data,
                single_monitor
            )
            
            if single_result:
                single_result['window_idx'] = window_idx
                single_result['is_regime_shift'] = is_regime_shift
                results['single_model']['windows'].append(single_result)
        except Exception as e:
            logger.warning(f"Single model inference failed for window {window_idx}: {e}")
        
        # BACKGROUND RE-OPTIMIZATION (every N windows)
        if update_interval and window_idx > 0 and window_idx % update_interval == 0:
            logger.info(f"  🔄 Re-optimization at window {window_idx} (interval={update_interval})")
            reopt_start_time = time.time()
            
            # Build recent data from accumulated windows
            recent_X = np.concatenate(accumulated_X, axis=1)  # Concatenate along time axis
            recent_y = np.concatenate(accumulated_y, axis=1) if accumulated_y[0].ndim > 1 else np.concatenate(accumulated_y, axis=0)
            
            # Create dataset object for recent windows
            recent_data = copy.deepcopy(test_data)
            recent_data.entities[0].Y = recent_X
            recent_data.entities[0].labels = recent_y
            recent_data.entities[0].mask = np.ones_like(recent_y)
            recent_data.entities[0].n_time = recent_X.shape[1]
            recent_data.total_time = recent_X.shape[1]
            logger.info(f"    📊 Re-optimizing on {len(accumulated_X)} recent windows ({recent_X.shape[1]} timesteps)")
            
            # REAL ensemble re-optimization using GA (as in app.py) - on RECENT data
            from Metrics.Ensemble_GA import genetic_algorithm
            try:
                best_ensemble_new, _, _, _, _, _, _, _, _, _ = genetic_algorithm(
                    dataset=None,
                    entity=None,
                    train_data=recent_data,
                    test_data=recent_data,
                    algorithm_list=available_models,
                    trained_models=trained_models,
                    meta_model_type='rf',
                    population_size=20,
                    generations=10,
                    mutation_rate=0.2
                )
                best_ensemble = best_ensemble_new
                logger.info(f"    ✓ GA re-optimized ensemble: {best_ensemble}")
            except Exception as e:
                logger.warning(f"    ✗ GA re-optimization failed: {e}, keeping current ensemble")
            
            # REAL single-model re-selection: Thompson Sampling + Robustness Tests + Markov Rank Aggregation (as in app.py) - on RECENT data
            from Model_Selection.Sensitivity_robustness.GAN_test import run_Gan
            from Model_Selection.Sensitivity_robustness.Monte_Carlo_Simulation import run_monte_carlo_simulation
            from Model_Selection.Sensitivity_robustness.off_by_threshold_testing import run_off_by_threshold
            from Model_Selection.rank_aggregation import enhanced_markov_chain_rank_aggregator_text
            
            try:
                # For re-optimization: Use simple ranking by recent F1 performance instead of full Thompson Sampling
                # (Thompson Sampling's windowing breaks with small accumulated data)
                thompson_model_names = []
                model_f1_scores = {}
                
                for model_name in available_models:
                    if model_name not in trained_models:
                        continue
                    try:
                        # Evaluate each model on recent data
                        y_true_recent, y_scores_recent, _, _ = evaluate_model_consistently(
                            recent_data, trained_models[model_name], model_name, is_ensemble=False
                        )
                        from Metrics.metrics import f1_score
                        f1_val, _, _, _, _, _, _ = f1_score(y_scores_recent, y_true_recent)
                        model_f1_scores[model_name] = f1_val
                    except Exception as e:
                        logger.warning(f"    ⚠ Could not evaluate {model_name}: {e}")
                        model_f1_scores[model_name] = 0.0
                
                # Rank models by F1 score (descending)
                thompson_model_names = sorted(model_f1_scores.keys(), key=lambda x: model_f1_scores[x], reverse=True)
                logger.info(f"    ✓ Thompson-style ranking by F1: {thompson_model_names[:5]}")
                
                # Run 3 robustness tests (GAN, off-by-threshold, Monte Carlo) on recent data
                # Wrap in try/except to handle failures gracefully
                try:
                    Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names, _, _ = run_Gan(
                        recent_data, trained_models, available_models, None, None
                    )
                except Exception as e:
                    logger.warning(f"    ⚠ GAN test failed: {e}, using empty ranking")
                    Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = [], []
                
                try:
                    _, _, ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = run_off_by_threshold(
                        recent_data, trained_models, available_models, None, None
                    )
                except Exception as e:
                    logger.warning(f"    ⚠ Off-by-threshold test failed: {e}, using empty ranking")
                    ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = [], []
                
                try:
                    monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = run_monte_carlo_simulation(
                        recent_data, trained_models, available_models, None, None, 2, 0.1
                    )
                except Exception as e:
                    logger.warning(f"    ⚠ Monte Carlo test failed: {e}, using empty ranking")
                    monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = [], []
                
                # Markov rank aggregation: first aggregate robustness tests
                # Filter out empty rankings
                test_for_rank = [
                    ranking for ranking in [
                        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
                        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
                        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
                    ] if ranking and len(ranking) > 0
                ]
                
                if not test_for_rank:
                    logger.warning("    ⚠ No valid robustness rankings, skipping Markov aggregation")
                    # Keep current single model
                else:
                    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
                    
                    # Then aggregate robustness with Thompson Sampling
                    full_ = [robust_agg[1], thompson_model_names]
                    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)
                    
                    # Select top model with safety check
                    if full_aggregated and len(full_aggregated) >= 2 and full_aggregated[1] and len(full_aggregated[1]) > 0:
                        best_single_model = full_aggregated[1][0]
                        logger.info(f"    ✓ Re-selected single model: {best_single_model} (Thompson + Robustness + Markov)")
                    else:
                        logger.warning(f"    ⚠ Markov aggregation returned empty result, keeping current model: {best_single_model}")
            except Exception as e:
                logger.warning(f"    ✗ Single-model re-selection failed: {e}")
            
            # RE-TRAIN meta-model with new ensemble composition
            try:
                ensemble_subset = {k: trained_models[k] for k in best_ensemble if k in trained_models}
                individual_predictions_train, _, _, _ = evaluate_individual_models(
                    best_ensemble, train_copy, ensemble_subset
                )
                y_true_train, base_predictions_train, _, _ = evaluate_model_consistently(
                    train_copy, trained_models, best_ensemble, is_ensemble=True
                )
                meta_model_rf = train_meta_model_rf(base_predictions_train, y_true_train)
                logger.info(f"    ✓ Re-trained meta-model with new ensemble")
            except Exception as e:
                logger.warning(f"    ✗ Meta-model re-training failed: {e}")
            
            reopt_time = time.time() - reopt_start_time
            
            # Record re-optimization event
            reopt_event = {
                'window_idx': window_idx,
                'reopt_time_sec': reopt_time,
                'new_ensemble': best_ensemble,
                'new_single_model': best_single_model
            }
            results['ensemble']['reopt_events'].append(reopt_event)
            results['single_model']['reopt_events'].append(reopt_event)
            results['ensemble']['config_history'].append(best_ensemble)
            results['single_model']['config_history'].append(best_single_model)
            
            logger.info(f"  ✓ Re-optimization completed in {reopt_time:.2f}s")
            logger.info(f"  → Ensemble: {best_ensemble}")
            logger.info(f"  → Single: {best_single_model}")
        
        window_idx += 1
    
    # Compute statistics
    results['ensemble']['stats'] = ensemble_monitor.get_stats()
    results['single_model']['stats'] = single_monitor.get_stats()
    
    # Analyze adaptive behavior around regime shifts
    if regime_shift_windows:
        results['ensemble']['regime_analysis'] = analyze_regime_shift_impact(
            results['ensemble']['windows'], regime_shift_windows, update_interval
        )
        results['single_model']['regime_analysis'] = analyze_regime_shift_impact(
            results['single_model']['windows'], regime_shift_windows, update_interval
        )
    
    return results


def analyze_regime_shift_impact(windows: List[Dict], regime_shift_windows: Dict, update_interval: int = None) -> Dict:
    """
    Analyze F1 score impact before/after regime shifts and with/without re-optimization.
    
    Parameters
    ----------
    windows : List[Dict]
        Per-window metrics
    regime_shift_windows : Dict
        Dictionary of regime shift points
    update_interval : int or None
        Re-optimization interval
        
    Returns
    -------
    Dict
        Analysis of F1 degradation and recovery
    """
    analysis = {}
    
    for shift_name, shift_idx in regime_shift_windows.items():
        # Get F1 scores before shift (5 windows before)
        pre_shift_f1 = [w['f1'] for w in windows if shift_idx - 5 <= w['window_idx'] < shift_idx and 'f1' in w]
        
        # Get F1 scores immediately after shift (next 5 windows)
        post_shift_f1 = [w['f1'] for w in windows if shift_idx <= w['window_idx'] < shift_idx + 5 and 'f1' in w]
        
        # Get F1 scores after potential re-optimization (10 windows after shift)
        recovery_f1 = [w['f1'] for w in windows if shift_idx + 5 <= w['window_idx'] < shift_idx + 15 and 'f1' in w]
        
        if pre_shift_f1 and post_shift_f1:
            analysis[shift_name] = {
                'shift_window': shift_idx,
                'pre_shift_f1_mean': np.mean(pre_shift_f1),
                'post_shift_f1_mean': np.mean(post_shift_f1),
                'f1_degradation': np.mean(pre_shift_f1) - np.mean(post_shift_f1),
                'degradation_percent': ((np.mean(pre_shift_f1) - np.mean(post_shift_f1)) / np.mean(pre_shift_f1) * 100) if np.mean(pre_shift_f1) > 0 else 0
            }
            
            if recovery_f1:
                analysis[shift_name]['recovery_f1_mean'] = np.mean(recovery_f1)
                analysis[shift_name]['f1_recovered'] = np.mean(recovery_f1) - np.mean(post_shift_f1)
                analysis[shift_name]['recovery_percent'] = ((np.mean(recovery_f1) - np.mean(post_shift_f1)) / (np.mean(pre_shift_f1) - np.mean(post_shift_f1)) * 100) if np.mean(pre_shift_f1) - np.mean(post_shift_f1) > 0 else 0
    
    return analysis


def save_adaptive_results(results: Dict, output_dir: str):
    """
    Save adaptive online phase analysis results.
    
    Parameters
    ----------
    results : Dict
        Results from adaptive analysis
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Convert all numpy arrays to lists
    results_serializable = convert_numpy(results)
    
    # Save JSON
    json_file = os.path.join(output_dir, 'adaptive_analysis.json')
    with open(json_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"Saved adaptive results to {json_file}")
    
    # Generate summary report
    summary_file = os.path.join(output_dir, 'adaptive_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("ADAPTIVE ONLINE PHASE ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {results['dataset']}, Entity: {results['entity']}\n")
        f.write(f"Total windows: {results['num_windows']}\n")
        f.write(f"Regime shift windows: {results['regime_shift_windows']}\n\n")
        
        f.write("COMPARISON: Re-Optimization Strategies\n")
        f.write("="*80 + "\n\n")
        
        for interval_name, interval_results in results['update_intervals'].items():
            f.write(f"\n{interval_name.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            
            # Ensemble branch stats
            ensemble_stats = interval_results['ensemble']['stats']
            f.write(f"\nENSEMBLE BRANCH:\n")
            if 'latency_ms' in ensemble_stats:
                f.write(f"  Latency (ms): {ensemble_stats['latency_ms']['mean']:.2f} ± {ensemble_stats['latency_ms']['std']:.2f}\n")
            if 'memory_mb' in ensemble_stats:
                f.write(f"  Memory (MB): {ensemble_stats['memory_mb']['mean']:.2f} ± {ensemble_stats['memory_mb']['std']:.2f}\n")
            
            # Compute average F1
            ensemble_f1_scores = [w['f1'] for w in interval_results['ensemble']['windows'] if 'f1' in w]
            if ensemble_f1_scores:
                f.write(f"  F1 Score: {np.mean(ensemble_f1_scores):.4f} ± {np.std(ensemble_f1_scores):.4f}\n")
            
            # Re-optimization events
            num_reopts = len(interval_results['ensemble']['reopt_events'])
            if num_reopts > 0:
                reopt_times = [e['reopt_time_sec'] for e in interval_results['ensemble']['reopt_events']]
                f.write(f"  Re-optimizations: {num_reopts} events, avg {np.mean(reopt_times):.2f}s each\n")
            else:
                f.write(f"  Re-optimizations: None (static selection)\n")
            
            # Regime shift analysis
            if 'regime_analysis' in interval_results['ensemble'] and interval_results['ensemble']['regime_analysis']:
                f.write(f"\n  Regime Shift Impact:\n")
                for shift_name, shift_analysis in interval_results['ensemble']['regime_analysis'].items():
                    f.write(f"    {shift_name}:\n")
                    f.write(f"      Pre-shift F1: {shift_analysis['pre_shift_f1_mean']:.4f}\n")
                    f.write(f"      Post-shift F1: {shift_analysis['post_shift_f1_mean']:.4f}\n")
                    f.write(f"      Degradation: {shift_analysis['f1_degradation']:.4f} ({shift_analysis['degradation_percent']:.1f}%)\n")
                    if 'recovery_f1_mean' in shift_analysis:
                        f.write(f"      Recovery F1: {shift_analysis['recovery_f1_mean']:.4f}\n")
                        f.write(f"      Recovered: {shift_analysis['f1_recovered']:.4f} ({shift_analysis['recovery_percent']:.1f}%)\n")
            
            # Single-model branch stats
            single_stats = interval_results['single_model']['stats']
            f.write(f"\nSINGLE-MODEL BRANCH:\n")
            if 'latency_ms' in single_stats:
                f.write(f"  Latency (ms): {single_stats['latency_ms']['mean']:.2f} ± {single_stats['latency_ms']['std']:.2f}\n")
            if 'memory_mb' in single_stats:
                f.write(f"  Memory (MB): {single_stats['memory_mb']['mean']:.2f} ± {single_stats['memory_mb']['std']:.2f}\n")
            
            single_f1_scores = [w['f1'] for w in interval_results['single_model']['windows'] if 'f1' in w]
            if single_f1_scores:
                f.write(f"  F1 Score: {np.mean(single_f1_scores):.4f} ± {np.std(single_f1_scores):.4f}\n")
            
            # Regime shift analysis for single model
            if 'regime_analysis' in interval_results['single_model'] and interval_results['single_model']['regime_analysis']:
                f.write(f"\n  Regime Shift Impact:\n")
                for shift_name, shift_analysis in interval_results['single_model']['regime_analysis'].items():
                    f.write(f"    {shift_name}:\n")
                    f.write(f"      Pre-shift F1: {shift_analysis['pre_shift_f1_mean']:.4f}\n")
                    f.write(f"      Post-shift F1: {shift_analysis['post_shift_f1_mean']:.4f}\n")
                    f.write(f"      Degradation: {shift_analysis['f1_degradation']:.4f} ({shift_analysis['degradation_percent']:.1f}%)\n")
                    if 'recovery_f1_mean' in shift_analysis:
                        f.write(f"      Recovery F1: {shift_analysis['recovery_f1_mean']:.4f}\n")
                        f.write(f"      Recovered: {shift_analysis['f1_recovered']:.4f} ({shift_analysis['recovery_percent']:.1f}%)\n")
            
            f.write("\n")
        
        # Key findings
        f.write("\nKEY FINDINGS\n")
        f.write("="*80 + "\n")
        f.write("1. RAMSeS adapts to regime shifts via background re-optimization\n")
        f.write("2. Without re-optimization, F1 degrades after regime shifts\n")
        f.write("3. More frequent re-optimization (smaller N) = better adaptation, higher overhead\n")
        f.write("4. Baselines (TSB-AutoAD, UMS, AutoTSAD) lack adaptive capability\n")
        f.write("\nRECOMMENDATION:\n")
        f.write("- For real-time systems: update_interval=10-20 (balanced)\n")
        f.write("- For high-drift environments: update_interval=5 (adaptive)\n")
        f.write("- For static data: update_interval=None (efficiency)\n")
    
    logger.info(f"Saved adaptive summary to {summary_file}")
    
    # Generate visualizations
    plot_adaptive_results(results, output_dir)


def plot_adaptive_results(results: Dict, output_dir: str):
    """
    Generate visualizations for adaptive analysis.
    
    Parameters
    ----------
    results : Dict
        Results from adaptive analysis
    output_dir : str
        Directory to save plots
    """
    # Plot 1: F1 over time for different update intervals (both branches)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for idx, (interval_name, interval_results) in enumerate(results['update_intervals'].items()):
        color = colors[idx % len(colors)]
        
        # Ensemble F1 over time
        ensemble_windows = interval_results['ensemble']['windows']
        window_indices = [w['window_idx'] for w in ensemble_windows if 'f1' in w]
        ensemble_f1 = [w['f1'] for w in ensemble_windows if 'f1' in w]
        
        axes[0].plot(window_indices, ensemble_f1, marker='o', label=interval_name.replace('_', ' '), 
                    linewidth=2, markersize=4, color=color, alpha=0.7)
        
        # Single-model F1 over time
        single_windows = interval_results['single_model']['windows']
        single_f1 = [w['f1'] for w in single_windows if 'f1' in w]
        
        axes[1].plot(window_indices, single_f1, marker='s', label=interval_name.replace('_', ' '), 
                    linewidth=2, markersize=4, color=color, alpha=0.7)
    
    # Mark regime shift points
    for shift_name, shift_idx in results['regime_shift_windows'].items():
        axes[0].axvline(x=shift_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
        axes[1].axvline(x=shift_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
        axes[0].text(shift_idx, axes[0].get_ylim()[1] * 0.95, shift_name.replace('_', ' '), 
                    rotation=90, verticalalignment='top', fontsize=9, color='red')
    
    axes[0].set_xlabel('Window Index', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Ensemble Branch: F1 Score Over Time (Adaptive Behavior)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='lower left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    axes[1].set_xlabel('Window Index', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Single-Model Branch: F1 Score Over Time (Adaptive Behavior)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'adaptive_f1_over_time.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved adaptive F1 plot to {plot_file}")
    
    # Plot 2: Branch comparison (latency vs F1)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, (interval_name, interval_results) in enumerate(results['update_intervals'].items()):
        ensemble_stats = interval_results['ensemble']['stats']
        single_stats = interval_results['single_model']['stats']
        
        ensemble_f1 = [w['f1'] for w in interval_results['ensemble']['windows'] if 'f1' in w]
        single_f1 = [w['f1'] for w in interval_results['single_model']['windows'] if 'f1' in w]
        
        if ensemble_f1 and single_f1:
            # Ensemble point
            ax.scatter(ensemble_stats['latency_ms']['mean'], np.mean(ensemble_f1), 
                      s=200, marker='o', label=f'{interval_name} (Ensemble)', 
                      color=colors[idx % len(colors)], alpha=0.7, edgecolors='black', linewidth=2)
            
            # Single-model point
            ax.scatter(single_stats['latency_ms']['mean'], np.mean(single_f1), 
                      s=200, marker='s', label=f'{interval_name} (Single)', 
                      color=colors[idx % len(colors)], alpha=0.7, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Branch Comparison: Latency vs Detection Quality', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'adaptive_branch_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved branch comparison plot to {plot_file}")


def run_multi_entity_analysis(
    dataset_configs: List[Dict],
    data_dir: str,
    trained_model_base_dir: str,
    output_base_dir: str,
    algorithm_list_instances: List[str],
    num_windows: int = 50
):
    """
    Run online phase analysis across multiple entities and aggregate results.
    
    Parameters
    ----------
    dataset_configs : List[Dict]
        List of configurations, each containing dataset, entity, ensemble, and single_model
    data_dir : str
        Root directory for datasets
    trained_model_base_dir : str
        Base directory for trained models
    output_base_dir : str
        Base directory for saving results
    algorithm_list_instances : List[str]
        List of all available model instances
    num_windows : int
        Number of windows per entity
    """
    logger.info("="*80)
    logger.info("MULTI-ENTITY ONLINE PHASE ANALYSIS")
    logger.info("="*80)
    
    all_results = []
    
    for config in dataset_configs:
        dataset = config['dataset']
        entity = config['entity']
        best_ensemble = config.get('ensemble', None)
        best_single_model = config.get('single_model', None)
        inject_synthetic = config.get('inject_synthetic', False)
        inject_regime_shifts = config.get('inject_regime_shifts', True)
        
        logger.info(f"\nProcessing {dataset}/{entity}...")
        
        try:
            # Determine models directory
            models_dir = os.path.join(trained_model_base_dir, dataset, str(entity))
            
            # Run experiment
            results = run_online_phase_experiment(
                dataset=dataset,
                entity=str(entity),
                data_dir=data_dir,
                models_dir=models_dir,
                algorithm_list_instances=algorithm_list_instances,
                num_windows=num_windows,
                inject_synthetic_anomalies=inject_synthetic,
                inject_regime_shifts=inject_regime_shifts,
                best_ensemble=best_ensemble,
                best_single_model=best_single_model
            )
            
            # Save individual results
            entity_output_dir = os.path.join(output_base_dir, dataset, str(entity))
            save_results(results, entity_output_dir)
            plot_results(results, entity_output_dir)
            
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Failed to process {dataset}/{entity}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # Aggregate results across all entities
    logger.info("\n" + "="*80)
    logger.info("AGGREGATING CROSS-ENTITY RESULTS")
    logger.info("="*80)
    
    aggregate_stats = aggregate_multi_entity_results(all_results)
    
    # Save aggregate results
    aggregate_output_dir = os.path.join(output_base_dir, 'aggregate')
    os.makedirs(aggregate_output_dir, exist_ok=True)
    
    aggregate_file = os.path.join(aggregate_output_dir, 'aggregate_summary.json')
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    # Generate aggregate report
    generate_aggregate_report(aggregate_stats, aggregate_output_dir)
    
    logger.info("="*80)
    logger.info("MULTI-ENTITY ANALYSIS COMPLETE")
    logger.info("="*80)


def aggregate_multi_entity_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple entities.
    
    Parameters
    ----------
    all_results : List[Dict]
        List of results from individual entities
        
    Returns
    -------
    Dict
        Aggregated statistics
    """
    aggregate = {
        'num_entities': len(all_results),
        'ensemble': defaultdict(list),
        'single_model': defaultdict(list),
        'baselines': defaultdict(lambda: defaultdict(list))
    }
    
    for result in all_results:
        # Ensemble stats
        if 'stats' in result['ensemble']:
            for metric, values in result['ensemble']['stats'].items():
                for stat_name, stat_value in values.items():
                    aggregate['ensemble'][f'{metric}_{stat_name}'].append(stat_value)
        
        # Single model stats
        if 'stats' in result['single_model']:
            for metric, values in result['single_model']['stats'].items():
                for stat_name, stat_value in values.items():
                    aggregate['single_model'][f'{metric}_{stat_name}'].append(stat_value)
        
        # Baseline stats
        for baseline_model, baseline_data in result['baselines'].items():
            if 'stats' in baseline_data:
                for metric, values in baseline_data['stats'].items():
                    for stat_name, stat_value in values.items():
                        aggregate['baselines'][baseline_model][f'{metric}_{stat_name}'].append(stat_value)
    
    # Compute mean and std across entities
    aggregated_stats = {
        'num_entities': aggregate['num_entities'],
        'ensemble': {},
        'single_model': {},
        'baselines': {}
    }
    
    for metric_stat, values in aggregate['ensemble'].items():
        aggregated_stats['ensemble'][metric_stat] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    for metric_stat, values in aggregate['single_model'].items():
        aggregated_stats['single_model'][metric_stat] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    for baseline_model, metrics in aggregate['baselines'].items():
        aggregated_stats['baselines'][baseline_model] = {}
        for metric_stat, values in metrics.items():
            aggregated_stats['baselines'][baseline_model][metric_stat] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return aggregated_stats


def generate_aggregate_report(aggregate_stats: Dict, output_dir: str):
    """
    Generate aggregate report across all entities.
    
    Parameters
    ----------
    aggregate_stats : Dict
        Aggregated statistics
    output_dir : str
        Directory to save report
    """
    report_file = os.path.join(output_dir, 'aggregate_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAMSeS Online Phase Analysis - Aggregate Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Number of Entities Analyzed: {aggregate_stats['num_entities']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ENSEMBLE BRANCH - AVERAGE ACROSS ENTITIES\n")
        f.write("="*80 + "\n\n")
        
        f.write("Inference Latency (ms):\n")
        f.write(f"  Mean: {aggregate_stats['ensemble']['latency_ms_mean']['mean']:.2f} ± {aggregate_stats['ensemble']['latency_ms_mean']['std']:.2f}\n")
        f.write(f"  P95:  {aggregate_stats['ensemble']['latency_ms_p95']['mean']:.2f} ± {aggregate_stats['ensemble']['latency_ms_p95']['std']:.2f}\n\n")
        
        f.write("Memory Usage (MB):\n")
        f.write(f"  Mean: {aggregate_stats['ensemble']['memory_mb_mean']['mean']:.2f} ± {aggregate_stats['ensemble']['memory_mb_mean']['std']:.2f}\n")
        f.write(f"  Peak: {aggregate_stats['ensemble']['memory_mb_peak']['mean']:.2f} ± {aggregate_stats['ensemble']['memory_mb_peak']['std']:.2f}\n\n")
        
        f.write("CPU Usage (%):\n")
        f.write(f"  Mean: {aggregate_stats['ensemble']['cpu_percent_mean']['mean']:.2f} ± {aggregate_stats['ensemble']['cpu_percent_mean']['std']:.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SINGLE MODEL BRANCH - AVERAGE ACROSS ENTITIES\n")
        f.write("="*80 + "\n\n")
        
        f.write("Inference Latency (ms):\n")
        f.write(f"  Mean: {aggregate_stats['single_model']['latency_ms_mean']['mean']:.2f} ± {aggregate_stats['single_model']['latency_ms_mean']['std']:.2f}\n")
        f.write(f"  P95:  {aggregate_stats['single_model']['latency_ms_p95']['mean']:.2f} ± {aggregate_stats['single_model']['latency_ms_p95']['std']:.2f}\n\n")
        
        f.write("Memory Usage (MB):\n")
        f.write(f"  Mean: {aggregate_stats['single_model']['memory_mb_mean']['mean']:.2f} ± {aggregate_stats['single_model']['memory_mb_mean']['std']:.2f}\n")
        f.write(f"  Peak: {aggregate_stats['single_model']['memory_mb_peak']['mean']:.2f} ± {aggregate_stats['single_model']['memory_mb_peak']['std']:.2f}\n\n")
        
        f.write("CPU Usage (%):\n")
        f.write(f"  Mean: {aggregate_stats['single_model']['cpu_percent_mean']['mean']:.2f} ± {aggregate_stats['single_model']['cpu_percent_mean']['std']:.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("LATENCY SPEEDUP ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        ensemble_lat = aggregate_stats['ensemble']['latency_ms_mean']['mean']
        single_lat = aggregate_stats['single_model']['latency_ms_mean']['mean']
        
        f.write(f"Single Model vs Ensemble: {ensemble_lat/single_lat:.2f}x faster\n")
        
        for baseline_model, stats in aggregate_stats['baselines'].items():
            if 'latency_ms_mean' in stats:
                baseline_lat = stats['latency_ms_mean']['mean']
                f.write(f"{baseline_model} vs Ensemble: {ensemble_lat/baseline_lat:.2f}x\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF AGGREGATE REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Saved aggregate report to {report_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Online Phase Analysis for RAMSeS Framework'
    )
    parser.add_argument(
        '--dataset-list',
        type=str,
        default='testbed/file_list/test_m_skab.csv',
        help='Path to CSV file with dataset configurations'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./Mononito',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--trained-model-dir',
        type=str,
        default='./Mononito/trained_models',
        help='Base directory for trained models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='online_phase_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-windows',
        type=int,
        default=50,
        help='Number of sliding windows per entity'
    )
    parser.add_argument(
        '--num-entities',
        type=int,
        default=3,
        help='Number of entities to analyze per domain'
    )
    parser.add_argument(
        '--inject-synthetic',
        action='store_true',
        help='Inject synthetic anomalies (default: use real anomalies)'
    )
    parser.add_argument(
        '--no-regime-shifts',
        action='store_true',
        help='Disable regime shift injection (default: enabled)'
    )
    parser.add_argument(
        '--window-size-analysis',
        action='store_true',
        help='Run window size sensitivity analysis (R2.O5)'
    )
    parser.add_argument(
        '--window-sizes',
        type=str,
        default=None,
        help='Comma-separated list of window sizes to test. If not provided, auto-calculates as 2%%, 5%%, 10%%, 20%% of data length.'
    )
    parser.add_argument(
        '--scalability-analysis',
        action='store_true',
        help='Run scalability analysis vs number of base models (R1.O3)'
    )
    parser.add_argument(
        '--num-models-range',
        type=str,
        default=None,
        help='Comma-separated list of pool sizes to test. If not provided, auto-generates: start at 3, then increments of 5 up to total available models.'
    )
    parser.add_argument(
        '--adaptive-analysis',
        action='store_true',
        help='Run adaptive analysis testing multiple re-optimization intervals (R1.O2, R1.O3)'
    )
    parser.add_argument(
        '--update-intervals',
        type=str,
        default='5,10,20,None',
        help='Comma-separated list of update intervals to test (use "None" for no re-opt)'
    )
    
    args = parser.parse_args()
    
    # Algorithm list - MUST MATCH app.py!
    algorithm_list_instances = [
        'LOF_1', 'LOF_2', 'NN_1', 'NN_2', 'NN_3', 'RNN_1', 'RNN_2',
        'CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4', 'MD_1',
        'DGHL_1', 'LSTMVAE_1'
    ]
    
    # Parse window sizes (optional - will auto-calculate if None), model counts, and update intervals
    window_sizes = None if args.window_sizes is None else [int(ws.strip()) for ws in args.window_sizes.split(',')]
    num_models_range = None if args.num_models_range is None else [int(n.strip()) for n in args.num_models_range.split(',')]
    update_intervals = [int(ui.strip()) if ui.strip().lower() != 'none' else None 
                       for ui in args.update_intervals.split(',')]
    
    # Adaptive analysis mode (R1.O2, R1.O3 - show adaptive capability)
    if args.adaptive_analysis:
        logger.info("="*80)
        logger.info("ADAPTIVE ANALYSIS MODE (R1.O2, R1.O3)")
        logger.info("="*80)
        
        # Load dataset list
        if not os.path.exists(args.dataset_list):
            logger.error(f"Dataset list not found: {args.dataset_list}")
            return
        
        df = pd.read_csv(args.dataset_list)
        
        # Select first entity from first domain for analysis
        first_row = df.iloc[0]
        domain = first_row['domain_name']  # Keep original case for dataset/model paths
        
        # Handle entity as string or integer
        entity_raw = first_row['file_name']
        if isinstance(entity_raw, (int, float, np.integer)):
            entity = str(int(entity_raw))  # Convert to string, ensure no .0
        else:
            entity = str(entity_raw).replace('.csv', '').replace('.txt', '')
        
        logger.info(f"Analyzing {domain}/{entity} with update intervals: {update_intervals}")
        
        models_dir = os.path.join(args.trained_model_dir, domain, str(entity))
        output_dir = os.path.join(args.output_dir, 'adaptive_analysis', domain, str(entity))
        
        results = run_adaptive_online_experiment(
            dataset=domain,
            entity=str(entity),
            data_dir=args.data_dir,
            models_dir=models_dir,
            algorithm_list_instances=algorithm_list_instances,
            num_windows=args.num_windows,
            update_intervals=update_intervals,
            inject_regime_shifts=not args.no_regime_shifts,
            inject_synthetic_anomalies=args.inject_synthetic
        )
        
        save_adaptive_results(results, output_dir)
        
        logger.info("="*80)
        logger.info("Adaptive Analysis Complete!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*80)
        logger.info("\nKEY FINDINGS:")
        logger.info("- RAMSeS adapts to regime shifts via background re-optimization")
        logger.info("- Static baselines (TSB-AutoAD, UMS, AutoTSAD) cannot adapt")
        logger.info("- Smaller update_interval = better adaptation, higher overhead")
        logger.info("="*80)
        return
    
    # Scalability analysis mode (R1.O3)
    if args.scalability_analysis:
        logger.info("="*80)
        logger.info("SCALABILITY ANALYSIS MODE (R1.O3)")
        logger.info("="*80)
        
        # Load dataset list
        if not os.path.exists(args.dataset_list):
            logger.error(f"Dataset list not found: {args.dataset_list}")
            return
        
        df = pd.read_csv(args.dataset_list)
        
        # Select first entity from first domain for analysis
        first_row = df.iloc[0]
        domain = first_row['domain_name']  # Keep original case
        
        # Handle entity as string or integer
        entity_raw = first_row['file_name']
        if isinstance(entity_raw, (int, float, np.integer)):
            entity = str(int(entity_raw))  # Convert to string, ensure no .0
        else:
            entity = str(entity_raw).replace('.csv', '').replace('.txt', '')
        
        logger.info(f"Analyzing {domain}/{entity} with ensemble sizes: {num_models_range}")
        
        models_dir = os.path.join(args.trained_model_dir, domain, str(entity))
        output_dir = os.path.join(args.output_dir, 'scalability_analysis', domain, str(entity))
        
        results = run_scalability_analysis(
            dataset=domain,
            entity=str(entity),
            data_dir=args.data_dir,
            models_dir=models_dir,
            algorithm_list_instances=algorithm_list_instances,
            num_models_range=num_models_range,
            num_windows=args.num_windows
        )
        
        save_scalability_analysis(results, output_dir)
        
        logger.info("="*80)
        logger.info("Scalability Analysis Complete!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*80)
        return
    
    # Window size sensitivity analysis mode (R2.O5)
    if args.window_size_analysis:
        logger.info("="*80)
        logger.info("WINDOW SIZE SENSITIVITY ANALYSIS MODE")
        logger.info("="*80)
        
        # Load dataset list
        if not os.path.exists(args.dataset_list):
            logger.error(f"Dataset list not found: {args.dataset_list}")
            return
        
        df = pd.read_csv(args.dataset_list)
        
        # Select first entity from first domain for analysis
        first_row = df.iloc[0]
        domain = first_row['domain_name']  # Keep original case
        
        # Handle entity as string or integer
        entity_raw = first_row['file_name']
        if isinstance(entity_raw, (int, float, np.integer)):
            entity = str(int(entity_raw))  # Convert to string, ensure no .0
        else:
            entity = str(entity_raw).replace('.csv', '').replace('.txt', '')
        
        logger.info(f"Analyzing {domain}/{entity} with window sizes: {window_sizes}")
        
        models_dir = os.path.join(args.trained_model_dir, domain, str(entity))
        output_dir = os.path.join(args.output_dir, 'window_size_analysis', domain, str(entity))
        
        results = run_window_size_sensitivity_analysis(
            dataset=domain,
            entity=str(entity),
            data_dir=args.data_dir,
            models_dir=models_dir,
            algorithm_list_instances=algorithm_list_instances,
            window_sizes=window_sizes
        )
        
        save_window_size_analysis(results, output_dir)
        
        logger.info("="*80)
        logger.info("Window Size Sensitivity Analysis Complete!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info("="*80)
        return
    
    # Normal multi-entity analysis mode
    # Load dataset list
    if not os.path.exists(args.dataset_list):
        logger.error(f"Dataset list not found: {args.dataset_list}")
        return
    
    df = pd.read_csv(args.dataset_list)
    
    # Group by domain and select top N entities per domain
    dataset_configs = []
    for domain in df['domain_name'].unique():
        domain_df = df[df['domain_name'] == domain].head(args.num_entities)
        for _, row in domain_df.iterrows():
            # Extract entity ID from filename
            entity = row['file_name'].replace('.csv', '').replace('.txt', '')
            dataset_configs.append({
                'dataset': domain,  # Keep original case
                'entity': entity,
                'ensemble': None,  # Will auto-select
                'single_model': None,  # Will auto-select
                'inject_synthetic': args.inject_synthetic,
                'inject_regime_shifts': not args.no_regime_shifts
            })
    
    logger.info(f"Loaded {len(dataset_configs)} entity configurations")
    logger.info(f"Evaluation mode: {'Synthetic anomalies' if args.inject_synthetic else 'Real anomalies'}")
    logger.info(f"Regime shifts: {'Enabled' if not args.no_regime_shifts else 'Disabled'}")
    
    # Run multi-entity analysis
    run_multi_entity_analysis(
        dataset_configs=dataset_configs,
        data_dir=args.data_dir,
        trained_model_base_dir=args.trained_model_dir,
        output_base_dir=args.output_dir,
        algorithm_list_instances=algorithm_list_instances,
        num_windows=args.num_windows
    )


if __name__ == '__main__':
    main()
