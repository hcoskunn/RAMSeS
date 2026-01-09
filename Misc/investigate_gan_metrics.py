#!/usr/bin/env python3
"""
Script to investigate the GAN test metric anomaly (F1=0.99999 vs PR-AUC=0.5)
"""

import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import matplotlib.pyplot as plt

def analyze_range_based_metrics_bug():
    """
    Investigate the range-based F1 calculation bug that causes F1=0.99999
    """
    
    print("="*80)
    print("INVESTIGATING GAN METRICS BUG")
    print("="*80)
    
    # Simulate a typical scenario: mostly normal data with some anomalies
    n_samples = 1000
    y_true = np.zeros(n_samples)
    # Add some anomalies (10% anomaly rate)
    anomaly_indices = np.random.choice(n_samples, size=100, replace=False)
    y_true[anomaly_indices] = 1
    
    # Simulate random predictions (like a bad model)
    y_scores = np.random.rand(n_samples)
    
    print(f"\nDataset characteristics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Anomalies: {np.sum(y_true)} ({100*np.sum(y_true)/n_samples:.1f}%)")
    print(f"  Normal: {np.sum(y_true==0)} ({100*np.sum(y_true==0)/n_samples:.1f}%)")
    
    # Test the range-based calculation
    print("\n" + "="*80)
    print("RANGE-BASED METRICS CALCULATION")
    print("="*80)
    
    # Use window_size from the GAN test (default is 1000)
    window_size = 1000
    n_splits = 1000
    
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    range_precision = np.zeros(thresholds.shape)
    range_recall = np.zeros(thresholds.shape)
    range_f1 = np.zeros(thresholds.shape)
    
    for i, threshold in enumerate(thresholds[:10]):  # Just test first 10 thresholds
        y_pred = y_scores >= threshold
        
        # Calculate range-based metrics
        for idx in range(len(y_pred)):
            start_idx = max(0, idx - window_size)
            end_idx = min(len(y_pred), idx + window_size + 1)
            
            TP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 1))
            FP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 0))
            FN = np.sum((y_pred[start_idx:end_idx] == 0) & (y_true[start_idx:end_idx] == 1))
            
            precision = TP / (TP + FP + 0.00001)
            recall = TP / (TP + FN + 0.00001)
            f1 = 2 * precision * recall / (precision + recall + 0.00001)
            
            range_precision[i] += precision
            range_recall[i] += recall
            range_f1[i] += f1
        
        range_precision[i] /= len(y_pred)
        range_recall[i] /= len(y_pred)
        range_f1[i] /= len(y_pred)
        
        if i < 3:
            print(f"\nThreshold {threshold:.4f}:")
            print(f"  Avg Precision: {range_precision[i]:.6f}")
            print(f"  Avg Recall: {range_recall[i]:.6f}")
            print(f"  Avg F1: {range_f1[i]:.6f}")
    
    # The bug: when window_size >= n_samples, every point sees ALL data
    print("\n" + "="*80)
    print("BUG ANALYSIS")
    print("="*80)
    
    print(f"\nWindow size: {window_size}")
    print(f"Dataset size: {n_samples}")
    print(f"Window covers: {2*window_size + 1} points")
    
    if window_size >= n_samples:
        print("\n⚠️  BUG DETECTED: Window size >= dataset size!")
        print("   Every point's metrics are calculated over the ENTIRE dataset!")
        print("   This causes averaging to produce near-constant metrics.")
    else:
        print("\n✓ Window size is appropriate")
    
    # Calculate what happens with full dataset visibility
    print("\n" + "="*80)
    print("SIMULATING FULL-DATASET-VISIBILITY BUG")
    print("="*80)
    
    # Use threshold that gives ~50% positive predictions
    threshold = np.median(y_scores)
    y_pred = y_scores >= threshold
    
    # With full dataset visibility, every point gets the same metrics
    TP_total = np.sum((y_pred == 1) & (y_true == 1))
    FP_total = np.sum((y_pred == 1) & (y_true == 0))
    FN_total = np.sum((y_pred == 0) & (y_true == 1))
    TN_total = np.sum((y_pred == 0) & (y_true == 0))
    
    precision_global = TP_total / (TP_total + FP_total + 0.00001)
    recall_global = TP_total / (TP_total + FN_total + 0.00001)
    f1_global = 2 * precision_global * recall_global / (precision_global + recall_global + 0.00001)
    
    print(f"\nGlobal metrics (what every point sees):")
    print(f"  TP: {TP_total}, FP: {FP_total}, FN: {FN_total}, TN: {TN_total}")
    print(f"  Precision: {precision_global:.6f}")
    print(f"  Recall: {recall_global:.6f}")
    print(f"  F1: {f1_global:.6f}")
    
    print(f"\nAfter averaging across all {n_samples} points:")
    print(f"  Avg F1 = {f1_global:.6f} (same as global)")
    
    # Now test PR-AUC calculation
    print("\n" + "="*80)
    print("PR-AUC CALCULATION")
    print("="*80)
    
    # The PR-AUC is calculated from the precision-recall curve
    # When all points have same precision/recall, curve collapses to a point
    
    # Simulate what happens with constant precision/recall across thresholds
    constant_precisions = np.ones(100) * precision_global
    constant_recalls = np.ones(100) * recall_global
    
    # AUC of a constant line
    prauc_constant = auc(constant_recalls, constant_precisions)
    
    print(f"\nWith constant precision={precision_global:.4f} and recall={recall_global:.4f}:")
    print(f"  PR-AUC = {prauc_constant:.6f}")
    print(f"  (This is just precision × recall_range, not meaningful)")
    
    # Compare with proper PR-AUC
    precision_proper, recall_proper, _ = precision_recall_curve(y_true, y_scores)
    prauc_proper = auc(recall_proper, precision_proper)
    
    print(f"\nProper PR-AUC (sklearn): {prauc_proper:.6f}")
    
    # Explain the 0.99999 F1 mystery
    print("\n" + "="*80)
    print("EXPLAINING F1 = 0.99999")
    print("="*80)
    
    print("\nThe F1=0.99999 occurs when:")
    print("  1. Window size >= dataset size (every point sees all data)")
    print("  2. Threshold is chosen to maximize F1 (with epsilon=0.00001)")
    print("  3. At optimal threshold, TP/(TP+FP+ε) and TP/(TP+FN+ε) are high")
    print("  4. But with epsilon=0.00001, even perfect classification gives:")
    print(f"     F1 = 2*1*1/(1+1+0.00001) ≈ 1.999999/2.000001 ≈ 0.999999")
    
    # Verify this
    perfect_tp = 100
    perfect_fp = 0
    perfect_fn = 0
    perfect_precision = perfect_tp / (perfect_tp + perfect_fp + 0.00001)
    perfect_recall = perfect_tp / (perfect_tp + perfect_fn + 0.00001)
    perfect_f1 = 2 * perfect_precision * perfect_recall / (perfect_precision + perfect_recall + 0.00001)
    
    print(f"\nWith perfect classification (TP={perfect_tp}, FP=0, FN=0):")
    print(f"  Precision = {perfect_tp}/(100+0+0.00001) = {perfect_precision:.10f}")
    print(f"  Recall    = {perfect_tp}/(100+0+0.00001) = {perfect_recall:.10f}")
    print(f"  F1        = {perfect_f1:.10f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\n1. The range-based metric calculation has a CRITICAL BUG:")
    print("   - Default window_size=1000 is used regardless of dataset size")
    print("   - For small datasets (<2000 points), window covers entire dataset")
    print("   - This makes every point's metrics identical (global averages)")
    print("\n2. The F1=0.99999 is an artifact of:")
    print("   - Epsilon regularization (0.00001) in division")
    print("   - Optimal threshold selection maximizing this metric")
    print("   - Not a real measure of model performance!")
    print("\n3. The PR-AUC≈0.5 is CORRECT:")
    print("   - It uses proper sklearn calculation")
    print("   - Shows the model is actually random (like coin flip)")
    print("\n4. RECOMMENDATION:")
    print("   - Fix window_size to be adaptive: min(1000, dataset_size // 10)")
    print("   - Or use point-based metrics instead of range-based for GAN test")
    print("   - The current GAN test results are MEANINGLESS")


if __name__ == "__main__":
    analyze_range_based_metrics_bug()
