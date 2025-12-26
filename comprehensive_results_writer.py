# Helper module for writing comprehensive results
import time
from datetime import datetime

def write_comprehensive_results(output_file, dataset, entity, iteration, 
                                results_dict, timing_dict):
    """Write comprehensive results including timing, scores, and chosen models."""
    
    with open(output_file, "w") as f:
        # Header
        f.write("="*80 + "\n")
        f.write("RAMSeS Framework - Comprehensive Results\n")
        f.write(f"Dataset: {dataset} | Entity: {entity} | Iteration: {iteration}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Computational Overhead
        f.write("="*80 + "\n")
        f.write("COMPUTATIONAL OVERHEAD (seconds)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Per-Module Timing:\n")
        f.write("-" * 50 + "\n")
        total_module_time = 0
        for module_name, module_time in timing_dict.items():
            if module_name != "end_to_end":
                f.write(f"  {module_name:30s} : {module_time:9.4f}s\n")
                total_module_time += module_time
        
        f.write("-" * 50 + "\n")
        f.write(f"  {'Total Module Time':30s} : {total_module_time:9.4f}s\n")
        e2e_time = timing_dict.get("end_to_end", total_module_time)
        f.write(f"  {'End-to-End Time':30s} : {e2e_time:9.4f}s\n")
        f.write(f"  {'Overhead (E2E - Modules)':30s} : {e2e_time - total_module_time:9.4f}s\n\n")
        
        # GA Results
        f.write("="*80 + "\n")
        f.write("GENETIC ALGORITHM - ENSEMBLE SELECTION\n")
        f.write("="*80 + "\n\n")
        
        ga_results = results_dict.get("ga", {})
        f.write(f"Best Ensemble: {ga_results.get('ensemble', 'N/A')}\n")
        f.write(f"  F1 Score      : {ga_results.get('f1', 0.0):.6f}\n")
        f.write(f"  PR-AUC        : {ga_results.get('pr_auc', 0.0):.6f}\n")
        f.write(f"  Fitness       : {ga_results.get('fitness', 0.0):.6f}\n")
        f.write(f"  Meta-Model    : {ga_results.get('meta_model_type', 'N/A')}\n")
        f.write(f"  Ensemble Size : {len(ga_results.get('ensemble', []))}\n")
        f.write(f"  Time          : {timing_dict.get('1_Genetic_Algorithm', 0.0):.4f}s\n\n")
        # Thompson Sampling Results
        f.write("="*80 + "\n")
        f.write("THOMPSON SAMPLING - ONLINE MODEL SELECTION\n")
        f.write("="*80 + "\n\n")
        
        thompson_results = results_dict.get("thompson", {})
        f.write("Top-5 Models (Ranked):\n")
        for i, model in enumerate(thompson_results.get("top_models", [])[:5], 1):
            f.write(f"  {i}. {model}\n")
        f.write(f"\nBest Model: {thompson_results.get('best_model', 'N/A')}\n")
        f.write(f"Time      : {timing_dict.get('2_Thompson_Sampling', 0.0):.4f}s\n\n")
        
        # Robustness Tests
        f.write("="*80 + "\n")
        f.write("ROBUSTNESS TESTS - PER-MODULE PERFORMANCE\n")
        f.write("="*80 + "\n\n")
        
        # GAN Results
        f.write("GAN Robustness Test:\n")
        f.write("-" * 50 + "\n")
        gan_results_rob = results_dict.get("gan", {})
        f.write(f"  Best Model (by F1)    : {gan_results_rob.get('best_model', 'N/A')}\n")
        f.write(f"  Best F1 Score         : {gan_results_rob.get('best_f1', 0.0):.6f}\n")
        f.write(f"  Best PR-AUC Score     : {gan_results_rob.get('best_pr_auc', 0.0):.6f}\n")
        f.write(f"  Computational Time    : {timing_dict.get('3_GAN_Robustness', 0.0):.4f}s\n")
        f.write("\n  Top-5 by F1 (with scores):\n")
        f1_scores = gan_results_rob.get("f1_scores", [])
        for i, model in enumerate(gan_results_rob.get("f1_names", [])[:5], 1):
            score = f1_scores[i-1] if isinstance(f1_scores, list) and len(f1_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | F1: {score:.6f}\n")
        f.write("\n  Top-5 by PR-AUC (with scores):\n")
        pr_auc_scores = gan_results_rob.get("pr_auc_scores", [])
        for i, model in enumerate(gan_results_rob.get("pr_auc_names", [])[:5], 1):
            score = pr_auc_scores[i-1] if isinstance(pr_auc_scores, list) and len(pr_auc_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | PR-AUC: {score:.6f}\n")
        f.write("\n")
        
        # Borderline Results
        f.write("Borderline Sensitivity Test:\n")
        f.write("-" * 50 + "\n")
        borderline_results = results_dict.get("borderline", {})
        f.write(f"  Best Model (by F1)    : {borderline_results.get('best_model', 'N/A')}\n")
        f.write(f"  Best F1 Score         : {borderline_results.get('best_f1', 0.0):.6f}\n")
        f.write(f"  Best PR-AUC Score     : {borderline_results.get('best_pr_auc', 0.0):.6f}\n")
        f.write(f"  Computational Time    : {timing_dict.get('4_Borderline_Sensitivity', 0.0):.4f}s\n")
        f.write("\n  Top-5 by F1 (with scores):\n")
        f1_scores = borderline_results.get("f1_scores", [])
        for i, model in enumerate(borderline_results.get("f1_names", [])[:5], 1):
            score = f1_scores[i-1] if isinstance(f1_scores, list) and len(f1_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | F1: {score:.6f}\n")
        f.write("\n  Top-5 by PR-AUC (with scores):\n")
        pr_auc_scores = borderline_results.get("pr_auc_scores", [])
        for i, model in enumerate(borderline_results.get("pr_auc_names", [])[:5], 1):
            score = pr_auc_scores[i-1] if isinstance(pr_auc_scores, list) and len(pr_auc_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | PR-AUC: {score:.6f}\n")
        f.write("\n")
        # Monte Carlo Results
        f.write("Monte Carlo Simulation:\n")
        f.write("-" * 50 + "\n")
        mc_results = results_dict.get("monte_carlo", {})
        f.write(f"  Best Model (by F1)    : {mc_results.get('best_model_f1', 'N/A')}\n")
        f.write(f"  Best Model (by PR-AUC): {mc_results.get('best_model_pr_auc', 'N/A')}\n")
        f.write(f"  Computational Time    : {timing_dict.get('5_Monte_Carlo', 0.0):.4f}s\n")
        f.write("\n  Top-5 by F1 (with scores):\n")
        f1_scores = mc_results.get("f1_scores", [])
        for i, model in enumerate(mc_results.get("f1_names", [])[:5], 1):
            score = f1_scores[i-1] if isinstance(f1_scores, list) and len(f1_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | F1: {score:.6f}\n")
        f.write("\n  Top-5 by PR-AUC (with scores):\n")
        pr_auc_scores = mc_results.get("pr_auc_scores", [])
        for i, model in enumerate(mc_results.get("pr_auc_names", [])[:5], 1):
            score = pr_auc_scores[i-1] if isinstance(pr_auc_scores, list) and len(pr_auc_scores) >= i else 0.0
            f.write(f"    {i}. {model:20s} | PR-AUC: {score:.6f}\n")
        f.write("\n")
        
        # Rank Aggregation
        f.write("="*80 + "\n")
        f.write("RANK AGGREGATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        aggregation_results = results_dict.get("aggregation", {})
        
        f.write("Robust Aggregation (GAN → Borderline → Monte Carlo):\n")
        f.write("-" * 50 + "\n")
        robust_agg = aggregation_results.get("robust_agg", [])
        if isinstance(robust_agg, (list, tuple)) and len(robust_agg) > 1:
            f.write(f"  Best Model: {robust_agg[1] if len(robust_agg) > 1 else 'N/A'}\n")
            f.write(f"  Full Ranking: {robust_agg}\n")
        else:
            f.write(f"  Result: {robust_agg}\n")
        f.write(f"  Time: {timing_dict.get('6_Rank_Aggregation', 0.0):.4f}s\n")
        f.write("\n")
        
        f.write("Final Aggregation (Robust + Thompson Sampling):\n")
        f.write("-" * 50 + "\n")
        final_agg = aggregation_results.get("final_agg", [])
        if isinstance(final_agg, (list, tuple)) and len(final_agg) > 1:
            f.write(f"  Best Model: {final_agg[1] if len(final_agg) > 1 else 'N/A'}\n")
            f.write(f"  Full Ranking: {final_agg}\n")
        else:
            f.write(f"  Result: {final_agg}\n")
        f.write("\n")
        
        # Framework Decision
        f.write("="*80 + "\n")
        f.write("FRAMEWORK FINAL DECISION\n")
        f.write("="*80 + "\n\n")
        
        final_decision = results_dict.get("final_decision", {})
        
        f.write(f"Framework Choice      : {final_decision.get('choice', 'N/A').upper()}\n\n")
        
        f.write("Performance Comparison:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Single Model:\n")
        f.write(f"    Model        : {final_decision.get('single_model', 'N/A')}\n")
        f.write(f"    F1 Score     : {final_decision.get('single_model_f1', 0.0):.6f}\n")
        f.write(f"    PR-AUC       : {final_decision.get('single_model_pr_auc', 0.0):.6f}\n\n")
        
        f.write(f"  Ensemble:\n")
        f.write(f"    Models       : {final_decision.get('ensemble', 'N/A')}\n")
        f.write(f"    Size         : {len(final_decision.get('ensemble', []))}\n")
        f.write(f"    Meta-Model   : {final_decision.get('meta_model_type', 'N/A')}\n")
        f.write(f"    F1 Score     : {final_decision.get('ensemble_f1', 0.0):.6f}\n")
        f.write(f"    PR-AUC       : {final_decision.get('ensemble_pr_auc', 0.0):.6f}\n")
        f.write(f"    Fitness      : {final_decision.get('ensemble_fitness', 0.0):.6f}\n\n")
        
        # Decision Rationale
        ensemble_f1 = final_decision.get("ensemble_f1", 0.0)
        single_f1 = final_decision.get("single_model_f1", 0.0)
        f1_diff = ensemble_f1 - single_f1
        
        f.write("Decision Rationale:\n")
        f.write("-" * 50 + "\n")
        if final_decision.get("choice") == "ensemble":
            f.write(f"  ✓ Ensemble selected because F1 ({ensemble_f1:.6f}) is HIGHER\n")
            f.write(f"    than single model F1 ({single_f1:.6f})\n")
            f.write(f"    Improvement: +{f1_diff:.6f} ({f1_diff/max(single_f1, 0.0001)*100:.2f}%)\n")
        else:
            f.write(f"  ✓ Single model selected because F1 ({single_f1:.6f}) is HIGHER\n")
            f.write(f"    than ensemble F1 ({ensemble_f1:.6f})\n")
            f.write(f"    Advantage: +{-f1_diff:.6f} ({-f1_diff/max(ensemble_f1, 0.0001)*100:.2f}%)\n")
        f.write("\n")
        
        # End
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
