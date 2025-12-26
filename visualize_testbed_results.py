#!/usr/bin/env python3
"""
Visualization script for RAMSeS testbed results.
Generates plots and charts from the aggregated results.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path

def load_results(results_dir='myresults/testbed_aggregated'):
    """Load the most recent testbed results."""
    results_path = Path(results_dir)
    
    # Find most recent CSV file
    csv_files = list(results_path.glob('detailed_results_*.csv'))
    if not csv_files:
        print(f"❌ No results found in {results_dir}")
        return None
    
    latest_csv = sorted(csv_files)[-1]
    print(f"Loading: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    df_success = df[df['status'] == 'success'].copy()
    
    print(f"Loaded {len(df_success)} successful experiments")
    return df_success

def plot_method_comparison(df, output_dir='myresults/testbed_aggregated'):
    """Create bar chart comparing all methods."""
    methods = ['ga', 'thompson', 'gan', 'borderline', 'montecarlo']
    method_names = ['GA', 'Thompson', 'GAN', 'Borderline', 'Monte Carlo']
    
    f1_means = [df[f'{m}_f1'].mean() for m in methods]
    f1_stds = [df[f'{m}_f1'].std() for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(methods))
    
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('RAMSeS Method Comparison - F1 Scores Across All Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, f1_means, f1_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'method_comparison_f1.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_timing_comparison(df, output_dir='myresults/testbed_aggregated'):
    """Create bar chart comparing computational overhead."""
    modules = ['ga', 'thompson', 'gan', 'borderline', 'montecarlo']
    module_names = ['GA', 'Thompson', 'GAN', 'Borderline', 'Monte Carlo']
    
    time_means = [df[f'{m}_duration'].mean() for m in modules]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(modules))
    
    bars = ax.bar(x, time_means, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_xlabel('Module', fontsize=12)
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12)
    ax.set_title('RAMSeS Computational Overhead - Average Runtime per Module', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(module_names)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, time_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(time_means)*0.01,
                f'{mean:.1f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'timing_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_per_dataset_f1(df, output_dir='myresults/testbed_aggregated'):
    """Create grouped bar chart for F1 scores per dataset."""
    methods = ['ga_f1', 'thompson_f1', 'gan_f1', 'borderline_f1', 'montecarlo_f1']
    method_labels = ['GA', 'Thompson', 'GAN', 'Borderline', 'MC']
    
    # Group by dataset
    dataset_means = df.groupby('dataset')[methods].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(dataset_means))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        offset = (i - 2) * width
        ax.bar([xi + offset for xi in x], dataset_means[method], 
               width, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Scores by Dataset and Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_means.index, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'per_dataset_f1.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_f1_distributions(df, output_dir='myresults/testbed_aggregated'):
    """Create box plots showing F1 score distributions."""
    methods = ['ga_f1', 'thompson_f1', 'gan_f1', 'borderline_f1', 'montecarlo_f1']
    method_labels = ['GA', 'Thompson', 'GAN', 'Borderline', 'Monte Carlo']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [df[method].dropna() for method in methods]
    
    bp = ax.boxplot(data, labels=method_labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Distribution by Method', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'f1_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_timing_breakdown(df, output_dir='myresults/testbed_aggregated'):
    """Create stacked bar chart showing timing breakdown."""
    modules = ['ga_duration', 'thompson_duration', 'gan_duration', 
               'borderline_duration', 'montecarlo_duration']
    module_labels = ['GA', 'Thompson', 'GAN', 'Borderline', 'Monte Carlo']
    
    # Calculate percentages
    totals = df[modules].sum(axis=1)
    percentages = df[modules].div(totals, axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average percentages
    avg_percentages = percentages.mean()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    wedges, texts, autotexts = ax.pie(avg_percentages, labels=module_labels, 
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors, textprops={'fontsize': 10})
    
    ax.set_title('Average Computational Overhead Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'timing_breakdown_pie.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize RAMSeS testbed results')
    parser.add_argument('-d', '--dir', default='myresults/testbed_aggregated',
                        help='Directory containing testbed results')
    args = parser.parse_args()
    
    print("="*80)
    print("RAMSeS Testbed Results Visualization")
    print("="*80)
    print()
    
    # Load data
    df = load_results(args.dir)
    if df is None:
        sys.exit(1)
    
    print(f"\nGenerating visualizations...")
    
    # Create plots
    plot_method_comparison(df, args.dir)
    plot_timing_comparison(df, args.dir)
    plot_per_dataset_f1(df, args.dir)
    plot_f1_distributions(df, args.dir)
    plot_timing_breakdown(df, args.dir)
    
    print()
    print("="*80)
    print("✅ All visualizations generated successfully!")
    print(f"Location: {args.dir}")
    print("="*80)

if __name__ == '__main__':
    main()
