#!/usr/bin/env python3
"""
Visualize RAMSeS Testbed Results

Generate plots and charts from testbed results including:
- Computational overhead by domain
- F1 score comparison across modules
- Memory usage analysis
- Per-dataset performance
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class TestbedVisualizer:
    """Visualize testbed results"""
    
    def __init__(self, results_dir: str = "testbed_results"):
        """
        Initialize visualizer
        
        Parameters
        ----------
        results_dir : str
            Directory containing testbed results
        """
        self.results_dir = results_dir
        self.summary_file = f"{results_dir}/overall_summary.json"
        
        if not Path(self.summary_file).exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")
        
        with open(self.summary_file, 'r') as f:
            self.summary = json.load(f)
        
        logger.info(f"Loaded summary for {self.summary['total_domains']} domains")
    
    def plot_computational_overhead(self, output_file: str = None):
        """Plot average computational overhead by domain"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        domains = []
        avg_times = []
        total_times = []
        
        for domain, stats in self.summary['domains'].items():
            domains.append(domain)
            avg_times.append(stats['avg_runtime'])
            total_times.append(stats['total_runtime'])
        
        # Average runtime per dataset
        bars1 = ax1.bar(domains, avg_times, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Domain', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Computational Overhead per Dataset', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=10)
        
        # Total runtime per domain
        bars2 = ax2.bar(domains, [t/60 for t in total_times], color='coral', alpha=0.8)
        ax2.set_xlabel('Domain', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Runtime (minutes)', fontsize=12, fontweight='bold')
        ax2.set_title('Total Computational Overhead per Domain', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}m',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved computational overhead plot to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_module_timing_breakdown(self, output_file: str = None):
        """Plot average timing breakdown by module across domains"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect module timings for all domains
        all_modules = set()
        domain_data = {}
        
        for domain, stats in self.summary['domains'].items():
            module_times = stats.get('avg_module_times', {})
            domain_data[domain] = module_times
            all_modules.update(module_times.keys())
        
        # Remove end_to_end from modules list
        all_modules.discard('end_to_end')
        modules = sorted(all_modules)
        
        # Create grouped bar chart
        x = np.arange(len(modules))
        width = 0.8 / len(domain_data)
        
        for idx, (domain, module_times) in enumerate(domain_data.items()):
            times = [module_times.get(module, 0) for module in modules]
            offset = (idx - len(domain_data)/2) * width + width/2
            ax.bar(x + offset, times, width, label=domain, alpha=0.8)
        
        ax.set_xlabel('Module', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Average Module Timing Breakdown by Domain', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in modules], rotation=45, ha='right')
        ax.legend(title='Domain', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved module timing plot to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_f1_scores_comparison(self, output_file: str = None):
        """Plot F1 scores comparison across modules and domains"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect F1 scores
        modules = ['ga', 'thompson', 'gan', 'borderline', 'monte_carlo', 'final_selected']
        module_labels = ['GA\nEnsemble', 'Thompson\nSampling', 'GAN\nRobustness', 
                        'Borderline\nSensitivity', 'Monte\nCarlo', 'Final\nSelected']
        
        domain_data = {}
        for domain, stats in self.summary['domains'].items():
            f1_scores = stats.get('avg_f1_scores', {})
            domain_data[domain] = [f1_scores.get(m, 0) for m in modules]
        
        # Create grouped bar chart
        x = np.arange(len(modules))
        width = 0.8 / len(domain_data)
        
        for idx, (domain, f1_values) in enumerate(domain_data.items()):
            offset = (idx - len(domain_data)/2) * width + width/2
            bars = ax.bar(x + offset, f1_values, width, label=domain, alpha=0.8)
            
            # Add value labels for final_selected only
            if idx == 0:  # Add labels only once
                for i, bar in enumerate(bars):
                    if i == len(bars) - 1:  # final_selected
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{height:.3f}',
                                  ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Module', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Average F1 Scores by Module and Domain', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(module_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(title='Domain', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Perfect Score')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved F1 scores plot to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_memory_usage(self, output_file: str = None):
        """Plot memory usage statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        domains = []
        avg_memory = []
        peak_memory = []
        
        for domain, stats in self.summary['domains'].items():
            domains.append(domain)
            avg_memory.append(stats.get('avg_memory_mb', 0))
            peak_memory.append(stats.get('peak_memory_mb', 0))
        
        # Average memory usage
        bars1 = ax1.bar(domains, avg_memory, color='mediumseagreen', alpha=0.8)
        ax1.set_xlabel('Domain', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Memory (MB)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Memory Usage per Domain', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}MB',
                    ha='center', va='bottom', fontsize=10)
        
        # Peak memory usage
        bars2 = ax2.bar(domains, peak_memory, color='orangered', alpha=0.8)
        ax2.set_xlabel('Domain', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Peak Memory per Domain', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}MB',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved memory usage plot to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_overall_summary(self, output_file: str = None):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Total runtime by domain
        ax1 = fig.add_subplot(gs[0, :2])
        domains = list(self.summary['domains'].keys())
        total_times = [self.summary['domains'][d]['total_runtime']/60 for d in domains]
        bars = ax1.barh(domains, total_times, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Total Runtime (minutes)', fontweight='bold')
        ax1.set_title('Total Computational Time per Domain', fontweight='bold', fontsize=12)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}m', ha='left', va='center', fontsize=9)
        
        # 2. Dataset count
        ax2 = fig.add_subplot(gs[0, 2])
        dataset_counts = [self.summary['domains'][d]['total_datasets'] for d in domains]
        ax2.pie(dataset_counts, labels=domains, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Datasets per Domain', fontweight='bold', fontsize=12)
        
        # 3. F1 scores comparison
        ax3 = fig.add_subplot(gs[1, :])
        modules = ['ga', 'thompson', 'gan', 'borderline', 'monte_carlo', 'final_selected']
        x = np.arange(len(modules))
        width = 0.8 / len(domains)
        
        for idx, domain in enumerate(domains):
            f1_scores = self.summary['domains'][domain].get('avg_f1_scores', {})
            values = [f1_scores.get(m, 0) for m in modules]
            offset = (idx - len(domains)/2) * width + width/2
            ax3.bar(x + offset, values, width, label=domain, alpha=0.8)
        
        ax3.set_xlabel('Module', fontweight='bold')
        ax3.set_ylabel('Average F1 Score', fontweight='bold')
        ax3.set_title('Average F1 Scores by Module', fontweight='bold', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['GA', 'Thompson', 'GAN', 'Borderline', 'MC', 'Final'])
        ax3.legend(title='Domain')
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Memory usage
        ax4 = fig.add_subplot(gs[2, 0])
        avg_mem = [self.summary['domains'][d]['avg_memory_mb'] for d in domains]
        ax4.bar(domains, avg_mem, color='mediumseagreen', alpha=0.8)
        ax4.set_ylabel('Memory (MB)', fontweight='bold')
        ax4.set_title('Avg Memory Usage', fontweight='bold', fontsize=11)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Peak memory
        ax5 = fig.add_subplot(gs[2, 1])
        peak_mem = [self.summary['domains'][d]['peak_memory_mb'] for d in domains]
        ax5.bar(domains, peak_mem, color='orangered', alpha=0.8)
        ax5.set_ylabel('Memory (MB)', fontweight='bold')
        ax5.set_title('Peak Memory Usage', fontweight='bold', fontsize=11)
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        summary_text = f"""
        OVERALL SUMMARY
        
        Total Domains: {self.summary['total_domains']}
        Total Datasets: {self.summary['total_datasets']}
        
        Total Time: {self.summary['total_time']/3600:.2f} hours
        
        Avg Time/Dataset: 
        {self.summary['total_time']/self.summary['total_datasets']:.1f}s
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('RAMSeS Testbed - Comprehensive Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary dashboard to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        output_dir = f"{self.results_dir}/plots"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating all visualization plots...")
        
        self.plot_computational_overhead(f"{output_dir}/computational_overhead.png")
        self.plot_module_timing_breakdown(f"{output_dir}/module_timing_breakdown.png")
        self.plot_f1_scores_comparison(f"{output_dir}/f1_scores_comparison.png")
        self.plot_memory_usage(f"{output_dir}/memory_usage.png")
        self.plot_overall_summary(f"{output_dir}/overall_summary_dashboard.png")
        
        logger.info(f"All plots saved to: {output_dir}/")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RAMSeS testbed results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='testbed_results',
        help='Directory containing testbed results'
    )
    parser.add_argument(
        '--plot',
        type=str,
        choices=['overhead', 'modules', 'f1', 'memory', 'summary', 'all'],
        default='all',
        help='Type of plot to generate'
    )
    
    args = parser.parse_args()
    
    visualizer = TestbedVisualizer(args.results_dir)
    
    output_dir = f"{args.results_dir}/plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.plot == 'all':
        visualizer.generate_all_plots()
    elif args.plot == 'overhead':
        visualizer.plot_computational_overhead(f"{output_dir}/computational_overhead.png")
    elif args.plot == 'modules':
        visualizer.plot_module_timing_breakdown(f"{output_dir}/module_timing_breakdown.png")
    elif args.plot == 'f1':
        visualizer.plot_f1_scores_comparison(f"{output_dir}/f1_scores_comparison.png")
    elif args.plot == 'memory':
        visualizer.plot_memory_usage(f"{output_dir}/memory_usage.png")
    elif args.plot == 'summary':
        visualizer.plot_overall_summary(f"{output_dir}/overall_summary_dashboard.png")


if __name__ == '__main__':
    main()
