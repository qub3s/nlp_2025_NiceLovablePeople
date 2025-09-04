import numpy as np
import pandas as pd
import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from torch.utils.data import DataLoader
from datasets import load_multitask_data, SentencePairDataset
from multitask_classifier import MultitaskBERT  # Import your model class
from evaluation import model_eval_multitask  # Import your evaluation function

def load_model_and_evaluate(model_path, args, device):
    """
    Load a saved model and evaluate it on STS dev set
    """
    # Load saved model
    saved = torch.load(model_path, map_location=device)
    config = saved["model_config"]
    
    # Initialize model
    model = MultitaskBERT(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()
    
    # Load STS dev data
    _, _, _, sts_dev_data, _ = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )
    
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args)
    sts_dev_dataloader = DataLoader(
        sts_dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_dataset.collate_fn,
    )
    
    # Evaluate model
    with torch.no_grad():
        _, _, _, _, _, _, sts_dev_corr, _, _, _, _, _ = model_eval_multitask(
            None, None, sts_dev_dataloader, None, model=model, device=device, task="sts"
        )
    
    return sts_dev_corr

def load_results_from_directory(base_dir="sts_sweep_results_25seeds", args=None):
    """
    Load all models from the sweep directory and evaluate them
    """
    results = []
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    
    # Find all experiment directories
    dirs = glob.glob(f"{base_dir}/alpha_*_seed_*")
    
    for dir_path in tqdm(dirs, desc="Evaluating models"):
        try:
            # Extract parameters from directory name
            dir_name = os.path.basename(dir_path)
            parts = dir_name.split('_')
            alpha = float(parts[1])
            seed = int(parts[3])
            
            # Find model file
            model_files = glob.glob(os.path.join(dir_path, "*.pt"))
            if not model_files:
                print(f"No model file found in {dir_path}")
                continue
            
            model_path = model_files[0]  # Take the first model file found
            
            # Evaluate model
            correlation = load_model_and_evaluate(model_path, args, device)
            
            results.append({
                'alpha': alpha,
                'seed': seed,
                'final_correlation': correlation,
                'directory': dir_path,
                'model_path': model_path
            })
            
        except Exception as e:
            print(f"Error processing {dir_path}: {e}")
            continue
    
    return pd.DataFrame(results)

def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    """
    Calculate bootstrap confidence intervals
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        bootstrap_stats.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, ci + (100 - ci) / 2)
    mean = np.mean(bootstrap_stats)
    
    return mean, lower, upper

def analyze_alpha_performance(results_df):
    """
    Analyze performance by alpha value with confidence intervals
    """
    analysis = results_df.groupby('alpha').agg({
        'final_correlation': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)
    
    # Calculate confidence intervals
    analysis['ci_lower'] = analysis[('final_correlation', 'mean')] - 1.96 * analysis[('final_correlation', 'std')] / np.sqrt(analysis[('final_correlation', 'count')])
    analysis['ci_upper'] = analysis[('final_correlation', 'mean')] + 1.96 * analysis[('final_correlation', 'std')] / np.sqrt(analysis[('final_correlation', 'count')])
    
    return analysis

def create_alpha_visualizations(results_df, output_dir="alpha_analysis_plots"):
    """
    Create visualizations for alpha parameter analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Plot 1: Alpha vs Correlation with confidence intervals
    plt.figure(figsize=(12, 8))
    
    alpha_stats = results_df.groupby('alpha')['final_correlation'].agg(['mean', 'std', 'count'])
    alpha_stats['ci'] = 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    
    plt.errorbar(alpha_stats.index, alpha_stats['mean'], yerr=alpha_stats['ci'], 
                 fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Effect of Alpha Parameter on STS Performance\n(SimCSE + SBERT Combined)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(alpha_stats.index)
    plt.savefig(f'{output_dir}/alpha_vs_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution by alpha
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=results_df, x='alpha', y='final_correlation')
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Distribution of Correlation Scores by Alpha Value', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/alpha_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Violin plot showing distribution
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=results_df, x='alpha', y='final_correlation')
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Distribution of Correlation Scores by Alpha Value', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/alpha_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(results_df):
    """
    Perform statistical tests to determine significance between alpha values
    """
    tests = {}
    
    # Test different alpha values
    alpha_groups = [group['final_correlation'].values 
                   for _, group in results_df.groupby('alpha')]
    
    if len(alpha_groups) > 1:
        # ANOVA test
        f_stat, p_value = stats.f_oneway(*alpha_groups)
        tests['alpha_anova'] = {'f_stat': f_stat, 'p_value': p_value}
        
        # Pairwise t-tests (Bonferroni corrected)
        alpha_values = sorted(results_df['alpha'].unique())
        pairwise_tests = {}
        for i in range(len(alpha_values)):
            for j in range(i+1, len(alpha_values)):
                alpha1, alpha2 = alpha_values[i], alpha_values[j]
                group1 = results_df[results_df['alpha'] == alpha1]['final_correlation']
                group2 = results_df[results_df['alpha'] == alpha2]['final_correlation']
                
                if len(group1) > 1 and len(group2) > 1:  # Need at least 2 samples
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    pairwise_tests[f"{alpha1}_vs_{alpha2}"] = {
                        't_stat': t_stat, 
                        'p_value': p_val,
                        'significant': p_val < 0.05/len(alpha_values)  # Bonferroni correction
                    }
        tests['alpha_pairwise'] = pairwise_tests
    
    return tests

def create_detailed_report(results_df, output_file="alpha_analysis_report.md"):
    """
    Create a detailed markdown report of the analysis
    """
    report = [
        "# Alpha Parameter Analysis Report",
        "## Summary Statistics",
        f"- Total experiments: {len(results_df)}",
        f"- Alpha values tested: {sorted(results_df['alpha'].unique())}",
        f"- Seeds per alpha: {results_df.groupby('alpha').size().to_dict()}",
        "",
        "## Performance by Alpha",
        "| Alpha | Mean Correlation | Std Dev | Count | 95% CI Lower | 95% CI Upper |",
        "|-------|------------------|---------|-------|--------------|--------------|"
    ]
    
    alpha_stats = results_df.groupby('alpha')['final_correlation'].agg(['mean', 'std', 'count'])
    alpha_stats['ci_lower'] = alpha_stats['mean'] - 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    alpha_stats['ci_upper'] = alpha_stats['mean'] + 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    
    for alpha, row in alpha_stats.iterrows():
        report.append(f"| {alpha} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} | {row['ci_lower']:.4f} | {row['ci_upper']:.4f} |")
    
    # Best performing alpha
    best_alpha = results_df.loc[results_df['final_correlation'].idxmax(), 'alpha']
    best_corr = results_df['final_correlation'].max()
    
    report.extend([
        "",
        "## Best Performing Configuration",
        f"- **Best Alpha**: {best_alpha}",
        f"- **Best Correlation**: {best_corr:.4f}",
        "",
        "## Statistical Significance",
        "### ANOVA Test"
    ])
    
    tests = perform_statistical_tests(results_df)
    if 'alpha_anova' in tests:
        report.append(f"- F-statistic: {tests['alpha_anova']['f_stat']:.4f}")
        report.append(f"- p-value: {tests['alpha_anova']['p_value']:.4e}")
        report.append(f"- Significant: {tests['alpha_anova']['p_value'] < 0.05}")
    
    if 'alpha_pairwise' in tests:
        report.append("")
        report.append("### Pairwise Comparisons (Bonferroni corrected)")
        for comparison, stats in tests['alpha_pairwise'].items():
            sig_flag = "✓" if stats['significant'] else "✗"
            report.append(f"- {comparison}: p = {stats['p_value']:.4e} {sig_flag}")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return report

def main():
    """Main analysis function"""
    from types import SimpleNamespace
    
    # Create minimal args for evaluation
    args = SimpleNamespace(
        sst_dev="data/ids-sst-dev.csv",
        quora_dev="data/quora-dev.csv", 
        sts_dev="data/sts-dev.csv",
        etpc_dev="data/etpc-dev.csv",
        batch_size=64,
        use_gpu=True,
        hidden_dropout_prob=0.3,
        # Add other necessary arguments that your datasets need
    )
    
    print("Loading and evaluating models from sweep...")
    results_df = load_results_from_directory("sts_sweep_results_25seeds", args)
    
    if len(results_df) == 0:
        print("No results found! Make sure the sweep has completed.")
        return
    
    print(f"Evaluated {len(results_df)} models")
    print(f"Alphas tested: {sorted(results_df['alpha'].unique())}")
    
    # Save raw results
    results_df.to_csv('alpha_sweep_results.csv', index=False)
    
    # Analyze performance
    print("Analyzing alpha performance...")
    alpha_analysis = analyze_alpha_performance(results_df)
    print("\nAlpha Performance Summary:")
    print(alpha_analysis)
    
    # Create visualizations
    print("Creating visualizations...")
    create_alpha_visualizations(results_df)
    
    # Create detailed report
    print("Generating detailed report...")
    create_detailed_report(results_df)
    
    # Print summary
    best_alpha = results_df.loc[results_df['final_correlation'].idxmax(), 'alpha']
    best_corr = results_df['final_correlation'].max()
    
    print(f"\n=== BEST PERFORMING ALPHA ===")
    print(f"Alpha: {best_alpha}")
    print(f"Correlation: {best_corr:.4f}")
    print(f"Results saved to alpha_sweep_results.csv and alpha_analysis_report.md")

if __name__ == "__main__":
    from tqdm import tqdm
    main()