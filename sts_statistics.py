import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

def extract_results_from_filenames(file_pattern="*.txt"):
    """
    Extract results from txt filenames containing correlation values
    """
    results = []
    files = glob.glob(file_pattern)
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            
            # Extract seed, alpha, and correlation using regex
            pattern = r"seed_(\d+)_alpha_([\d.]+)_batch_[\d.]+_corr_([\d.]+)\.txt"
            match = re.search(pattern, filename)
            
            if match:
                seed = int(match.group(1))
                alpha = float(match.group(2))
                correlation = float(match.group(3))
                
                results.append({
                    'alpha': alpha,
                    'seed': seed,
                    'correlation': correlation,
                    'filename': filename
                })
            else:
                print(f"Could not parse filename: {filename}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
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
        'correlation': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)
    
    # Calculate confidence intervals
    analysis['ci_lower'] = analysis[('correlation', 'mean')] - 1.96 * analysis[('correlation', 'std')] / np.sqrt(analysis[('correlation', 'count')])
    analysis['ci_upper'] = analysis[('correlation', 'mean')] + 1.96 * analysis[('correlation', 'std')] / np.sqrt(analysis[('correlation', 'count')])
    
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
    
    alpha_stats = results_df.groupby('alpha')['correlation'].agg(['mean', 'std', 'count'])
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
    sns.boxplot(data=results_df, x='alpha', y='correlation')
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Distribution of Correlation Scores by Alpha Value', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/alpha_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Violin plot showing distribution
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=results_df, x='alpha', y='correlation')
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Distribution of Correlation Scores by Alpha Value', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/alpha_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Individual points with jitter
    plt.figure(figsize=(14, 8))
    sns.stripplot(data=results_df, x='alpha', y='correlation', jitter=True, alpha=0.6)
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Individual Correlation Scores by Alpha Value', fontsize=16)
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/alpha_points.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(results_df):
    """
    Perform statistical tests to determine significance between alpha values
    """
    tests = {}
    
    # Test different alpha values
    alpha_groups = [group['correlation'].values 
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
                group1 = results_df[results_df['alpha'] == alpha1]['correlation']
                group2 = results_df[results_df['alpha'] == alpha2]['correlation']
                
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
    
    alpha_stats = results_df.groupby('alpha')['correlation'].agg(['mean', 'std', 'count'])
    alpha_stats['ci_lower'] = alpha_stats['mean'] - 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    alpha_stats['ci_upper'] = alpha_stats['mean'] + 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    
    for alpha, row in alpha_stats.iterrows():
        report.append(f"| {alpha} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} | {row['ci_lower']:.4f} | {row['ci_upper']:.4f} |")
    
    # Best performing alpha
    best_alpha = results_df.loc[results_df['correlation'].idxmax(), 'alpha']
    best_corr = results_df['correlation'].max()
    
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
    folder_path = "/path/to/your/txt/files"
    file_pattern = os.path.join(folder_path, "*.txt")
    
    results_df = extract_results_from_filenames(file_pattern)
    
    print(f"Extracted {len(results_df)} results")
    print(f"Alphas tested: {sorted(results_df['alpha'].unique())}")
    
    # Save raw results
    results_df.to_csv('alpha_sweep_results.csv', index=False)
    print("Saved raw results to alpha_sweep_results.csv")
    
    # Analyze performance
    print("Analyzing alpha performance...")
    alpha_analysis = analyze_alpha_performance(results_df)
    print("\nAlpha Performance Summary:")
    print(alpha_analysis)
    
    # Create visualizations
    print("Creating visualizations...")
    create_alpha_visualizations(results_df)
    print("Visualizations saved to alpha_analysis_plots/")
    
    # Create detailed report
    print("Generating detailed report...")
    create_detailed_report(results_df)
    print("Report saved to alpha_analysis_report.md")
    
    # Print summary
    best_alpha = results_df.loc[results_df['correlation'].idxmax(), 'alpha']
    best_corr = results_df['correlation'].max()
    worst_alpha = results_df.loc[results_df['correlation'].idxmin(), 'alpha']
    worst_corr = results_df['correlation'].min()
    
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Best Alpha: {best_alpha} (Correlation: {best_corr:.4f})")
    print(f"Worst Alpha: {worst_alpha} (Correlation: {worst_corr:.4f})")
    print(f"Overall Mean: {results_df['correlation'].mean():.4f}")
    print(f"Overall Std: {results_df['correlation'].std():.4f}")
    
    # Show mean performance by alpha
    print(f"\n=== MEAN PERFORMANCE BY ALPHA ===")
    for alpha, group in results_df.groupby('alpha'):
        print(f"Alpha {alpha}: {group['correlation'].mean():.4f} ± {group['correlation'].std():.4f} (n={len(group)})")

if __name__ == "__main__":
    main()