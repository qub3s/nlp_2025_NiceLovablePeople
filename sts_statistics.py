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

def create_alpha_visualizations(results_df, output_dir="alpha_analysis_plots"):
    """
    Create visualizations for alpha parameter analysis with confidence intervals
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    # Calculate statistics for each alpha
    alpha_stats = results_df.groupby('alpha')['correlation'].agg(['mean', 'std', 'count'])
    alpha_stats['ci'] = 1.96 * alpha_stats['std'] / np.sqrt(alpha_stats['count'])
    
    # Get unique alpha values and sort them
    alphas = sorted(results_df['alpha'].unique())
    
    # Calculate means and confidence intervals for each alpha
    means = []
    ci_lower = []
    ci_upper = []
    
    for alpha in alphas:
        data = results_df[results_df['alpha'] == alpha]['correlation']
        mean, lower, upper = bootstrap_confidence_interval(data)
        means.append(mean)
        ci_lower.append(lower)
        ci_upper.append(upper)

    # Plot Alpha vs Scores
    plt.figure(figsize=(8, 5))

    plt.fill_between(alpha_stats.index, 
                    alpha_stats['mean'] - alpha_stats['ci'], 
                    alpha_stats['mean'] + alpha_stats['ci'], 
                    alpha=0.3, color='#1f77b4', label='95% CI')

    plt.plot(alpha_stats.index, alpha_stats['mean'], 
            'o-', color="blue", linewidth=2, markersize=6, 
            label='Mean Correlation')

    plt.xlabel('Alpha Value', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.title('Effect of Alpha Parameter on STS Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.xlim(alpha_stats.index.min() - 0.1, alpha_stats.index.max() + 0.1)

    plt.legend(loc='best', fontsize=10)
    plt.savefig(f'{output_dir}/alpha_vs_correlation_ci.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot Individual points with mean and CI
    plt.figure(figsize=(12, 8))
    
    for i, alpha in enumerate(alphas):
        alpha_data = results_df[results_df['alpha'] == alpha]['correlation']
        jitter = np.random.normal(0, 0.05, len(alpha_data))
        plt.scatter([i + j for j in jitter], alpha_data, alpha=0.6, s=40)

    plt.errorbar(range(len(alphas)), means, yerr=[np.array(means)-np.array(ci_lower), 
                                                 np.array(ci_upper)-np.array(means)], 
                 fmt='o-', color='red', capsize=5, capthick=2, markersize=8, linewidth=2,
                 label='Mean with 95% CI')
    
    plt.xlabel('Alpha Value', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Individual Scores with Mean and Confidence Intervals', fontsize=16)
    plt.xticks(range(len(alphas)), alphas)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/alpha_individual_points_ci.png', dpi=300, bbox_inches='tight')
    plt.close()

    

if __name__ == "__main__":
    """Main analysis function"""
    folder_path = "simcse_sbert_data"
    file_pattern = os.path.join(folder_path, "*.txt")

    results_df = extract_results_from_filenames(file_pattern)

    print(f"Extracted {len(results_df)} results")
    print(f"Alphas tested: {sorted(results_df['alpha'].unique())}")
    create_alpha_visualizations(results_df)