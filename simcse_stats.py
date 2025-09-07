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
    print(f"Found {len(files)} files")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")
            pattern = r"simcse_seed_(\d+)_alpha_0\.5_batch_([\d.]+)_corr_([\d.]+)\.txt"
            match = re.search(pattern, filename)
            
            if match:
                seed = int(match.group(1))
                alpha = 0.5
                batch = float(match.group(2))
                correlation = float(match.group(3))
                
                results.append({
                    'alpha': alpha,
                    'batch': batch,
                    'seed': seed,
                    'correlation': correlation,
                    'filename': filename
                })
                print(f"Successfully parsed: seed={seed}, batch={batch}, corr={correlation}")
            else:
                print(f"Could not parse filename: {filename}")
                alt_pattern = r".*seed_(\d+).*batch_([\d.]+).*corr_([\d.]+)\.txt"
                alt_match = re.search(alt_pattern, filename)
                if alt_match:
                    print(f"Alternative pattern matched: {alt_match.groups()}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Successfully parsed {len(results)} files")
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

def create_batch_visualizations(results_df, output_dir="batch_analysis_plots"):
    """
    Create visualizations for batch parameter analysis with confidence intervals
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Batch size
    batch_stats = results_df.groupby('batch')['correlation'].agg(['mean', 'std', 'count'])
    batch_stats['ci'] = 1.96 * batch_stats['std'] / np.sqrt(batch_stats['count'])

    batches = sorted(results_df['batch'].unique())
    
    # Confidence intervals
    means = []
    ci_lower = []
    ci_upper = []
    
    for batch in batches:
        data = results_df[results_df['batch'] == batch]['correlation']
        mean, lower, upper = bootstrap_confidence_interval(data)
        means.append(mean)
        ci_lower.append(lower)
        ci_upper.append(upper)

    # Plot Batch vs Scores
    plt.figure(figsize=(8, 4))

    plt.fill_between(batch_stats.index, 
                    batch_stats['mean'] - batch_stats['ci'], 
                    batch_stats['mean'] + batch_stats['ci'], 
                    alpha=0.3, color='green', label='95% CI')

    plt.plot(batch_stats.index, batch_stats['mean'], 
            'o-', color= 'green', linewidth=2, markersize=6, 
            label='Mean Correlation')
    
    plt.xscale('log')

    plt.xlabel('Number of Batches', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.title('Effect of Batch Size on STS Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.xlim(batch_stats.index.min() - 0.5, batch_stats.index.max() + 0.5)

    plt.legend(loc='best', fontsize=10)
    plt.savefig(f'{output_dir}/simcs_batch_vs_correlation_ci.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot Individual points with mean and CI
    plt.figure(figsize=(12, 8))
    
    for i, batch in enumerate(batches):
        batch_data = results_df[results_df['batch'] == batch]['correlation']
        jitter = np.random.normal(0, 0.05, len(batch_data))
        plt.scatter([i + j for j in jitter], batch_data, alpha=0.6, s=40)
    
    plt.errorbar(range(len(batches)), means, yerr=[np.array(means)-np.array(ci_lower), 
                                                  np.array(ci_upper)-np.array(means)], 
                 fmt='o-', color='red', capsize=5, capthick=2, markersize=8, linewidth=2,
                 label='Mean with 95% CI')
    
    plt.xlabel('Number of Batches', fontsize=14)
    plt.ylabel('Pearson Correlation', fontsize=14)
    plt.title('Individual Scores with Mean and Confidence Intervals (Batch Size)', fontsize=16)
    plt.xticks(range(len(batches)), [int(b) if b.is_integer() else b for b in batches])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/simcse_batch_individual_points_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
   

if __name__ == "__main__":
    """Main analysis function"""
    folder_path = "simcse_data"
    file_pattern = os.path.join(folder_path, "*.txt") 
    results_df = extract_results_from_filenames(file_pattern)
    create_batch_visualizations(results_df)