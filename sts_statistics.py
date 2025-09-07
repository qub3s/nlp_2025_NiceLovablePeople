import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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

    plt.figure(figsize=(8, 5))

    colors = ['red', 'purple', 'blue']
    cmap = LinearSegmentedColormap.from_list('red_blue', colors, N=100)

    x_continuous = np.linspace(alpha_stats.index.min(), alpha_stats.index.max(), 1000)
    mean_continuous = np.interp(x_continuous, alpha_stats.index, alpha_stats['mean'])
    ci_continuous = np.interp(x_continuous, alpha_stats.index, alpha_stats['ci'])

    for i in range(len(x_continuous) - 1):
        x1, x2 = x_continuous[i], x_continuous[i + 1]
        y1_low = mean_continuous[i] - ci_continuous[i]
        y1_high = mean_continuous[i] + ci_continuous[i]
        y2_low = mean_continuous[i + 1] - ci_continuous[i + 1]
        y2_high = mean_continuous[i + 1] + ci_continuous[i + 1]
        
        color = cmap((x1 + x2) / 2)
        
        plt.fill_between([x1, x2], [y1_low, y2_low], [y1_high, y2_high], 
                        alpha=0.3, color=color, edgecolor=None)

    for i in range(len(alpha_stats.index) - 1):
        alpha_val1 = alpha_stats.index[i]
        alpha_val2 = alpha_stats.index[i + 1]
        color = cmap((alpha_val1 + alpha_val2) / 2)
        
        plt.plot([alpha_val1, alpha_val2], 
                [alpha_stats.loc[alpha_val1, 'mean'], alpha_stats.loc[alpha_val2, 'mean']], 
                '-', color=color, linewidth=2, label='Mean Correlation' if i == 0 else "")

    for i, alpha_val in enumerate(alpha_stats.index):
        color = cmap(alpha_val)
        plt.plot(alpha_val, alpha_stats.loc[alpha_val, 'mean'], 
                'o', color=color, markersize=6, markeredgecolor='white', linewidth=1,
                label='Data Points' if i == 0 else "")

    plt.xlabel('Alpha Value', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.title('Effect of Alpha Parameter on STS Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.xlim(alpha_stats.index.min() - 0.01, alpha_stats.index.max() + 0.01)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Alpha Value', fontsize=10)



    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Mean Correlation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
            markersize=6, label='Data Points'),
        Line2D([0], [0], color='black', linewidth=8, alpha=0.3, label='95% CI')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.savefig(f'{output_dir}/alpha_vs_correlation_ci.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()



    # Window
    plt.figure(figsize=(3, 4))

    colors = ['red', 'purple', 'blue']
    cmap = LinearSegmentedColormap.from_list('red_blue', colors, N=100)

    x_continuous = np.linspace(alpha_stats.index.min(), alpha_stats.index.max(), 1000)
    mean_continuous = np.interp(x_continuous, alpha_stats.index, alpha_stats['mean'])
    ci_continuous = np.interp(x_continuous, alpha_stats.index, alpha_stats['ci'])

    for i in range(len(x_continuous) - 1):
        x1, x2 = x_continuous[i], x_continuous[i + 1]
        y1_low = mean_continuous[i] - ci_continuous[i]
        y1_high = mean_continuous[i] + ci_continuous[i]
        y2_low = mean_continuous[i + 1] - ci_continuous[i + 1]
        y2_high = mean_continuous[i + 1] + ci_continuous[i + 1]
        
        color = cmap((x1 + x2) / 2)
        
        plt.fill_between([x1, x2], [y1_low, y2_low], [y1_high, y2_high], 
                        alpha=0.3, color=color, edgecolor=None)

    for i in range(len(alpha_stats.index) - 1):
        alpha_val1 = alpha_stats.index[i]
        alpha_val2 = alpha_stats.index[i + 1]
        color = cmap((alpha_val1 + alpha_val2) / 2)
        
        plt.plot([alpha_val1, alpha_val2], 
                [alpha_stats.loc[alpha_val1, 'mean'], alpha_stats.loc[alpha_val2, 'mean']], 
                '-', color=color, linewidth=2, label='Mean Correlation' if i == 0 else "")

    for i, alpha_val in enumerate(alpha_stats.index):
        color = cmap(alpha_val)
        plt.plot(alpha_val, alpha_stats.loc[alpha_val, 'mean'], 
                'o', color=color, markersize=6, markeredgecolor='white', linewidth=1,
                label='Data Points' if i == 0 else "")

    plt.xlabel('Alpha Value', fontsize=12)
    plt.ylabel('Pearson Correlation', fontsize=12)
    plt.tight_layout()

    plt.xlim(0.9 - 0.01, alpha_stats.index.max() + 0.01)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])

    plt.savefig(f'{output_dir}/alpha_vs_correlation_ci_window.png', 
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