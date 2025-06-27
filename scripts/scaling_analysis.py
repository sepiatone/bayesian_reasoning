# %%
# Import libraries and load pairwise logprobs data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Load pairwise logprobs data
print("Loading pairwise logprobs data...")
pairwise_df = pd.read_csv("data/pairwise_logprobs_groupedby_xhkmt.csv")
print(f"Loaded {len(pairwise_df)} pairwise rows")
print(f"Columns: {pairwise_df.columns.tolist()}")

# %%
# Calculate ratios from logprob pairs
def extract_pair_values(pair_str):
    """Extract tuple values from string representation"""
    if pd.isna(pair_str):
        return np.nan, np.nan
    try:
        # Remove parentheses and split by comma
        clean_str = pair_str.strip('()').replace("'", "").replace('"', '')
        val1, val2 = [float(x.strip()) for x in clean_str.split(',')]
        return val1, val2
    except:
        return np.nan, np.nan

print("Calculating logprob ratios...")

# Extract individual values from pairs
prior_vals = pairwise_df['prior_logprob_pair'].apply(extract_pair_values)
likelihood_vals = pairwise_df['likelihood_logprob_pair'].apply(extract_pair_values)
posterior_vals = pairwise_df['posterior_logprob_pair'].apply(extract_pair_values)

# Calculate ratios (differences in log space)
pairwise_df['log_prior_ratio'] = [p[0] - p[1] if not pd.isna(p[0]) else np.nan for p in prior_vals]
pairwise_df['log_likelihood_ratio'] = [l[0] - l[1] if not pd.isna(l[0]) else np.nan for l in likelihood_vals]
pairwise_df['log_posterior_ratio'] = [p[0] - p[1] if not pd.isna(p[0]) else np.nan for p in posterior_vals]

# Calculate log odds update
pairwise_df['log_odds_update'] = pairwise_df['log_posterior_ratio'] - pairwise_df['log_prior_ratio']

print(f"Calculated ratios for {len(pairwise_df)} rows")
print(f"Log odds update range: {pairwise_df['log_odds_update'].min():.3f} to {pairwise_df['log_odds_update'].max():.3f}")

# %%
# Filter for only 143k checkpoint of Pythia models and prepare data for main plots
print("Filtering data for main plots...")

# For main plots (1-3), only keep 143k checkpoint of Pythia models
filtered_df = pairwise_df.copy()
pythia_mask = (filtered_df['Model Family'] == 'Pythia') & (filtered_df['Training Steps'] != '143k')
filtered_df = filtered_df[~pythia_mask]

print(f"After filtering Pythia to 143k only: {len(filtered_df)} rows")

# Remove rows with NaN values in key metrics
clean_df = filtered_df.dropna(subset=['log_likelihood_ratio', 'log_odds_update']).copy()
print(f"After removing NaN values: {len(clean_df)} rows")

# For Plot 4, keep all Pythia training steps (not just 143k)
print("Preparing data for training steps analysis...")
pythia_all_steps_df = pairwise_df[pairwise_df['Model Family'] == 'Pythia'].copy()
pythia_all_steps_df = pythia_all_steps_df.dropna(subset=['log_likelihood_ratio', 'log_odds_update'])
print(f"Pythia all steps data: {len(pythia_all_steps_df)} rows")

# %%
# Plot 1: Log Odds Update vs Log Likelihood Ratio by Model and Family
print("Creating Plot 1: Log Odds Update vs Log Likelihood Ratio...")

# Organize models by family
model_families = {}
for family in clean_df["Model Family"].unique():
    family_data = clean_df[clean_df["Model Family"] == family]
    # Sort models by parameter count
    models = sorted(family_data["Language Model"].unique(),
                   key=lambda x: family_data[family_data["Language Model"] == x]["#Params (B)"].iloc[0])
    model_families[family] = models

print("Model organization:")
for family, models in model_families.items():
    print(f"{family}: {models}")

# Create subplot grid
n_families = len(model_families)
max_models = max(len(models) for models in model_families.values())

fig, axes = plt.subplots(n_families, max_models, figsize=(5*max_models, 3.5*n_families))
if n_families == 1:
    axes = axes.reshape(1, -1)
elif max_models == 1:
    axes = axes.reshape(-1, 1)

def add_scatter_stats(ax, x, y, model_name):
    """Add statistical annotations to scatter plot."""
    if len(x) < 2:
        ax.text(0.05, 0.95, f"N = {len(x)}\nInsufficient data",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)
        return None
    
    # Calculate statistics
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Plot line of best fit
    x_range = np.linspace(min(x), max(x), 100)
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, 'r-', linewidth=2, alpha=0.8)
    
    # Add diagonal line (perfect coherence)
    ax_min = min(min(x), min(y))
    ax_max = max(max(x), max(y))
    ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', alpha=0.5, linewidth=1)
    
    # Add axis lines (x=0 and y=0)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)
    
    # Add statistics text (removed p-value)
    stats_text = f"N = {len(x):,}\nr = {r_value:.3f}"
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    ax.text(0.65, 0.05, stats_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)
    
    return r_value, slope, p_value

# Store results for later analysis
coherence_results = []

# Create consistent color mapping for model families
all_families = clean_df['Model Family'].unique()
family_colors = dict(zip(all_families, plt.cm.tab10(np.linspace(0, 1, len(all_families)))))

for row_idx, (family, models) in enumerate(model_families.items()):
    for col_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]
        
        # Get data for this model
        model_data = clean_df[clean_df["Language Model"] == model]
        
        if len(model_data) > 0:
            x = model_data["log_likelihood_ratio"].values
            y = model_data["log_odds_update"].values
            
            # Create scatter plot with family-consistent colors
            family = model_data['Model Family'].iloc[0]
            ax.scatter(x, y, alpha=0.6, s=20, color=family_colors[family])
            stats = add_scatter_stats(ax, x, y, model)
            
            if stats:
                r_value, slope, p_value = stats
                # Store results
                model_info = model_data.iloc[0]
                coherence_results.append({
                    'Language Model': model,
                    'Model Family': family,
                    '#Params (B)': model_info['#Params (B)'],
                    'Training Steps': model_info['Training Steps'],
                    'r_value': r_value,
                    'slope': slope,
                    'p_value': p_value,
                    'n_points': len(model_data),
                    **{col: model_info[col] for col in ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO', 'Average ⬆️'] if col in model_info}
                })
        else:
            ax.text(0.5, 0.5, f"No data\n{model}", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(model, fontsize=14)
        
        ax.set_xlabel('Log Likelihood Ratio', fontsize=12)
        ax.set_ylabel('Log Odds Update', fontsize=12)
        # Remove grid
        # ax.grid(True, alpha=0.3)
    
    # Hide unused subplots in this row
    for empty_idx in range(len(models), max_models):
        axes[row_idx, empty_idx].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig('results/plots/new/log_odds_vs_likelihood_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

# Convert results to DataFrame
coherence_df = pd.DataFrame(coherence_results)
print(f"Collected coherence results for {len(coherence_df)} models")

# %%
# Plot 2: Bayesian Coherence Coefficient vs Benchmark Scores
print("Creating Plot 2: BCC vs Benchmark Scores...")

benchmarks = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
n_benchmarks = len(benchmarks)

fig, axes = plt.subplots(1, n_benchmarks, figsize=(3.5*n_benchmarks, 5))

for i, benchmark in enumerate(benchmarks):
    ax = axes[i]
    
    # Filter out NaN values for this benchmark
    valid_data = coherence_df.dropna(subset=[benchmark, 'r_value'])
    
    if len(valid_data) > 0:
        x = valid_data[benchmark]
        y = valid_data['r_value']
        
        # Color by model family
        families = valid_data['Model Family'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
        
        for family, color in zip(families, colors):
            family_data = valid_data[valid_data['Model Family'] == family]
            ax.scatter(family_data[benchmark], family_data['r_value'], 
                      label=family, alpha=0.7, s=50, color=color)
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=1)
            
            # Calculate correlation and add p-value
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(x, y)
            ax.text(0.35, 0.05, f"r = {corr:.3f}\np = {p_value:.2e}\nN = {len(valid_data)}", 
                   transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)
    
    ax.set_xlabel(benchmark, fontsize=12)
    if i == 0:
        ax.set_ylabel('Bayesian Coherence Coefficient', fontsize=12)
    # Remove subplot titles
    # ax.set_title(f'BCC vs {benchmark}')
    # Remove grid
    # ax.grid(True, alpha=0.3)

# Move legend to the right side
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.94, 0.5), loc='center left', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, right=0.92)  # Increase spacing and adjust right margin for legend
plt.savefig('results/plots/new/bcc_vs_benchmarks.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Plot 3: Bayesian Coherence Coefficient vs Model Parameters (log scale)
print("Creating Plot 3: BCC vs Model Parameters...")

plt.figure(figsize=(10, 6))

# Filter out NaN values
valid_data = coherence_df.dropna(subset=['#Params (B)', 'r_value'])

if len(valid_data) > 0:
    families = valid_data['Model Family'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for family, color in zip(families, colors):
        family_data = valid_data[valid_data['Model Family'] == family]
        plt.scatter(family_data['#Params (B)'], family_data['r_value'], 
                   label=family, alpha=0.7, s=50, color=color)
    
    # Add trend line in log space
    log_params = np.log10(valid_data['#Params (B)'])
    z = np.polyfit(log_params, valid_data['r_value'], 1)
    p = np.poly1d(z)
    x_trend_log = np.linspace(log_params.min(), log_params.max(), 100)
    x_trend = 10**x_trend_log
    plt.plot(x_trend, p(x_trend_log), "k--", alpha=0.8, linewidth=2)
    
    # Calculate correlation and add p-value in log space
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(log_params, valid_data['r_value'])
    plt.text(0.75, 0.05, f"r = {corr:.3f} (log scale)\np = {p_value:.2e}\nN = {len(valid_data)}", 
           transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)

plt.xscale('log')
plt.ylabel('Bayesian Coherence Coefficient', fontsize=12)
plt.legend(fontsize=11)
# Remove grid
# plt.grid(True, alpha=0.3)
plt.tight_layout()

# Move xlabel to bottom after setting log scale
plt.xlabel('Model Parameters (B)', fontsize=12)

plt.savefig('results/plots/new/bcc_vs_parameters.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Collect coherence results for all Pythia training steps
print("Collecting coherence results for all Pythia training steps...")

# Add results for all Pythia training steps to coherence_results
pythia_training_results = []

for model in pythia_all_steps_df['Language Model'].unique():
    for step in pythia_all_steps_df[pythia_all_steps_df['Language Model'] == model]['Training Steps'].unique():
        model_step_data = pythia_all_steps_df[
            (pythia_all_steps_df['Language Model'] == model) & 
            (pythia_all_steps_df['Training Steps'] == step)
        ]
        
        if len(model_step_data) >= 2:  # Need at least 2 points for regression
            x = model_step_data["log_likelihood_ratio"].values
            y = model_step_data["log_odds_update"].values
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            model_info = model_step_data.iloc[0]
            pythia_training_results.append({
                'Language Model': model,
                'Model Family': 'Pythia',
                '#Params (B)': model_info['#Params (B)'],
                'Training Steps': step,
                'r_value': r_value,
                'slope': slope,
                'p_value': p_value,
                'n_points': len(model_step_data),
                **{col: model_info[col] for col in ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO', 'Average ⬆️'] if col in model_info}
            })

print(f"Collected {len(pythia_training_results)} Pythia training step results")

# %%
# Plot 4: Training Data Analysis (for Pythia models - all training steps)
print("Creating Plot 4: BCC vs Training Steps (Pythia all steps)...")

if len(pythia_training_results) > 0:
    pythia_results_df = pd.DataFrame(pythia_training_results)
    
    # Filter out 'unknown' training steps and convert to numeric
    valid_steps_data = pythia_results_df[pythia_results_df['Training Steps'] != 'unknown'].copy()
    if len(valid_steps_data) > 0:
        valid_steps_data['Training Steps Numeric'] = valid_steps_data['Training Steps'].str.replace('k', '').astype(float)
        
        plt.figure(figsize=(10, 6))
        
        # Get unique models and assign colors (sorted by parameter count descending)
        models = sorted(valid_steps_data['Language Model'].unique(), 
                       key=lambda x: valid_steps_data[valid_steps_data['Language Model'] == x]['#Params (B)'].iloc[0], 
                       reverse=True)  # Descending order
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for model, color in zip(models, colors):
            model_data = valid_steps_data[valid_steps_data['Language Model'] == model]
            params = model_data['#Params (B)'].iloc[0]
            plt.scatter(model_data['Training Steps Numeric'], model_data['r_value'], 
                       label=f'{model} ({params}B)', alpha=0.7, s=50, color=color)
            
            # Add connecting lines for each model
            if len(model_data) > 1:
                sorted_data = model_data.sort_values('Training Steps Numeric')
                plt.plot(sorted_data['Training Steps Numeric'], sorted_data['r_value'], 
                        color=color, alpha=0.5, linewidth=1)
        
        # Add overall trend line in log space
        if len(valid_steps_data) > 1:
            log_steps = np.log10(valid_steps_data['Training Steps Numeric'])
            z = np.polyfit(log_steps, valid_steps_data['r_value'], 1)
            p = np.poly1d(z)
            x_trend_log = np.linspace(log_steps.min(), log_steps.max(), 100)
            x_trend = 10**x_trend_log
            plt.plot(x_trend, p(x_trend_log), "k--", alpha=0.8, linewidth=2)
            
            # Calculate correlation and add p-value
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(log_steps, valid_steps_data['r_value'])
            plt.text(0.75, 0.05, f"r = {corr:.3f} (log scale)\np = {p_value:.2e}\nN = {len(valid_steps_data)}", 
                   transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=12)
        
        plt.xscale('log')
        plt.ylabel('Bayesian Coherence Coefficient', fontsize=12)
        plt.legend(fontsize=11)
        # Remove grid
        # plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Move xlabel to bottom after setting log scale
        plt.xlabel('Training Steps (thousands)', fontsize=12)
        
        plt.savefig('results/plots/new/bcc_vs_training_steps.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No valid training steps data for analysis")
else:
    print("No Pythia training data available for analysis")

# %%
# GPT-2 Focused Subplot: Just GPT-2 and GPT-2 XL
print("Creating GPT-2 focused subplot (GPT-2 and GPT-2 XL only)...")

# Filter for just GPT-2 and GPT-2 XL
gpt2_models = ['gpt2', 'gpt2-xl']
gpt2_df = clean_df[clean_df['Language Model'].isin(gpt2_models)].copy()

if len(gpt2_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Use consistent colors from the family color mapping
    gpt2_color = family_colors['GPT 2']
    
    for i, model in enumerate(gpt2_models):
        ax = axes[i]
        
        # Get data for this model
        model_data = gpt2_df[gpt2_df["Language Model"] == model]
        
        if len(model_data) > 0:
            x = model_data["log_likelihood_ratio"].values
            y = model_data["log_odds_update"].values
            
            # Create scatter plot with consistent GPT-2 family color
            ax.scatter(x, y, alpha=0.6, s=20, color=gpt2_color)
            stats = add_scatter_stats(ax, x, y, model)
        else:
            ax.text(0.5, 0.5, f"No data\n{model}", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(model, fontsize=14)
        
        ax.set_xlabel('Log Likelihood Ratio', fontsize=12)
        ax.set_ylabel('Log Odds Update', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/new/gpt2_focused_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"GPT-2 focused subplot saved with {len(gpt2_df)} data points")
else:
    print("No GPT-2 data available for focused subplot")

# %%
# Summary statistics
print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)

print(f"Total models analyzed: {len(coherence_df)}")
print(f"Model families: {coherence_df['Model Family'].unique()}")

print(f"\nBayesian Coherence Coefficient (r_value) statistics:")
print(f"Mean: {coherence_df['r_value'].mean():.3f}")
print(f"Std: {coherence_df['r_value'].std():.3f}")
print(f"Min: {coherence_df['r_value'].min():.3f}")
print(f"Max: {coherence_df['r_value'].max():.3f}")

print(f"\nTop 5 models by BCC:")
top_models = coherence_df.nlargest(5, 'r_value')[['Language Model', 'Model Family', '#Params (B)', 'r_value']]
print(top_models.to_string(index=False))

print(f"\nCorrelations with BCC:")
for benchmark in benchmarks:
    if benchmark in coherence_df.columns:
        corr = coherence_df[benchmark].corr(coherence_df['r_value'])
        if not np.isnan(corr):
            print(f"{benchmark}: r = {corr:.3f}")

# Parameter correlation (log scale)
if '#Params (B)' in coherence_df.columns:
    log_params = np.log10(coherence_df['#Params (B)'].dropna())
    valid_r = coherence_df.dropna(subset=['#Params (B)'])['r_value']
    if len(log_params) == len(valid_r):
        param_corr = np.corrcoef(log_params, valid_r)[0,1]
        print(f"Log Parameters: r = {param_corr:.3f}")

# Save results
coherence_df.to_csv('results/coherence_analysis_results.csv', index=False)
print(f"\nResults saved to results/coherence_analysis_results.csv")