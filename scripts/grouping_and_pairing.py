# %%
# Import libraries and load data
import pandas as pd
import numpy as np
from itertools import combinations

# Load and merge logprobs data
logprob_data_paths = ["data/logprobs.csv"]
logprobs = [pd.read_csv(logprob_data_path) for logprob_data_path in logprob_data_paths]
logprobs = pd.concat(logprobs)
logprobs.reset_index(drop=True, inplace=True)

print(f"Loaded logprobs data: {len(logprobs)} rows")
print(f"Columns: {logprobs.columns.tolist()}")

# %%
# Load evals data and create lookup tables
evals_df = pd.read_parquet(
    "hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet"
)

params_lookup = (
    evals_df.drop_duplicates(subset=["fullname"])
    .set_index("fullname")["#Params (B)"]
    .to_dict()
)

selected_evals = [
    "IFEval",
    "BBH", 
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
    "Average ⬆️",
]

# Create eval lookup dictionary
eval_lookup = {}
for model_name in evals_df["fullname"].unique():
    eval_lookup[model_name] = {
        eval_name: evals_df[evals_df["fullname"] == model_name][eval_name].iloc[0]
        for eval_name in selected_evals
        if not evals_df[evals_df["fullname"] == model_name].empty
    }

print(f"Created lookup tables for {len(params_lookup)} models")

# %%
# Add columns and rename/filter data
def format_model_kwargs(kwarg_str):
    if '"revision": "step' in kwarg_str:
        step = kwarg_str.split("step")[1].split('"')[0]
        step_k = str(int(step) // 1000) + "k"
        return step_k
    else:
        return "unknown"

# Add parameter count
logprobs["#Params (B)"] = logprobs["model_name"].map(params_lookup).astype(float)

# Add eval scores
for eval_name in selected_evals:
    logprobs[eval_name] = logprobs["model_name"].map(
        lambda model: eval_lookup.get(model, {}).get(eval_name, np.nan)
    )

# Add model family
logprobs["Model Family"] = logprobs["model_name"].apply(
    lambda model_name: model_name.split("/")[1].split("-")[0]
)

# Rename columns
logprobs = logprobs.rename(columns={
    "model_name": "Language Model",
    "model_kwargs": "Training Steps"
})

# Transform values
model_family_mapping = {
    "Llama": "Llama 3",
    "gpt2": "GPT 2", 
    "pythia": "Pythia",
    "Qwen2.5": "Qwen 2.5",
    "Falcon3": "Falcon 3",
}
logprobs["Model Family"] = logprobs["Model Family"].map(model_family_mapping).fillna(logprobs["Model Family"])

logprobs["Language Model"] = logprobs["Language Model"].apply(lambda x: x.split("/")[1])
logprobs["Training Steps"] = logprobs["Training Steps"].apply(format_model_kwargs)

# Filter data
valid_families = ["Llama 3", "GPT 2", "Pythia", "Qwen 2.5", "Falcon 3"]
logprobs = logprobs[logprobs["Model Family"].isin(valid_families)]

print(f"After filtering: {len(logprobs)} rows")

# %%
# Define grouping columns and perform pairwise grouping
group_by_cols = [
    "evidence_text",          # x
    "conversation_history",   # h  
    "class_category",         # k
    "Language Model",         # m
    "Training Steps",         # t
]

def create_pairwise_grouped_data(df, group_cols):
    """
    Create pairwise grouped data where each pair of rows within a group becomes one row.
    
    Args:
        df: Input dataframe
        group_cols: Columns to group by
        value_cols: Columns to create pairwise combinations for
    
    Returns:
        DataFrame with pairwise grouped data
    """
    pairwise_rows = []
    
    # Get columns that should be inherited (constant within groups)
    inherit_cols = [col for col in df.columns if col not in group_cols]
    
    grouped = df.groupby(group_cols, observed=True)
    
    for group_name, group_df in grouped:
        # Skip groups with less than 2 rows
        if len(group_df) < 2:
            continue
            
        # Create all pairwise combinations
        for (idx1, row1), (idx2, row2) in combinations(group_df.iterrows(), 2):
            pairwise_row = {}
            
            # Add grouping columns
            if isinstance(group_name, tuple):
                for i, col in enumerate(group_cols):
                    pairwise_row[col] = group_name[i]
            else:
                pairwise_row[group_cols[0]] = group_name
            
            # Add pairwise columns as tuples
            for col in inherit_cols:
                if row1[col] == row2[col]:
                    pairwise_row[col] = row1[col]
                else:
                    pairwise_row[f"{col}_pair"] = (row1[col], row2[col])
            
            pairwise_rows.append(pairwise_row)
    
    return pd.DataFrame(pairwise_rows)

print("Creating pairwise grouped data...")
pairwise_df = create_pairwise_grouped_data(logprobs, group_by_cols)

print(f"Pairwise grouped data: {len(pairwise_df)} rows")
print(f"Columns: {pairwise_df.columns.tolist()}")

# # %%
# # Add computed metrics from the pairwise data
# def compute_pairwise_metrics(row):
#     """Compute metrics from pairwise tuples."""
#     prior_pair = row["prior_logprob_pair"]
#     likelihood_pair = row["likelihood_logprob_pair"] 
#     posterior_pair = row["posterior_logprob_pair"]
    
#     # Extract individual values
#     prior1, prior2 = prior_pair
#     likelihood1, likelihood2 = likelihood_pair
#     posterior1, posterior2 = posterior_pair
    
#     # Compute ratios (differences in log space)
#     log_prior_ratio = prior1 - prior2
#     log_likelihood_ratio = likelihood1 - likelihood2
#     log_posterior_ratio = posterior1 - posterior2
    
#     # Compute averages
#     log_prior_avg = (prior1 + prior2) / 2
#     log_likelihood_avg = (likelihood1 + likelihood2) / 2
#     log_posterior_avg = (posterior1 + posterior2) / 2
    
#     # Compute log odds update
#     log_odds_update = log_posterior_ratio - log_prior_ratio
    
#     return pd.Series({
#         "log_prior_ratio": log_prior_ratio,
#         "log_likelihood_ratio": log_likelihood_ratio,
#         "log_posterior_ratio": log_posterior_ratio,
#         "log_prior_avg": log_prior_avg,
#         "log_likelihood_avg": log_likelihood_avg,
#         "log_posterior_avg": log_posterior_avg,
#         "log_odds_update": log_odds_update
#     })

# print("Computing pairwise metrics...")
# metrics = pairwise_df.apply(compute_pairwise_metrics, axis=1)
# pairwise_df = pd.concat([pairwise_df, metrics], axis=1)

# print(f"Added metrics. Final shape: {pairwise_df.shape}")

# %%
# Create filename based on group_by_cols first letters
group_letters = {
    "evidence_text": "x",
    "conversation_history": "h", 
    "class_category": "k",
    "Language Model": "m",
    "Training Steps": "t"
}

filename_suffix = "".join([group_letters[col] for col in group_by_cols])
output_filename = f"data/pairwise_logprobs_groupedby_{filename_suffix}.csv"

print(f"Saving to {output_filename}...")
pairwise_df.to_csv(output_filename, index=False)

print(f"Saved {len(pairwise_df)} pairwise rows to {output_filename}")
print(f"Final columns: {pairwise_df.columns.tolist()}")

# %%
# Display summary statistics
print("\nSummary Statistics:")
print("=" * 50)
print(f"Total pairwise comparisons: {len(pairwise_df):,}")
print(f"Unique models: {pairwise_df['Language Model'].nunique()}")
print(f"Unique model families: {pairwise_df['Model Family'].nunique()}")
print(f"Model families: {pairwise_df['Model Family'].unique().tolist()}")
# %%