import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional, Tuple
from scipy.stats import spearmanr, pearsonr, linregress

def pairwise_ratios_and_averages(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> pd.DataFrame:
    """
    Compute all pairwise ratios and averages for a group.
    Returns dictionary with keys: log_prior_ratio, log_likelihood_ratio, log_posterior_ratio,
    log_prior_avg, log_likelihood_avg, log_posterior_avg
    """
    valid_rows = group.dropna(subset=[log_prior_col, log_likelihood_col, log_posterior_col])
    
    if len(valid_rows) < 2:
        return pd.DataFrame({
            "log_prior_ratio": [np.nan],
            "log_likelihood_ratio": [np.nan], 
            "log_posterior_ratio": [np.nan],
            "log_prior_avg": [np.nan],
            "log_likelihood_avg": [np.nan],
            "log_posterior_avg": [np.nan]
        })
    
    log_prior_ratios = []
    log_likelihood_ratios = []
    log_posterior_ratios = []
    log_prior_avgs = []
    log_likelihood_avgs = []
    log_posterior_avgs = []
    
    for row1, row2 in combinations(valid_rows.itertuples(), 2):
        row1_series = valid_rows.loc[row1.Index]
        row2_series = valid_rows.loc[row2.Index]
        
        # Calculate ratios (differences in log space)
        log_prior_ratios.append(row1_series[log_prior_col] - row2_series[log_prior_col])
        log_likelihood_ratios.append(row1_series[log_likelihood_col] - row2_series[log_likelihood_col])
        log_posterior_ratios.append(row1_series[log_posterior_col] - row2_series[log_posterior_col])
        
        # Calculate averages
        log_prior_avgs.append((row1_series[log_prior_col] + row2_series[log_prior_col]) / 2)
        log_likelihood_avgs.append((row1_series[log_likelihood_col] + row2_series[log_likelihood_col]) / 2)
        log_posterior_avgs.append((row1_series[log_posterior_col] + row2_series[log_posterior_col]) / 2)
    
    return pd.DataFrame({
        "log_prior_ratio": log_prior_ratios,
        "log_likelihood_ratio": log_likelihood_ratios, 
        "log_posterior_ratio": log_posterior_ratios,
        "log_prior_avg": log_prior_avgs,
        "log_likelihood_avg": log_likelihood_avgs,
        "log_posterior_avg": log_posterior_avgs
    })
    
    
def log_odds_update(group: pd.DataFrame, log_prior_ratio_col: str, log_posterior_ratio_col: str) -> pd.DataFrame:
    """
    Compute all pairwise log-odds updates for a group.
    Returns dictionary with keys: log_odds_update
    """
    return group[log_posterior_ratio_col] - group[log_prior_ratio_col]

def linear_regression(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Calculate correlation coefficient, p-value, and slope between two columns.
    
    Returns:
        DataFrame with columns: correlation_coefficient, p_value, slope
    """
    valid_data = df[[col1, col2]].dropna()
    
    null_data = pd.DataFrame({
        "slope": [np.nan],
        "intercept": [np.nan],
        "r_value": [np.nan],
        "p_value": [np.nan],
        "slope_stderr": [np.nan],
        "intercept_stderr": [np.nan]
    })    
    
    if len(valid_data) < 2:
        return null_data
    
    if np.std(valid_data[col1]) == 0 or np.std(valid_data[col2]) == 0:
        return null_data
    
    # Calculate slope using linear regression
    result = linregress(valid_data[col1], valid_data[col2])
    
    return pd.DataFrame({
        "slope": [result.slope],
        "intercept": [result.intercept],
        "r_value": [result.rvalue],
        "p_value": [result.pvalue],
        "slope_stderr": [result.stderr],
        "intercept_stderr": [result.intercept_stderr]
    })
    

def pairwise_bce_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str, square: bool = False) -> Optional[float]:
    """
    Calculates the pairwise Bayesian Coherence Error (BCE) in a group.

    For each pair, this is the difference between the log-odds update and
    the log-likelihood ratio. It's the residual of the log-space Bayes' rule update.
    """
    log_odds_updates, log_likelihood_ratios, valid_rows = compute_all_pairwise_updates(group, log_prior_col, log_likelihood_col, log_posterior_col)
    
    if len(valid_rows) < 2:
        return np.nan
    
    # Calculate differences between log odds updates and log likelihood ratios
    differences = log_odds_updates - log_likelihood_ratios

    errors = differences ** 2 if square else np.abs(differences)

    # Return the list itself so that Analyzer explodes it and each pair
    # contributes one data-point.  Down-stream averaging will then weight
    # pairs correctly without manual intervention.
    return errors.tolist()

def cbc_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
    """
    Calculates the Correlation-Based Coherence (CBC) for a group.
    
    CBC measures how well the log-odds updates correlate with log-likelihood ratios.
    Perfect coherence should yield correlation = 1.0.
    """
    log_odds_updates, log_likelihood_ratios, valid_rows = compute_all_pairwise_updates(group, log_prior_col, log_likelihood_col, log_posterior_col)
    
    if len(valid_rows) < 2:
        return np.nan
    
    # Need at least 2 pairs for meaningful correlation
    if len(log_odds_updates) < 2:
        return np.nan
        
    # Check for zero variance (constant values)
    if np.std(log_odds_updates) == 0 or np.std(log_likelihood_ratios) == 0:
        return np.nan # Correlation is not well-defined

    return np.corrcoef(log_odds_updates, log_likelihood_ratios)[0, 1]

def scs_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
    """
    Calculates the Sign Consistency Score (SCS) for a group.
    
    SCS measures the fraction of pairs where the sign of log-odds update 
    matches the sign of log-likelihood ratio. Perfect coherence = 1.0.
    
    For each pair (i,j):
    - u_ij = log-odds update = [log P(c_i|x,h) - log P(c_j|x,h)] - [log P(c_i|h) - log P(c_j|h)]  
    - v_ij = log-likelihood ratio = log P(x|c_i,h) - log P(x|c_j,h)
    - SCS = (1/|pairs|) * Σ 1[sign(u_ij) = sign(v_ij)]
    """
    log_odds_updates, log_likelihood_ratios, valid_rows = compute_all_pairwise_updates(group, log_prior_col, log_likelihood_col, log_posterior_col)
    
    if len(valid_rows) < 2:
        return np.nan
    
    log_odds_update_signs = np.sign(log_odds_updates)
    log_likelihood_ratio_signs = np.sign(log_likelihood_ratios)
    
    # Agreement indicator for each pair (True/False -> 1.0/0.0)
    sign_agreements = (log_odds_update_signs == log_likelihood_ratio_signs).astype(float)

    # Return list of 1/0 agreements to allow weighting by pairs automatically.
    return sign_agreements.tolist()

def compute_all_pairwise_updates(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Tuple[list, list, pd.DataFrame]:
    """
    Compute all pairwise log_odds_updates and log_likelihood_ratios for a group.
    Returns lists that can be reused across multiple metrics, plus the valid_rows DataFrame.
    """
    valid_rows = group.dropna(subset=[log_prior_col, log_likelihood_col, log_posterior_col])
    
    log_odds_updates = []
    log_likelihood_ratios = []
    
    for row1, row2 in combinations(valid_rows.itertuples(), 2):
        # Convert back to Series for easier column access
        row1_series = valid_rows.loc[row1.Index]
        row2_series = valid_rows.loc[row2.Index]
        
        log_odds_update, log_likelihood_ratio = calculate_pairwise_updates(
            row1_series, row2_series, log_prior_col, log_likelihood_col, log_posterior_col
        )
        log_odds_updates.append(log_odds_update)
        log_likelihood_ratios.append(log_likelihood_ratio)
    
    return np.array(log_odds_updates), np.array(log_likelihood_ratios), valid_rows

def calculate_pairwise_updates(row1: pd.Series, row2: pd.Series, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Tuple[float, float]:
    """
    Calculate log odds update and log likelihood ratio for a pair of rows.
    
    Args:
        row1, row2: Pandas Series representing two data points
        log_prior_col: Column name for log prior probabilities
        log_likelihood_col: Column name for log likelihood probabilities  
        log_posterior_col: Column name for log posterior probabilities
        
    Returns:
        Tuple of (log_odds_update, log_likelihood_ratio)
    """
    log_odds_update = (row1[log_posterior_col] - row2[log_posterior_col]) - (row1[log_prior_col] - row2[log_prior_col])
    log_likelihood_ratio = row1[log_likelihood_col] - row2[log_likelihood_col]
    return log_odds_update, log_likelihood_ratio

def rbc_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
    """
    Calculates the Rank-Based Coherence (RBC) for a group.
    
    RBC computes Spearman correlation between:
    - Likelihood ranks: rank(log P(x | c_i, h)) 
    - Rank changes: rank(log P(c_i | x, h)) - rank(log P(c_i | h))
    
    This measures if evidence that makes a class more likely (higher likelihood rank)
    also causes that class to gain more in posterior ranking.
    """
    valid_rows = group.dropna(subset=[log_prior_col, log_likelihood_col, log_posterior_col])
    if len(valid_rows) < 2:
        return np.nan

    prior_ranks = valid_rows[log_prior_col].rank()
    posterior_ranks = valid_rows[log_posterior_col].rank()
    likelihood_ranks = valid_rows[log_likelihood_col].rank()

    # Calculate rank changes (Δ_i = r_i^post - r_i^prior)
    rank_changes = posterior_ranks - prior_ranks
    
    # Check for constant inputs to avoid warning
    if len(np.unique(likelihood_ranks)) == 1 or len(np.unique(rank_changes)) == 1:
        return np.nan
    
    # Spearman correlation between likelihood ranks and rank changes
    corr, _ = spearmanr(likelihood_ranks, rank_changes)
    
    return corr

def nbce_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str, eps: float = 1e-8) -> Optional[float]:
    """
    Calculates the Normalized Bayesian Coherence Error (NBCE) for a group.
    
    NBCE normalizes BCE by the magnitude of the terms to make it scale-invariant.
    This addresses the reviewer concern that uniform distributions have zero BCE.
    """
    log_odds_updates, log_likelihood_ratios, valid_rows = compute_all_pairwise_updates(group, log_prior_col, log_likelihood_col, log_posterior_col)
    
    if len(valid_rows) < 2:
        return np.nan
    
    numerator = log_odds_updates - log_likelihood_ratios
    # Fixed: Use absolute values consistently for proper normalization
    denominator = np.abs(log_odds_updates) + np.abs(log_likelihood_ratios) + eps
    
    relative_errors = (numerator / denominator) ** 2
    
    # Return per-pair errors to keep weighting consistent with BCE.
    return relative_errors.tolist()