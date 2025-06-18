import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional, Tuple
from scipy.stats import spearmanr

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

def entropy_of_group(group: pd.DataFrame, log_prob_col: str) -> Optional[float]:
    """
    Calculates the entropy of a distribution represented by log probabilities.
    """
    valid_probs = group[log_prob_col].dropna()
    if len(valid_probs) == 0:
        return np.nan
    
    # Normalize log probabilities to get a proper distribution
    log_probs = valid_probs - np.logaddexp.reduce(valid_probs)
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs)

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
    
    # Spearman correlation between likelihood ranks and rank changes
    corr, _ = spearmanr(likelihood_ranks, rank_changes)
    
    return corr

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

# def single_evidence_estimate(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
#     return group[log_prior_col] + group[log_likelihood_col] - group[log_posterior_col]

# def pairwise_error_of_group(group: pd.DataFrame, value_col: str, square: bool = True) -> Optional[float]:
#     """Calculates the pairwise squared errors for a column in a group."""
#     valid_values = group[value_col].dropna()
#     if len(valid_values) < 2:
#         return np.nan # MSE requires at least 2 data points for pairs

#     if len(valid_values) > 1000:
#          print(f"Warning: Calculating pairwise MSE for a large group ({len(valid_values)} items). This might be slow.")

#     # Calculate squared differences for all combinations
#     if square:
#         error = [(v1 - v2)**2 for v1, v2 in combinations(valid_values, 2)]
#     else:
#         error = [abs(v1 - v2) for v1, v2 in combinations(valid_values, 2)]
#     return error

# def mean_pairwise_error_of_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
#     """Calculates the mean pairwise squared errors for a column in a DataFrame group."""
#     return np.mean(pairwise_error_of_group(group, value_col))