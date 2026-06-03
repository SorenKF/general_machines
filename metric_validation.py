# %%

import os
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
import argparse

# %% imports
import pandas as pd
import numpy as np
from scipy import stats

from tqdm.auto import tqdm

from generalization_metric import (
    count_clause_tags,
    extract_clause_tags,
    compute_weighted_avg,
    score_all_texts_basic,
    TAG_WEIGHTS,
)

from pprint import pprint


# %%
TAG_WEIGHTS_INVERTED_GROUPS = {
    "##BOUNDED EVENT (SPECIFIC)": 1.0,
    "##BOUNDED EVENT (GENERIC)": 0.5,
    "##UNBOUNDED EVENT (SPECIFIC)": 1.0,
    "##UNBOUNDED EVENT (GENERIC)": 0.5,
    "##BASIC STATE": 1.0,
    "##COERCED STATE (SPECIFIC)": 1.0,
    "##COERCED STATE (GENERIC)": 0.5,
    "##PERFECT COERCED STATE (SPECIFIC)": 1.0,
    "##PERFECT COERCED STATE (GENERIC)": 0.5,
    "##GENERIC SENTENCE (DYNAMIC)": 0.0,
    "##GENERIC SENTENCE (STATIC)": 0.0,
    "##GENERIC SENTENCE (HABITUAL)": 0.0,
    "##GENERALIZING SENTENCE (DYNAMIC)": 1.0,
    "##GENERALIZING SENTENCE (STATIVE)": 1.0,
    "##OTHER": 1.0,
    "##IMPERATIVE": 1.0,
    "##QUESTION": 1.0,
}
# %%[markdown]
### Get label stats
# %%
TEST_INPUT = [
    "./data/outfox_augmented.csv",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_tag_freq_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tag frequency statistics for a dataset.

    Args:
        df: DataFrame with tagged text data

    Returns:
        DataFrame with tag frequency statistics (count, percentage, z-score, rank)
    """
    df_ctags = extract_clause_tags(df)
    ctags_freqs = count_clause_tags(df_ctags)

    df_ctags_freqs = pd.DataFrame.from_dict(
        ctags_freqs, orient="index", columns=["count"]
    )

    ctags_tags_total = df_ctags_freqs["count"].sum()
    ctags_tags_mean = df_ctags_freqs["count"].mean()
    ctags_tags_std = df_ctags_freqs["count"].std()

    df_ctags_freqs["freq_percentage"] = (
        (df_ctags_freqs["count"] / ctags_tags_total).mul(100).round(1)
    )

    # deviation from mean
    df_ctags_freqs["freq_deviation_from_mean"] = (
        df_ctags_freqs["count"] - ctags_tags_mean
    )

    # Z-score (how many std deviations from mean)
    df_ctags_freqs["freq_z_score"] = (
        df_ctags_freqs["count"] - ctags_tags_mean
    ) / ctags_tags_std

    # Rank by frequency
    df_ctags_freqs["frequency_rank"] = (
        df_ctags_freqs["count"].rank(ascending=False, method="first").astype(int)
    )

    return df_ctags_freqs


def get_baseline_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Get baseline genericity scores for all texts using specified weights.

    Args:
        df: DataFrame with tagged text data
        weights: Weight dict to use (defaults to TAG_WEIGHTS)

    Returns:
        DataFrame with scored texts including genericity_score column
    """
    if weights is None:
        weights = TAG_WEIGHTS
    return score_all_texts_basic(df, weights=weights)


def compare_with_baseline(
    baseline_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    scheme_name: str = "alternative",
) -> tuple[pd.DataFrame, dict]:
    """
    Compare alternative scores against baseline and compute delta statistics.

    Delta is computed as (baseline - alt), so:
    - Positive delta = baseline score was higher than alternative
    - Negative delta = alternative score was higher than baseline

    Args:
        baseline_df: DataFrame with baseline scores (must have 'genericity_score' column)
        alt_df: DataFrame with alternative scores (must have 'genericity_score' column)
        scheme_name: Name for the alternative scheme (for labeling)

    Returns:
        Tuple of (merged DataFrame with deltas, summary statistics dict)
    """
    # Identify common columns for merging (excluding the score column)
    common_cols = [
        c
        for c in baseline_df.columns
        if c in alt_df.columns and c != "genericity_score"
    ]

    merged = baseline_df.rename(columns={"genericity_score": "score_baseline"}).merge(
        alt_df.rename(columns={"genericity_score": "score_alt"}),
        on=common_cols,
        how="inner",
        suffixes=("_base", "_alt"),
    )

    # Delta = baseline - alt (positive means baseline was higher)
    merged["alt_delta_baseline"] = merged["score_baseline"] - merged["score_alt"]

    n_texts = len(merged)
    n_increased = (merged["alt_delta_baseline"] < 0).sum()  # alt > baseline
    n_decreased = (merged["alt_delta_baseline"] > 0).sum()  # alt < baseline
    n_unchanged = (merged["alt_delta_baseline"] == 0).sum()

    summary = {
        "scheme": scheme_name,
        "n_texts": n_texts,
        "mean_baseline": merged["score_baseline"].mean(),
        "mean_alt": merged["score_alt"].mean(),
        "mean_alt_delta_baseline": merged["alt_delta_baseline"].mean(),
        "std_delta": merged["alt_delta_baseline"].std(),
        "max_delta": merged["alt_delta_baseline"].max(),
        "min_delta": merged["alt_delta_baseline"].min(),
        "n_texts_increased": int(n_increased),
        "n_texts_decreased": int(n_decreased),
        "n_texts_unchanged": int(n_unchanged),
        "pct_texts_increased": n_increased / n_texts * 100,
        "pct_texts_decreased": n_decreased / n_texts * 100,
    }

    return merged, summary


def invert_freq_weights(
    counts: dict,
    eps_count: float = 1e-8,
    eps_norm: float = 1e-12,
    min_weight: float = 0.1,
) -> dict:
    """
    Compute inverse-frequency weights from tag counts. Inverse class frequency weighting based on this. NB: different from the manual per-group inverted weights in TAG_WEIGHTS_INVERTED_GROUPS. Based on the reciprocal of the absolute frequencies, normalized and scaled to prevent weight collapse.

    Args:
        counts: dict of tag -> count
        eps_count: epsilon added to counts before inversion (prevents div-by-zero for missing tags)
        eps_norm: epsilon for normalization denominator (prevents div-by-zero when all counts equal)
        min_weight: floor for normalized weights (prevents most-frequent tag from getting weight 0)

    Returns:
        dict of tag -> weight in [min_weight, 1.0]
    """
    # Step 1: Invert frequencies (rare tags get higher values)
    inv = 1.0 / (np.array(list(counts.values()), dtype=float) + eps_count)

    # Step 2: Normalize to [0, 1]
    inv_norm = (inv - inv.min()) / (inv.max() - inv.min() + eps_norm)

    # Step 3: Scale to [min_weight, 1.0] to prevent weight collapse
    inv_scaled = inv_norm * (1.0 - min_weight) + min_weight

    return {tag: float(inv_scaled[i]) for i, tag in enumerate(counts.keys())}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def run_tag_stats(dataset_path: str) -> dict:
    """
    Compute tag frequency statistics for a dataset.

    Args:
        dataset_path: Path to the dataset CSV

    Returns:
        Harmonized result dict with test name, dataset name, and result DataFrame
    """
    dataset = os.path.splitext(os.path.basename(dataset_path))[0]
    df = pd.read_csv(dataset_path)

    result_df = get_tag_freq_stats(df)
    result_df = result_df.reset_index().rename(columns={"index": "tag"})

    return {
        "test": "tag_stats",
        "dataset": dataset,
        "result": result_df,
    }


def run_ablation(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_df: pd.DataFrame,
    show_progress: bool = False,
) -> dict:
    """
    Leave-one-out ablation test: score texts with each tag excluded.

    Args:
        df: DataFrame with tagged text data
        dataset_name: Name of the dataset (for labeling results)
        baseline_df: Pre-computed baseline scores DataFrame
        show_progress: Whether to show progress bar

    Returns:
        Harmonized result dict with test name, dataset name, and result DataFrame
    """
    baseline_avg = baseline_df["genericity_score"].mean()

    # Get tag frequency stats for merging later
    tag_freq_stats = get_tag_freq_stats(df)

    # Run leave-one-out for each tag
    scoring_rounds = []
    ablation_alt_scores = {}  # Store per-text scores for each excluded tag
    label_weights = list(TAG_WEIGHTS.items())

    if show_progress:
        label_weights = tqdm(
            label_weights, desc=f"leave-one-out ({dataset_name})", unit="tag"
        )

    for label, weight in label_weights:
        remainder = TAG_WEIGHTS.copy()
        remainder.pop(label, None)

        assert label not in remainder.keys(), f"{label} is still in tag:weights dict!"

        scored_texts = score_all_texts_basic(df, weights=remainder)
        scoring_rounds.append(
            {
                "excluded_tag": label,
                "gen_score_mean": scored_texts["genericity_score"].mean().round(4),
            }
        )
        # Store per-text alt_scores for this ablation round
        ablation_alt_scores[label] = scored_texts

    # Build result DataFrame
    df_scoring_rounds = pd.DataFrame(scoring_rounds)

    # Compute statistics across ablations
    global_gen_score_std = df_scoring_rounds["gen_score_mean"].std()
    mean_of_means = df_scoring_rounds["gen_score_mean"].mean()

    df_scoring_rounds["baseline_mean"] = baseline_avg
    df_scoring_rounds["gen_score_delta_baseline"] = (
        df_scoring_rounds["gen_score_mean"] - baseline_avg
    )
    df_scoring_rounds["deviation_from_mean_of_means"] = (
        df_scoring_rounds["gen_score_mean"] - mean_of_means
    )
    df_scoring_rounds["z_score_across_ablations"] = (
        df_scoring_rounds["gen_score_mean"] - mean_of_means
    ) / global_gen_score_std

    # Merge with tag frequency stats
    result_df = pd.merge(
        df_scoring_rounds,
        tag_freq_stats.reset_index().rename(columns={"index": "excluded_tag"}),
        on="excluded_tag",
        how="left",
    )

    # Fill NaNs for tags not in data
    result_df["count"] = result_df["count"].fillna(0).astype(int)
    result_df["freq_percentage"] = result_df["freq_percentage"].fillna(0.0)

    return {
        "test": "ablation",
        "dataset": dataset_name,
        "result": result_df,
        "ablation_alt_scores": ablation_alt_scores,
    }


def run_uniform_value(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_df: pd.DataFrame,
    target_value: float = 0.5,
) -> dict:
    """
    Test with uniform weight value for all tags.

    Args:
        df: DataFrame with tagged text data
        dataset_name: Name of the dataset (for labeling results)
        baseline_df: Pre-computed baseline scores DataFrame
        target_value: Uniform weight value to apply to all tags

    Returns:
        Harmonized result dict with test name, dataset name, and result DataFrame
    """
    # Score with uniform weights
    weights_uniform = {k: target_value for k in TAG_WEIGHTS}
    alt_scores = score_all_texts_basic(df, weights=weights_uniform)

    # Compare with baseline
    result_df, summary = compare_with_baseline(
        baseline_df, alt_scores, scheme_name=f"uniform_{target_value}"
    )

    # Create summary DataFrame
    summary_df = pd.DataFrame([summary])

    return {
        "test": f"uniform_value_{target_value}",
        "dataset": dataset_name,
        "result": result_df,
        "summary": summary_df,
        "alt_scores": alt_scores,
    }


def run_inverse_weight_freqs(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_df: pd.DataFrame,
) -> dict:
    """
    Test with inverse-frequency weighted tags (weights derived from tag frequencies).

    Args:
        df: DataFrame with tagged text data
        dataset_name: Name of the dataset (for labeling results)
        baseline_df: Pre-computed baseline scores DataFrame

    Returns:
        Harmonized result dict with test name, dataset name, and result DataFrame
    """
    # Compute inverse frequency weights
    ctags = extract_clause_tags(df)
    ctags_counts = count_clause_tags(ctags)
    alt_weights = invert_freq_weights(ctags_counts)

    # Score with inverse weights
    alt_scores = score_all_texts_basic(df, weights=alt_weights)

    # Compare with baseline
    result_df, summary = compare_with_baseline(
        baseline_df, alt_scores, scheme_name="inverse_freq"
    )

    # Create summary DataFrame
    summary_df = pd.DataFrame([summary])

    return {
        "test": "inverse_weight_freqs",
        "dataset": dataset_name,
        "result": result_df,
        "summary": summary_df,
        "alt_scores": alt_scores,
    }


def run_inverse_weights_groups(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_df: pd.DataFrame,
) -> dict:
    """
    Test with predefined inverted group weights (TAG_WEIGHTS_INVERTED_GROUPS).

    Args:
        df: DataFrame with tagged text data
        dataset_name: Name of the dataset (for labeling results)
        baseline_df: Pre-computed baseline scores DataFrame

    Returns:
        Harmonized result dict with test name, dataset name, result DataFrame,
        and additional DataFrames for plotting (baseline_scores, alt_scores)
    """
    # Score with inverted group weights
    alt_scores = score_all_texts_basic(
        df,
        weights=TAG_WEIGHTS_INVERTED_GROUPS,
    )

    # Compare with baseline
    result_df, summary = compare_with_baseline(
        baseline_df, alt_scores, scheme_name="inverted_groups"
    )

    # Create summary DataFrame
    summary_df = pd.DataFrame([summary])

    return {
        "test": "inverse_weights_groups",
        "dataset": dataset_name,
        "result": result_df,
        "summary": summary_df,
        "baseline_scores": baseline_df,
        "alt_scores": alt_scores,
    }


# %%
def random_weight_test(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_df: pd.DataFrame,
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = False,
    show_progress: bool = False,
) -> dict:
    """
    Test with random weight assignments over multiple trials.
    Args:
        df: DataFrame with tagged text data
        dataset_name: Name of the dataset (for labeling results)
        baseline_df: Pre-computed baseline scores DataFrame
        n_trials: Number of random weight trials to run
        seed: Random seed for reproducibility
        verbose: Whether to print random weights for each trial
        show_progress: Whether to show progress bar
    Returns:
        Harmonized result dict with test name, dataset name, and result DataFrame
    """
    rng = np.random.default_rng(seed)
    scoring_rounds = []
    random_alt_scores = {}  # Store alt_scores for each trial

    trials = range(n_trials)
    if show_progress:
        trials = tqdm(trials, desc=f"random weights ({dataset_name})", unit="trial")

    for trial in trials:
        # Generate random weights in [0, 1] for each tag
        random_weights = {
            tag: float(rng.uniform(0.0, 1.0)) for tag in TAG_WEIGHTS.keys()
        }

        if verbose:
            print(f"Trial {trial + 1} random weights:")
            pprint(random_weights)

        # Score with random weights
        alt_scores = score_all_texts_basic(
            df,
            weights=random_weights,
        )

        # Store alt_scores for this trial
        random_alt_scores[f"trial_{trial + 1}"] = alt_scores

        # Compare with baseline
        result_df, summary = compare_with_baseline(
            baseline_df, alt_scores, scheme_name=f"random_trial_{trial + 1}"
        )

        # Store trial results
        scoring_rounds.append(
            {
                "trial": trial + 1,
                "mean_baseline_score": baseline_df["genericity_score"].mean(),
                "mean_alt_score": alt_scores["genericity_score"].mean(),
                "std_alt_score": alt_scores["genericity_score"].std(),
                "std_baseline_score": baseline_df["genericity_score"].std(),
                "mean_delta_baseline": summary["mean_alt_delta_baseline"],
            }
        )

    # Build result DataFrame
    result_df = pd.DataFrame(scoring_rounds)

    # Create summary statistics across all trials
    summary = {
        "test": f"random_weight_test_{n_trials}_trials",
        "n_trials": n_trials,
        "seed": seed,
        "mean_baseline_score": result_df["mean_baseline_score"].iloc[0],
        "std_baseline_score": result_df["std_baseline_score"].iloc[0],
        "mean_alt_score_across_trials": result_df["mean_alt_score"].mean(),
        "std_alt_score_across_trials": result_df["mean_alt_score"].std(),
        "mean_delta_baseline_across_trials": result_df["mean_delta_baseline"].mean(),
        "std_delta_baseline_across_trials": result_df["mean_delta_baseline"].std(),
        "min_delta_baseline": result_df["mean_delta_baseline"].min(),
        "max_delta_baseline": result_df["mean_delta_baseline"].max(),
    }
    summary_df = pd.DataFrame([summary])

    return {
        "test": f"random_weight_test_{n_trials}_trials",
        "dataset": dataset_name,
        "result": result_df,
        "summary": summary_df,
        "alt_scores": random_alt_scores,
    }




# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================


def write_single_result(
    result: dict,
    outpath: str = "./results/metric_validation/",
) -> None:
    """
    Write a single test result to disk immediately after computation.

    Args:
        result: Harmonized result dict with test, dataset, result keys
        outpath: Base output directory path
    """
    out_dir = Path(outpath)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_name = result["test"]
    dataset_name = result["dataset"]
    result_df = result["result"].copy()

    # Create test subdirectory
    fp = out_dir / test_name
    fp.mkdir(parents=True, exist_ok=True)

    # drop clause2labels to save space
    if "clause2labels" in result_df.columns:
        result_df = result_df.drop(columns=["clause2labels"])

    # Write main result CSV (compressed)
    csv_path = fp / f"{dataset_name}.csv.gz"
    result_df.to_csv(csv_path, index=False, compression="gzip")
    print(f"Wrote {csv_path}")

    # Write summary JSON if present
    if "summary" in result and result["summary"] is not None:
        summary_path = fp / f"{dataset_name}_summary.json"
        result["summary"].to_json(summary_path, orient="records", indent=2)
        print(f"Wrote {summary_path}")

    # Write alt_scores CSV if present (DataFrame) or dict of DataFrames
    if "alt_scores" in result and result["alt_scores"] is not None:
        alt_scores = result["alt_scores"]

        if isinstance(alt_scores, dict):
            # Dict of trial/tag -> DataFrame (random_weight_test case)
            alt_scores_dir = fp / f"{dataset_name}_alt_scores"
            alt_scores_dir.mkdir(parents=True, exist_ok=True)
            for key, alt_df in alt_scores.items():
                alt_df = alt_df.copy()
                if "clause2labels" in alt_df.columns:
                    alt_df = alt_df.drop(columns=["clause2labels"])
                safe_key = str(key).replace(" ", "_").replace("(", "").replace(")", "")
                key_path = alt_scores_dir / f"{safe_key}.csv.gz"
                alt_df.to_csv(key_path, index=False, compression="gzip")
            print(f"Wrote {len(alt_scores)} alt_scores to {alt_scores_dir}/")
        else:
            # Single DataFrame
            alt_scores = alt_scores.copy()
            if "clause2labels" in alt_scores.columns:
                alt_scores = alt_scores.drop(columns=["clause2labels"])
            alt_scores_path = fp / f"{dataset_name}_alt_scores.csv.gz"
            alt_scores.to_csv(alt_scores_path, index=False, compression="gzip")
            print(f"Wrote {alt_scores_path}")

    # Write ablation alt_scores (dict of tag -> DataFrame) if present
    if "ablation_alt_scores" in result and result["ablation_alt_scores"] is not None:
        alt_scores_dir = fp / f"{dataset_name}_alt_scores"
        alt_scores_dir.mkdir(parents=True, exist_ok=True)
        for tag, alt_df in result["ablation_alt_scores"].items():
            alt_df = alt_df.copy()
            if "clause2labels" in alt_df.columns:
                alt_df = alt_df.drop(columns=["clause2labels"])
            # Sanitize tag name for filename (remove ## prefix and special chars)
            safe_tag = (
                tag.replace("##", "")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            tag_path = alt_scores_dir / f"{safe_tag}.csv.gz"
            alt_df.to_csv(tag_path, index=False, compression="gzip")
        print(
            f"Wrote {len(result['ablation_alt_scores'])} ablation alt_scores to {alt_scores_dir}/"
        )

    # Write random_alt_scores (dict of trial -> DataFrame) if present
    if "random_alt_scores" in result and result["random_alt_scores"] is not None:
        alt_scores_dir = fp / f"{dataset_name}_alt_scores"
        alt_scores_dir.mkdir(parents=True, exist_ok=True)
        for trial_name, alt_df in result["random_alt_scores"].items():
            alt_df = alt_df.copy()
            if "clause2labels" in alt_df.columns:
                alt_df = alt_df.drop(columns=["clause2labels"])
            trial_path = alt_scores_dir / f"{trial_name}.csv.gz"
            alt_df.to_csv(trial_path, index=False, compression="gzip")
        print(
            f"Wrote {len(result['random_alt_scores'])} random trial alt_scores to {alt_scores_dir}/"
        )


# %%
if __name__ == "__main__":
    import gc

    outpath = "./results/metric_validation/"
    out_dir = Path(outpath)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = out_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset: compute baseline once, run all tests
    for dataset_path in tqdm(TEST_INPUT, desc="Datasets", unit="dataset"):
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"\nProcessing {dataset_name}...")

        # Load dataset once
        df = pd.read_csv(dataset_path)

        # Compute baseline scores once per dataset
        print("  Computing baseline scores...")
        baseline_df = get_baseline_scores(df)

        # Save baseline (drop clause2labels, use compression)
        baseline_to_save = baseline_df.copy()
        if "clause2labels" in baseline_to_save.columns:
            baseline_to_save = baseline_to_save.drop(columns=["clause2labels"])
        baseline_path = baseline_dir / f"{dataset_name}.csv.gz"
        baseline_to_save.to_csv(baseline_path, index=False, compression="gzip")
        del baseline_to_save
        print(f"  Wrote {baseline_path}")

        print("  Running tag stats...")
        r = run_tag_stats(dataset_path=dataset_path)
        write_single_result(r, outpath=outpath)
        del r
        gc.collect()

        # Run leave-one-out ablation
        print("  Running ablation...")
        r = run_ablation(
            df=df,
            dataset_name=dataset_name,
            baseline_df=baseline_df,
            show_progress=True,
        )
        write_single_result(r, outpath=outpath)
        del r
        gc.collect()

        # Run uniform value tests
        print("  Running uniform value tests...")
        for target_val in [0.0, 0.5, 1.0]:
            r = run_uniform_value(
                df=df,
                dataset_name=dataset_name,
                baseline_df=baseline_df,
                target_value=target_val,
            )
            write_single_result(r, outpath=outpath)
            del r
            gc.collect()

        # Run inverse weight freqs test (weights derived from tag frequencies)
        print("  Running inverse weight (freqs)...")
        r = run_inverse_weight_freqs(
            df=df,
            dataset_name=dataset_name,
            baseline_df=baseline_df,
        )
        write_single_result(r, outpath=outpath)
        del r
        gc.collect()

        # Run inverse weights groups test (predefined inverted group weights)
        print("  Running inverse weights (inverted groups)...")
        r = run_inverse_weights_groups(
            df=df,
            dataset_name=dataset_name,
            baseline_df=baseline_df,
        )
        write_single_result(r, outpath=outpath)
        del r
        gc.collect()

        # Run random weight test
        print("  Running random weight test...")
        r = random_weight_test(
            df=df,
            dataset_name=dataset_name,
            baseline_df=baseline_df,
            n_trials=50,
            seed=42,
            show_progress=True,
        )
        write_single_result(r, outpath=outpath)
        del r
        gc.collect()

        # Clean up dataset-level objects
        del df, baseline_df
        gc.collect()

    print("\nAll tests completed. Results written to:", outpath)

# %%
