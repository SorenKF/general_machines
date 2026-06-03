# this is a script to compute a metric for the degree of generalization in text.
# we will look at the sentence level and at the text level.

# %%
import argparse
import pandas as pd
from collections import Counter, defaultdict
from ast import literal_eval

from wtpsplit import SaT

from tqdm.auto import tqdm
import os
from pathlib import Path
# %%

from plotting import (
    plot_model_comparison,
    plot_model_text_scatter,
    plot_text_model_heatmap_most_variance,
    plot_text_model_heatmap_least_variance,
    plot_score_distribution,
    plot_model_score_distribution_rectangular,
    plot_length_correlation,
    plot_length_distribution,
    plot_length_relationship_lines,
    sanity_check_histogram_counts,
)

# %%

# 2. compute weighted average of the tags.

TAG_WEIGHTS = {
    "##BOUNDED EVENT (SPECIFIC)": 0.0,
    "##BOUNDED EVENT (GENERIC)": 0.5,
    "##UNBOUNDED EVENT (SPECIFIC)": 0.0,
    "##UNBOUNDED EVENT (GENERIC)": 0.5,
    "##BASIC STATE": 0.0,
    "##COERCED STATE (SPECIFIC)": 0.0,
    "##COERCED STATE (GENERIC)": 0.5,
    "##PERFECT COERCED STATE (SPECIFIC)": 0.0,
    "##PERFECT COERCED STATE (GENERIC)": 0.5,
    "##GENERIC SENTENCE (DYNAMIC)": 1.0,
    "##GENERIC SENTENCE (STATIC)": 1.0,
    "##GENERIC SENTENCE (HABITUAL)": 1.0,
    "##GENERALIZING SENTENCE (DYNAMIC)": 0.0,
    "##GENERALIZING SENTENCE (STATIVE)": 0.0,
    "##OTHER": 0.0,
    "##IMPERATIVE": 0.0,
    "##QUESTION": 0.0,
}


# %%
def get_model_splits(
    data: pd.DataFrame, compare_col: str = "model"
) -> dict[str, pd.DataFrame]:
    """
    Split the tagged data into groups per comparison column.

    Args:
        data: DataFrame to split
        compare_col: Column to group by (default: "model")
    """

    return {name: group for name, group in data.groupby(compare_col)}


# %%
# 1. extract clause tags.
def extract_clause_tags(data: pd.DataFrame) -> list[str]:
    """
    Extracts clause tags from the data.
    """
    clause_tags = []
    for _, row in data.iterrows():
        # Ensure clause2labels is properly parsed
        clause2labels = row["clause2labels"]

        # If it's still a string, parse it
        if isinstance(clause2labels, str):
            try:
                clause2labels = literal_eval(clause2labels)
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse clause2labels for row {row.name}")
                continue

        # Now extract the tags from the list of tuples
        if clause2labels and isinstance(clause2labels, list):
            for clause, tag in clause2labels:
                clause_tags.append(tag)
    return clause_tags


def count_clause_tags(
    clause_tags: list[str], default_weights=TAG_WEIGHTS, include_defaults=True
):
    """
    Counts the occurrence of each clause tag.
    include_defaults: whether to add 0 counts for tags not found. Used for fair comparison with baseline in metric_validation.
    """

    counts = Counter(clause_tags)

    if include_defaults:
        all_tags = default_weights.keys()
        for tag in all_tags:
            counts.setdefault(tag, 0)

    return counts


# %%

# 3. sentence level generalization metric.

# compute:
# tag frequencies per sentence "coverage".
# normalize tags by number of clauses in the sentence.


# %%
# 4. text level generalization metric.


def compute_weighted_avg(tags: list[str], weights: TAG_WEIGHTS) -> float:
    """
    Core function: compute weighted genericity score from a list of tags.
    """
    if not tags:
        return 0.0

    tag_counts = count_clause_tags(tags)

    total_weighted_score = 0.0
    total_tags = len(tags)

    for tag, count in tag_counts.items():
        if tag not in weights.keys():
            continue
        total_weighted_score += weights[tag] * count

    return total_weighted_score / total_tags


# %%
def score_all_texts(
    data: pd.DataFrame,
    weights: TAG_WEIGHTS,
    compare_col: str = "model",
    id_col: str = "",
) -> pd.DataFrame:
    """
    Score each individual text in the dataset per comparison group.

    version with grouping (groups ids by model, then split.)

    Args:
        data: DataFrame with text data
        compare_col: Column to group by for comparison (default: "model")

    Returns: DataFrame with global_idx, compare_col, and genericity_score columns
    """
    scores = []

    # Use id_col if set, otherwise fall back to global_idx, then local_idx
    if not id_col:
        idx_column = "global_idx" if "global_idx" in data.columns else "local_idx"
    else:
        idx_column = id_col

    for (idx, group_val), group in data.groupby([idx_column, compare_col]):
        tags = extract_clause_tags(group)
        score = compute_weighted_avg(tags, weights=weights)
        result = {idx_column: idx, compare_col: group_val, "genericity_score": score}

        # Add split info
        if "split" in data.columns:
            result["split"] = group["split"].iloc[0]

        scores.append(result)

    return pd.DataFrame(scores)


# %%


def score_all_texts_basic(
    data: pd.DataFrame,
    weights: dict = TAG_WEIGHTS,
    id_col: str = "",
) -> pd.DataFrame:
    """
    Basic version: Loop through dataframe, extract tags, compute scores, and assign back to input df.
    """
    scores = []

    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = id_col or (
        "global_idx" if "global_idx" in data.columns else "local_idx"
    )

    for _, row in data.iterrows():
        # Extract tags for this row
        clause2labels = row["clause2labels"]
        # If it's a string, parse it
        if isinstance(clause2labels, str):
            try:
                clause2labels = literal_eval(clause2labels)
            except Exception:
                clause2labels = []
        # Extract tags from list of tuples
        tags = [tag for _, tag in clause2labels] if clause2labels else []
        # Compute score
        score = compute_weighted_avg(tags, weights)
        scores.append(score)

    # Assign scores back to DataFrame
    data = data.copy()
    data["genericity_score"] = scores
    return data


# %%


# 5. comparison and visualization.
# %%
def compare_models(scored_df: pd.DataFrame, compare_col: str = "model") -> pd.DataFrame:
    """
    Compare groups by their average genericity scores.

    Args:
        scored_df: DataFrame with scored texts
        compare_col: Column to group by for comparison (default: "model")
    """
    return (
        scored_df.groupby(compare_col)["genericity_score"]
        .agg(["mean", "std", "count", "min", "max"])
        .round(4)
    )


# %%
def compare_texts(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare genericity scores for each text across models.
    Returns: DataFrame with text-level statistics across models
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    return (
        scored_df.groupby(idx_column)["genericity_score"]
        .agg(["mean", "std", "count", "min", "max"])
        .round(4)
        .sort_values("mean", ascending=False)
    )


# %%
def find_most_generic_texts(
    scored_df: pd.DataFrame, top_n: int = 100
) -> pd.DataFrame:  # Changed default to 100
    """
    Find the most generic texts across all models.

    Args:
        scored_df: DataFrame with scored texts
        top_n: Number of most generic texts to return (default: 100)
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    text_stats = compare_texts(scored_df)

    # Get the top N most generic texts
    most_generic = text_stats.head(top_n).copy()

    # Add detailed scores per group for these texts
    detailed_scores = []
    # Auto-detect compare_col (the column that's not idx_column, genericity_score, or split)
    compare_col = [
        col
        for col in scored_df.columns
        if col not in [idx_column, "genericity_score", "split"]
    ][0]

    for idx in most_generic.index:
        text_data = scored_df[scored_df[idx_column] == idx]
        for _, row in text_data.iterrows():
            result = {
                idx_column: idx,
                compare_col: row[compare_col],
                "genericity_score": row["genericity_score"],
                "avg_across_groups": most_generic.loc[idx, "mean"],
                "std_across_groups": most_generic.loc[
                    idx, "std"
                ],  # Add std for reference
            }
            # Add split info if available
            if "split" in row:
                result["split"] = row["split"]
            detailed_scores.append(result)

    return pd.DataFrame(detailed_scores).sort_values(
        ["avg_across_groups", idx_column], ascending=[False, True]
    )


# %%
def find_least_generic_texts(
    scored_df: pd.DataFrame, top_n: int = 100
) -> pd.DataFrame:  # Changed default to 100
    """
    Find the least generic texts across all models.

    Args:
        scored_df: DataFrame with scored texts
        top_n: Number of least generic texts to return (default: 100)
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    text_stats = compare_texts(scored_df)

    # Get the top N least generic texts (lowest mean scores)
    least_generic = text_stats.tail(top_n).copy()  # tail instead of head

    # Add detailed scores per group for these texts
    detailed_scores = []
    # Auto-detect compare_col (the column that's not idx_column, genericity_score, or split)
    compare_col = [
        col
        for col in scored_df.columns
        if col not in [idx_column, "genericity_score", "split"]
    ][0]

    for idx in least_generic.index:
        text_data = scored_df[scored_df[idx_column] == idx]
        for _, row in text_data.iterrows():
            result = {
                idx_column: idx,
                compare_col: row[compare_col],
                "genericity_score": row["genericity_score"],
                "avg_across_groups": least_generic.loc[idx, "mean"],
                "std_across_groups": least_generic.loc[
                    idx, "std"
                ],  # Add std for reference
            }
            # Add split info if available
            if "split" in row:
                result["split"] = row["split"]
            detailed_scores.append(result)

    return pd.DataFrame(detailed_scores).sort_values(
        ["avg_across_groups", idx_column],
        ascending=[True, True],  # ascending=True for lowest first
    )


# %%
def compare_text_variance(scored_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Find texts with highest variance across models (most disagreement).

    Args:
        scored_df: DataFrame with scored texts
        top_n: Number of most variable texts to return (default: 100)
    """
    text_stats = compare_texts(scored_df)
    return text_stats.sort_values("std", ascending=False).head(top_n)


# %%
def analyze_overall_length_correlation(length_scored_df: pd.DataFrame) -> dict:
    """
    Analyze correlation between text length and genericity for the entire dataset.
    """
    # Calculate overall correlations (across all models and texts)
    overall_corr_clauses = length_scored_df["genericity_score"].corr(
        length_scored_df["num_clauses"]
    )
    overall_corr_words = length_scored_df["genericity_score"].corr(
        length_scored_df["num_words"]
    )

    # Get summary statistics
    stats = {
        "overall_correlation_clauses": overall_corr_clauses,
        "overall_correlation_words": overall_corr_words,
        "total_texts": len(length_scored_df),
        "mean_clauses": length_scored_df["num_clauses"].mean(),
        "mean_words": length_scored_df["num_words"].mean(),
        "std_clauses": length_scored_df["num_clauses"].std(),
        "std_words": length_scored_df["num_words"].std(),
    }

    return stats


# %%
def save_results_to_csv(
    scored_df: pd.DataFrame,
    model_comparison: pd.DataFrame,
    text_comparison: pd.DataFrame,
    most_generic: pd.DataFrame,
    least_generic: pd.DataFrame,
    pivot_data_high_var: pd.DataFrame,
    pivot_data_low_var: pd.DataFrame,
    length_scored_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    overall_stats: dict,
    data_path: str,
    results_dir: str = "./results/",
    most_variable: pd.DataFrame = None,
    compare_col: str = "model",
):
    """
    Save all computed statistics DataFrames to CSV files in the results directory.
    Extracts prefix from the data_path filename and creates subdirectories.
    """
    # Extract prefix from filename
    data_filename = Path(data_path).stem
    prefix = data_filename.replace("corpus-", "").replace("-tagged", "")

    # Create subdirectory for this dataset
    dataset_results_dir = os.path.join(results_dir, prefix)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    # Save main results DataFrames
    scored_df.to_csv(f"{dataset_results_dir}/scored_texts.csv", index=False)
    print(f"Saved {prefix}/scored_texts.csv with {len(scored_df)} rows")

    model_comparison.to_csv(
        f"{dataset_results_dir}/{compare_col}_comparison.csv", index=True
    )
    print(
        f"Saved {prefix}/{compare_col}_comparison.csv with {len(model_comparison)} {compare_col}s"
    )

    text_comparison.to_csv(f"{dataset_results_dir}/text_comparison.csv", index=True)
    print(f"Saved {prefix}/text_comparison.csv with {len(text_comparison)} texts")

    most_generic.to_csv(f"{dataset_results_dir}/most_generic_texts.csv", index=False)
    print(f"Saved {prefix}/most_generic_texts.csv with {len(most_generic)} entries")

    least_generic.to_csv(f"{dataset_results_dir}/least_generic_texts.csv", index=False)
    print(f"Saved {prefix}/least_generic_texts.csv with {len(least_generic)} entries")

    # Save the full variance analysis if provided
    if most_variable is not None:
        most_variable.to_csv(
            f"{dataset_results_dir}/text_variance_analysis.csv", index=True
        )
        print(
            f"Saved {prefix}/text_variance_analysis.csv with {len(most_variable)} texts"
        )

    # Save heatmap pivot data
    pivot_data_high_var.to_csv(
        f"{dataset_results_dir}/high_variance_texts_pivot.csv", index=True
    )
    print(
        f"Saved {prefix}/high_variance_texts_pivot.csv with {len(pivot_data_high_var)} texts"
    )

    pivot_data_low_var.to_csv(
        f"{dataset_results_dir}/low_variance_texts_pivot.csv", index=True
    )
    print(
        f"Saved {prefix}/low_variance_texts_pivot.csv with {len(pivot_data_low_var)} texts"
    )

    # Save length analysis results
    length_scored_df.to_csv(
        f"{dataset_results_dir}/length_scored_texts.csv", index=False
    )
    print(
        f"Saved {prefix}/length_scored_texts.csv with {len(length_scored_df)} entries"
    )

    correlation_df.to_csv(f"{dataset_results_dir}/length_correlations.csv", index=False)
    print(
        f"Saved {prefix}/length_correlations.csv with {len(correlation_df)} {compare_col}s"
    )

    # Save overall statistics as CSV (convert dict to DataFrame)
    overall_stats_df = pd.DataFrame([overall_stats])
    overall_stats_df.to_csv(
        f"{dataset_results_dir}/overall_statistics.csv", index=False
    )
    print(f"Saved {prefix}/overall_statistics.csv")

    # Create a summary file with key metrics
    # Auto-detect compare_col for summary
    compare_col = [
        col
        for col in scored_df.columns
        if col not in [idx_column, "genericity_score", "split"]
    ][0]

    summary_data = {
        "metric": [
            "total_texts_analyzed",
            f"total_{compare_col}s",
            "mean_genericity_score_overall",
            "std_genericity_score_overall",
            "correlation_clauses_overall",
            "correlation_words_overall",
            f"most_generic_{compare_col}",
            f"least_generic_{compare_col}",
        ],
        "value": [
            len(scored_df[idx_column].unique()),
            len(scored_df[compare_col].unique()),
            scored_df["genericity_score"].mean(),
            scored_df["genericity_score"].std(),
            overall_stats["overall_correlation_clauses"],
            overall_stats["overall_correlation_words"],
            model_comparison["mean"].idxmax(),
            model_comparison["mean"].idxmin(),
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{dataset_results_dir}/analysis_summary.csv", index=False)
    print(f"Saved {prefix}/analysis_summary.csv")

    print(f"\nAll results saved to {dataset_results_dir}/")
    print("Files created:")
    for file in os.listdir(dataset_results_dir):
        if file.endswith(".csv"):
            print(f"  - {file}")


# %%
def analyze_length_correlation(
    data: pd.DataFrame, scored_df: pd.DataFrame, compare_col: str = "model"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze correlation between text length and genericity scores.
    Returns enhanced DataFrame with length info and correlation stats.

    Args:
        data: Original DataFrame with text data
        scored_df: DataFrame with scored texts
        compare_col: Column to group by for comparison (default: "model")
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in data.columns else "local_idx"

    # Calculate text lengths
    length_data = []
    for (idx, group_val), group in data.groupby([idx_column, compare_col]):
        text_data = group.iloc[0]  # All rows for same text+group should have same text

        # Count clauses and words
        num_clauses = (
            len(text_data["clause2labels"]) if text_data["clause2labels"] else 0
        )
        num_words = len(text_data["text"].split()) if pd.notna(text_data["text"]) else 0

        result = {
            idx_column: idx,
            compare_col: group_val,
            "num_clauses": num_clauses,
            "num_words": num_words,
        }

        # Add split info if available
        if "split" in text_data:
            result["split"] = text_data["split"]

        length_data.append(result)

    length_df = pd.DataFrame(length_data)

    # Merge with scored data
    length_scored_df = pd.merge(
        scored_df,
        length_df,
        on=[idx_column, compare_col]
        + (["split"] if "split" in scored_df.columns else []),
    )

    # Calculate correlations by group
    correlations = []
    for group_val in scored_df[compare_col].unique():
        group_data = length_scored_df[length_scored_df[compare_col] == group_val]

        corr_clauses = group_data["genericity_score"].corr(group_data["num_clauses"])
        corr_words = group_data["genericity_score"].corr(group_data["num_words"])

        correlations.append(
            {
                compare_col: group_val,
                "correlation_clauses": corr_clauses,
                "correlation_words": corr_words,
                "num_texts": len(group_data),
            }
        )

    correlation_df = pd.DataFrame(correlations)

    return length_scored_df, correlation_df


# %%
def create_global_idx(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create global_idx by stripping model name from local_idx.
    Converts "chatgpt_train_25" -> "train_25" etc.
    """
    data = data.copy()

    # Extract global_idx by removing model prefix from local_idx
    def extract_global_idx(row):
        local_idx = row["local_idx"]

        data_split = row["split"]

        global_idx = data_split + "_" + str(local_idx)

        return global_idx

    data["global_idx"] = data.apply(extract_global_idx, axis=1)

    return data


# %%
def main(
    data_path: str = "./data/corpus-test-tagged.csv",
    n_files: int = 100,
    results_base_dir: str = None,
    plots_base_dir: str = None,
    compare_col: str = "model",
    show_std: bool = True,
):
    """
    Main analysis function that can work with any dataset.
    Automatically extracts prefix from the data_path filename and creates subdirectories.

    Args:
        data_path: Path to the tagged CSV file
        n_files: Number of most/least generic and most variable texts to save to CSV (default: 100)
        results_base_dir: Base directory for results (default: derived from input filename)
        plots_base_dir: Base directory for plots (default: derived from input filename)
        compare_col: Column to group by for comparison (default: "model")
    """
    tqdm.pandas()

    # Extract prefix from filename
    input_path = Path(data_path)
    data_filename = input_path.stem

    # If base directories not provided, derive from input filename
    if results_base_dir is None:
        # Use the input filename stem directly as base_name
        results_base_dir = f"./results/{data_filename}"

    if plots_base_dir is None:
        # Use the input filename stem directly as base_name
        plots_base_dir = f"./plots/{data_filename}"

    # Load data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    data["clause2labels"] = data["clause2labels"].apply(literal_eval)

    # If compare_col is not set, just score and save, skip all comparison/visualization/analysis/print
    if not compare_col:
        print(
            "\n*** No --compare-col provided: skipping comparison and visualization, only scoring texts. ***\n"
        )
        # Create global_idx if it doesn't exist
        if "global_idx" not in data.columns:
            print("Creating global_idx from local_idx...")
            data = create_global_idx(data)
            print(f"Sample global_idx values: {data['global_idx'].head(10).tolist()}")

        # Use global_idx if available, otherwise fall back to local_idx
        idx_column = "global_idx" if "global_idx" in data.columns else "local_idx"

        # Score all texts (group by idx_column only)
        scored_df = score_all_texts_basic(data, id_col=idx_column, weights=TAG_WEIGHTS)

        # Merge scores back to input data
        out_df = pd.merge(
            data, scored_df[[idx_column, "genericity_score"]], on=idx_column, how="left"
        )

        # Save to results directory, same as usual
        if "split" in data.columns and len(data["split"].unique()) == 1:
            split_name = data["split"].unique()[0]
            results_dir = f"{results_base_dir}/{split_name}/"
        else:
            results_dir = f"{results_base_dir}/"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "scored_texts.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved scored texts to {out_path} ({len(out_df)} rows)")
        return out_df

    # Validate compare_col exists
    if compare_col not in data.columns:
        raise ValueError(
            f"Column '{compare_col}' not found in data. Available columns: {data.columns.tolist()}"
        )

    # Create global_idx if it doesn't exist
    if "global_idx" not in data.columns:
        print("Creating global_idx from local_idx...")
        data = create_global_idx(data)
        print(f"Sample global_idx values: {data['global_idx'].head(10).tolist()}")

    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in data.columns else "local_idx"

    print(f"Dataset info:")
    print(f"  Total rows: {len(data)}")
    print(f"  Unique texts: {len(data[idx_column].unique())}")
    print(f"  {compare_col.capitalize()}s: {data[compare_col].unique()}")
    print(f"  Using index column: {idx_column}")
    print(f"  Comparison column: {compare_col}")
    if "split" in data.columns:
        print(f"  Splits: {data['split'].unique()}")

    # Score all texts
    scored_df = score_all_texts(data, compare_col=compare_col, weights=TAG_WEIGHTS)

    # Run sanity check
    sanity_check_histogram_counts(scored_df, compare_col=compare_col)

    # Compare groups
    model_comparison = compare_models(scored_df, compare_col=compare_col)
    print(f"{compare_col.capitalize()} Comparison:")
    print(model_comparison)
    print()

    # Compare texts
    text_comparison = compare_texts(scored_df)
    print("Text Comparison (Top 10 most generic):")
    print(text_comparison.head(10))
    print()

    # Find most generic texts with group breakdown (using n_files parameter)
    most_generic = find_most_generic_texts(scored_df, top_n=n_files)
    num_groups = len(scored_df[compare_col].unique())
    print(f"Most Generic Texts (showing top 20 of {n_files} saved):")
    print(most_generic.head(20 * num_groups))  # Show top 20 texts × num_groups
    print()

    # Find least generic texts with group breakdown (using n_files parameter)
    least_generic = find_least_generic_texts(scored_df, top_n=n_files)
    print(f"Least Generic Texts (showing top 20 of {n_files} saved):")
    print(least_generic.head(20 * num_groups))  # Show top 20 texts × num_groups
    print()

    # Find texts with most disagreement (using n_files parameter)
    most_variable = compare_text_variance(scored_df, top_n=n_files)
    print(
        f"Texts with Most {compare_col.capitalize()} Disagreement (showing top 20 of {n_files} saved):"
    )
    print(most_variable.head(20))
    print()

    # Analyze length correlation
    length_scored_df, correlation_df = analyze_length_correlation(
        data, scored_df, compare_col=compare_col
    )
    print(f"Length-Genericity Correlations by {compare_col.capitalize()}:")
    print(correlation_df.round(4))
    print()

    # Analyze overall length correlation
    overall_stats = analyze_overall_length_correlation(length_scored_df)
    print("Overall Length-Genericity Correlation:")
    print(
        f"Correlation with clauses: {overall_stats['overall_correlation_clauses']:.4f}"
    )
    print(f"Correlation with words: {overall_stats['overall_correlation_words']:.4f}")
    print(f"Total data points: {overall_stats['total_texts']}")
    print()

    # Create visualizations
    print("Creating visualizations...")

    # Create plots subdirectory for this dataset
    # If split column exists, create split-specific directories
    if "split" in data.columns and len(data["split"].unique()) == 1:
        split_name = data["split"].unique()[0]
        plots_dir = f"{plots_base_dir}/{split_name}/"
    else:
        plots_dir = f"{plots_base_dir}/"

    os.makedirs(plots_dir, exist_ok=True)

    # 1. Group comparison plots
    plot_model_comparison(
        model_comparison,
        save_path=f"{plots_dir}{compare_col}_comparison.png",
        compare_col=compare_col,
        show_std=show_std,
    )

    # 2. Scatter plots of group-text scores
    plot_model_text_scatter(
        scored_df,
        save_path=f"{plots_dir}{compare_col}_text_scatter.png",
        compare_col=compare_col,
    )

    # 3. Heatmap of most variable texts (keep plots at 20 texts for readability)
    plot_text_model_heatmap_most_variance(
        scored_df,
        top_n_texts=20,  # Keep plots at 20 for readability
        save_path=f"{plots_dir}text_{compare_col}_heatmap_high_variance.png",
        compare_col=compare_col,
    )

    # 4. Heatmap of least variable texts (most agreement) (keep plots at 20 texts)
    plot_text_model_heatmap_least_variance(
        scored_df,
        top_n_texts=20,  # Keep plots at 20 for readability
        save_path=f"{plots_dir}text_{compare_col}_heatmap_low_variance.png",
        compare_col=compare_col,
    )

    # 5. Score distributions and correlations
    plot_score_distribution(
        scored_df,
        save_path=f"{plots_dir}score_distributions.png",
        compare_col=compare_col,
        show_std=show_std,
    )

    # Inside main(), after plot_score_distribution
    plot_model_score_distribution_rectangular(
        scored_df,
        save_path=f"{plots_dir}{compare_col}_score_distribution_rectangular.png",
        compare_col=compare_col,
        show_std=show_std,
    )

    # 6. Length correlation analysis (by group)
    plot_length_correlation(
        length_scored_df,
        correlation_df,
        save_path=f"{plots_dir}length_correlation.png",
        compare_col=compare_col,
    )

    # 7. Length distribution plots
    plot_length_distribution(
        length_scored_df,
        save_path=f"{plots_dir}length_distribution.png",
    )

    # 8. Length relationship line plots
    plot_length_relationship_lines(
        length_scored_df,
        save_path=f"{plots_dir}length_relationship_lines.png",
        compare_col=compare_col,
        show_std=show_std,
    )

    # Create pivot data for CSV files (using n_files and correct idx_column)
    print(
        f"Creating pivot data for CSV files (n_files={n_files}, using {idx_column})..."
    )

    # High variance pivot data for CSV (top n_files)
    text_variance = scored_df.groupby(idx_column)["genericity_score"].agg(
        ["mean", "std"]
    )
    top_variable_texts_csv = text_variance.nlargest(n_files, "std").index
    heatmap_data_high_var_csv = scored_df[
        scored_df[idx_column].isin(top_variable_texts_csv)
    ]
    pivot_data_high_var_csv = heatmap_data_high_var_csv.pivot(
        index=idx_column,
        columns=compare_col,
        values="genericity_score",
    )

    # Low variance pivot data for CSV (bottom n_files)
    least_variable_texts_csv = text_variance.nsmallest(n_files, "std").index
    heatmap_data_low_var_csv = scored_df[
        scored_df[idx_column].isin(least_variable_texts_csv)
    ]
    pivot_data_low_var_csv = heatmap_data_low_var_csv.pivot(
        index=idx_column,
        columns=compare_col,
        values="genericity_score",
    )

    # Save all results to CSV files
    print(f"\nSaving results to CSV files (n_files={n_files})...")

    # Determine results directory based on split
    if "split" in data.columns and len(data["split"].unique()) == 1:
        split_name = data["split"].unique()[0]
        results_dir = f"{results_base_dir}/{split_name}/"
    else:
        results_dir = f"{results_base_dir}/"

    save_results_to_csv(
        scored_df=scored_df,
        model_comparison=model_comparison,
        text_comparison=text_comparison,
        most_generic=most_generic,
        least_generic=least_generic,
        pivot_data_high_var=pivot_data_high_var_csv,  # Use CSV version with n_files
        pivot_data_low_var=pivot_data_low_var_csv,  # Use CSV version with n_files
        length_scored_df=length_scored_df,
        correlation_df=correlation_df,
        overall_stats=overall_stats,
        data_path=data_path,
        results_dir=results_dir,
        most_variable=most_variable,  # Pass the full variance data
        compare_col=compare_col,
    )

    return (
        scored_df,
        model_comparison,
        text_comparison,
        most_generic,
        least_generic,
        pivot_data_high_var_csv,  # Return CSV version
        pivot_data_low_var_csv,  # Return CSV version
        length_scored_df,
        correlation_df,
        overall_stats,
    )


# %%
def run_all_analyses(
    n_files: int = 100, compare_col: str = "model", show_std: bool = True
):
    """
    Run analysis on all tagged CSV files found in the data directory.

    Args:
        n_files: Number of most/least generic and most variable texts to save to CSV (default: 100)
        compare_col: Column to group by for comparison (default: "model")
        show_std: Whether to show standard deviation in plots (default: True)
    """
    import glob

    # Find all tagged CSV files in the data directory
    data_files = glob.glob("./data/corpus-*tagged.csv")

    results = {}
    datasets = []

    for data_path in data_files:
        datasets.append(Path(data_path).stem)

        try:
            print(f"\nAnalyzing {data_path}...")
            main(
                data_path=data_path,
                n_files=n_files,
                compare_col=compare_col,
                show_std=show_std,
            )
            results[Path(data_path).stem] = "Success"
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            results[Path(data_path).stem] = f"Error: {e}"

    # Print summary of results
    print("\nAnalysis Summary:")
    for dataset_name, status in results.items():
        print(f"  {dataset_name}: {status}")

    if len(results) < len(datasets):
        failed_datasets = set(datasets) - set(results.keys())
        print(f"\nFailed to analyze {len(failed_datasets)} datasets:")
        for dataset_path in failed_datasets:
            dataset_name = (
                Path(dataset_path).stem.replace("corpus-", "").replace("-tagged", "")
            )
            print(f"  ✗ {dataset_name}")

    return results


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute genericity metrics and generate visualizations for tagged text data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file with default model comparison
  python generalization_metric.py --input ./data/corpus-test-tagged.csv
  
  # Analyze GUM dataset comparing by domain
  python generalization_metric.py --input ./data/GUM_texts_tagged.csv --compare-col domain
  
  # Analyze with custom number of top/bottom texts to save
  python generalization_metric.py --input ./data/corpus-test-tagged.csv --n-files 200
  
  # Analyze all tagged files in ./data/
  python generalization_metric.py
""",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to a single tagged CSV file to analyze. If not provided, processes all corpus-*tagged.csv files in ./data/",
    )
    parser.add_argument(
        "--n-files",
        "-n",
        type=int,
        default=100,
        help="Number of most/least generic and most variable texts to save to CSV (default: 100)",
    )
    parser.add_argument(
        "--compare-col",
        "-c",
        type=str,
        # default="model",
        help="Column to group by for comparison (default: 'model'). Use 'domain' for GUM-like datasets.",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap resampling and std/error bars in plots (use when comparing independent single-sample groups like GUM domains)",
    )

    args = parser.parse_args()

    if args.input:
        # Process single file
        input_path = Path(args.input).expanduser()

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            exit(1)

        print(f"Analyzing {input_path}...")
        try:
            main(
                data_path=str(input_path),
                n_files=args.n_files,
                compare_col=args.compare_col,
                show_std=not args.no_bootstrap,
            )
            print(f"✓ Successfully analyzed {input_path}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            import traceback

            traceback.print_exc()
            exit(1)
    else:
        # Process all tagged files in data directory
        run_all_analyses(
            n_files=args.n_files,
            compare_col=args.compare_col,
            show_std=not args.no_bootstrap,
        )
