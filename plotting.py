# Plotting functions for genericity analysis
# Contains all visualization and plotting logic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

SHOW_PLOTS = False  # Set to True to display plots interactively, False to only save


# Predefined color and marker mappings for the models in the paper.
MODEL_STYLE_MAP = {
    "chatgpt": {"color": "#1f77b4", "marker": "o"},
    "flan_t5_xxl": {"color": "#ff7f0e", "marker": "o"},
    "human": {"color": "#2ca02c", "marker": "x"},
    "text_davinci_003": {"color": "#d62728", "marker": "o"},
}


def get_model_colors(models):
    """
    Get consistent colors for models across all plots.
    Uses predefined mapping from MODEL_STYLE_MAP, with fallback to seaborn palette.
    """
    import seaborn as sns

    colors = []
    # Generate fallback palette for any unknown values
    default_palette = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        if model in MODEL_STYLE_MAP:
            colors.append(MODEL_STYLE_MAP[model]["color"])
        else:
            # Use seaborn color for unknown models
            colors.append(default_palette[i])

    return colors


def get_model_markers(models):
    """
    Get consistent markers for models across all plots.
    Uses predefined mapping from MODEL_STYLE_MAP, with fallback to default markers.
    Helps distinguish models for colorblind users.
    """
    default_markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "+"]
    markers = []

    for i, model in enumerate(models):
        if model in MODEL_STYLE_MAP:
            markers.append(MODEL_STYLE_MAP[model]["marker"])
        else:
            # Use default marker cycling for unknown models
            markers.append(default_markers[i % len(default_markers)])

    return markers


def _show_plot():
    """Helper function to conditionally show plots based on SHOW_PLOTS toggle."""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    model_comparison: pd.DataFrame,
    save_path: str = None,
    compare_col: str = "model",
    show_std: bool = True,
):
    """
    Create a comprehensive comparison plot of group performance.
    Shows mean ± std and min-max ranges with mean markers.

    Args:
        model_comparison: DataFrame with comparison statistics
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
        show_std: Whether to show standard deviation error bars (default: True)
    """
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Get consistent colors for groups
    groups = model_comparison.index
    colors = get_model_colors(groups)

    x = range(len(groups))

    # Plot 1: Mean with std deviation error bars
    means = model_comparison["mean"].values
    stds = model_comparison["std"].values

    bars1 = ax1.bar(
        x,
        means,
        yerr=stds if show_std else None,
        alpha=0.7,
        color=colors,
        capsize=5,
        error_kw={"linewidth": 2, "ecolor": "black"},
    )

    ax1.set_xlabel(compare_col.capitalize(), fontsize=12)
    ax1.set_ylabel("Genericity Score", fontsize=12)
    title_suffix = " (Mean ± Std)" if show_std else " (Mean)"
    ax1.set_title(
        f"{compare_col.capitalize()} Genericity Scores{title_suffix}", fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=45, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    # Add mean values on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars1, means)):
        height = bar.get_height()
        y_offset = stds[i] + 0.02 if show_std else 0.02
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f"{mean_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Min-Max ranges with mean markers
    mins = model_comparison["min"].values
    maxs = model_comparison["max"].values
    ranges = maxs - mins

    # Draw lines from min to max
    for i, (min_val, max_val, mean_val, color) in enumerate(
        zip(mins, maxs, means, colors)
    ):
        ax2.plot(
            [i, i],
            [min_val, max_val],
            "o-",
            color=color,
            linewidth=3,
            markersize=8,
            alpha=0.7,
        )
        # Add mean marker
        ax2.plot(
            i,
            mean_val,
            "o",
            color="red",
            markersize=10,
            label="Mean" if i == 0 else "",
            zorder=10,
        )

    ax2.set_xlabel(compare_col.capitalize(), fontsize=12)
    ax2.set_ylabel("Genericity Score", fontsize=12)
    ax2.set_title(
        f"{compare_col.capitalize()} Score Ranges (Min-Max with Mean)",
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def plot_model_text_scatter(
    scored_df: pd.DataFrame, save_path: str = None, compare_col: str = "model"
):
    """
    Create scatter plots showing group vs text relationships.

    Args:
        scored_df: DataFrame with scored texts
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
    """
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get unique groups, colors, and markers
    groups = scored_df[compare_col].unique()
    colors = get_model_colors(groups)
    markers = get_model_markers(groups)

    # Plot 1: All data points
    for i, group in enumerate(groups):
        group_data = scored_df[scored_df[compare_col] == group]
        ax1.scatter(
            group_data.index,
            group_data["genericity_score"],
            alpha=0.6,
            label=group,
            color=colors[i],  # Use consistent color
            marker=markers[i],  # Use distinct marker
            s=50,  # Slightly larger markers
        )

    ax1.set_xlabel("Text Index")
    ax1.set_ylabel("Genericity Score")
    ax1.set_title(f"Genericity Scores by {compare_col.capitalize()} (All Texts)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Box plot by group
    group_scores = [
        scored_df[scored_df[compare_col] == group]["genericity_score"].values
        for group in groups
    ]

    bp = ax2.boxplot(group_scores, labels=groups, patch_artist=True, showmeans=True)

    # Color the boxes consistently
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel(compare_col.capitalize())
    ax2.set_ylabel("Genericity Score")
    ax2.set_title(f"Genericity Score Distribution by {compare_col.capitalize()}")
    ax2.grid(alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def plot_text_model_heatmap_most_variance(
    scored_df: pd.DataFrame,
    top_n_texts: int = 50,
    save_path: str = None,
    compare_col: str = "model",
):
    """
    Create a heatmap showing the most variable texts across groups.

    Args:
        scored_df: DataFrame with scored texts
        top_n_texts: Number of texts to show
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    # Get texts with highest variance (most disagreement between models)
    text_variance = scored_df.groupby(idx_column)["genericity_score"].agg(
        ["mean", "std"]
    )
    top_variable_texts = text_variance.nlargest(top_n_texts, "std").index

    # Create pivot table for heatmap
    heatmap_data = scored_df[scored_df[idx_column].isin(top_variable_texts)]
    pivot_data = heatmap_data.pivot(
        index=idx_column, columns=compare_col, values="genericity_score"
    )

    # Create the heatmap
    plt.figure(figsize=(10, max(8, top_n_texts * 0.4)))
    sns.heatmap(
        pivot_data,
        annot=True,
        cbar_kws={"label": "Genericity Score"},
        fmt=".3f",
        annot_kws={"fontsize": 8},
        vmin=0,  # Set scale from 0 to 1
        vmax=1,
    )
    plt.title(
        f"Heatmap of Top {top_n_texts} Most Variable Texts Across {compare_col.capitalize()}s"
    )
    plt.xlabel(compare_col.capitalize())
    plt.ylabel("Text ID")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()

    return pivot_data


def plot_text_model_heatmap_least_variance(
    scored_df: pd.DataFrame,
    top_n_texts: int = 50,
    save_path: str = None,
    compare_col: str = "model",
):
    """
    Create a heatmap showing the least variable texts across groups.

    Args:
        scored_df: DataFrame with scored texts
        top_n_texts: Number of texts to show
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
    """
    # Use global_idx if available, otherwise fall back to local_idx
    idx_column = "global_idx" if "global_idx" in scored_df.columns else "local_idx"

    # Get texts with lowest variance (most agreement between models)
    text_variance = scored_df.groupby(idx_column)["genericity_score"].agg(
        ["mean", "std"]
    )
    # Get all variance texts and take bottom n
    all_variance_texts = text_variance.sort_values("std")
    low_variance_texts = all_variance_texts.head(top_n_texts).index

    # Create pivot table for heatmap
    heatmap_data = scored_df[scored_df[idx_column].isin(low_variance_texts)]
    pivot_data = heatmap_data.pivot(
        index=idx_column, columns=compare_col, values="genericity_score"
    )

    # Create the heatmap
    plt.figure(figsize=(10, max(8, top_n_texts * 0.4)))
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap="RdYlBu_r",
        cbar_kws={"label": "Genericity Score"},
        fmt=".3f",
        annot_kws={"fontsize": 8},
        vmin=0,  # Set scale from 0 to 1
        vmax=1,
    )
    plt.title(
        f"Heatmap of Top {top_n_texts} Least Variable Texts Across {compare_col.capitalize()}s"
    )
    plt.xlabel(compare_col.capitalize())
    plt.ylabel("Text ID")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()

    return pivot_data


def plot_score_distribution(
    scored_df: pd.DataFrame,
    save_path: str = None,
    compare_col: str = "model",
    show_std: bool = True,
):
    """
    Plot the distribution of genericity scores for each group as lines (x axis) across texts (y axis).

    Args:
        scored_df: DataFrame with scored texts
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
        show_std: Whether to show standard deviation error bars (default: True)
    """
    groups = scored_df[compare_col].unique()
    colors = get_model_colors(groups)
    markers = get_model_markers(groups)

    fig, ax = plt.subplots(figsize=(12, 6))

    for group, color, marker in zip(groups, colors, markers):
        group_data = scored_df[scored_df[compare_col] == group]["genericity_score"]

        # Create histogram data with 10 bins for cleaner intervals
        counts, bin_edges = np.histogram(group_data, bins=10, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if show_std:
            # Calculate standard deviation using bootstrap
            n_bootstrap = 100
            bootstrap_counts = []

            for _ in range(n_bootstrap):
                # Resample with replacement
                sample = np.random.choice(
                    group_data, size=len(group_data), replace=True
                )
                boot_counts, _ = np.histogram(sample, bins=bin_edges)
                bootstrap_counts.append(boot_counts)

            bootstrap_counts = np.array(bootstrap_counts)
            counts_std = np.std(bootstrap_counts, axis=0)

            # Plot line with error bars
            ax.errorbar(
                bin_centers,
                counts,
                yerr=counts_std,
                marker=marker,
                linewidth=2,
                label=group,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.5,
                alpha=0.8,
                capsize=4,
                capthick=1.5,
            )
        else:
            # Plot line without error bars
            ax.plot(
                bin_centers,
                counts,
                marker=marker,
                linewidth=2,
                label=group,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.5,
                alpha=0.8,
            )

        # Find and mark the peak (maximum count)
        peak_idx = np.argmax(counts)
        peak_center = bin_centers[peak_idx]
        peak_count = counts[peak_idx]

        # Add a star marker at the peak
        ax.plot(
            peak_center,
            peak_count,
            marker="*",
            markersize=15,
            color=color,
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=10,  # Ensure it appears on top
        )

    # Add overall dataset line
    # Determine dataset split for legend
    if "split" in scored_df.columns:
        unique_splits = scored_df["split"].unique()
        if len(unique_splits) == 1:
            dataset_split = unique_splits[0]
        else:
            dataset_split = "full"
    else:
        dataset_split = "full"

    # Calculate overall histogram
    all_data = scored_df["genericity_score"]
    counts_overall, _ = np.histogram(all_data, bins=10, range=(0, 1))

    if show_std:
        # Calculate standard deviation using bootstrap for overall
        n_bootstrap = 100
        bootstrap_counts_overall = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(all_data, size=len(all_data), replace=True)
            boot_counts, _ = np.histogram(sample, bins=bin_edges)
            bootstrap_counts_overall.append(boot_counts)

        bootstrap_counts_overall = np.array(bootstrap_counts_overall)
        counts_std_overall = np.std(bootstrap_counts_overall, axis=0)

        # Plot overall line with error bars
        ax.errorbar(
            bin_centers,
            counts_overall,
            yerr=counts_std_overall,
            marker="s",
            linewidth=3,
            color="black",
            markersize=8,
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="--",
            alpha=0.8,
            capsize=4,
            capthick=1.5,
            zorder=5,
        )
    else:
        # Plot overall line without error bars
        ax.plot(
            bin_centers,
            counts_overall,
            marker="s",
            linewidth=3,
            color="black",
            markersize=8,
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="--",
            alpha=0.8,
            zorder=5,
        )

    # Find and mark the overall peak with X marker
    peak_idx_overall = np.argmax(counts_overall)
    peak_center_overall = bin_centers[peak_idx_overall]
    peak_count_overall = counts_overall[peak_idx_overall]

    ax.plot(
        peak_center_overall,
        peak_count_overall,
        marker="X",
        markersize=15,
        color="black",
        markeredgecolor="black",
        markeredgewidth=1,
        zorder=10,
    )

    # Add overall to legend last (using empty plot for legend ordering)
    ax.plot(
        [],
        [],
        color="black",
        linestyle="--",
        marker="s",
        label=f"Overall ({dataset_split})",
    )

    ax.set_xlabel("Genericity Bins", fontsize=12)
    ax.set_ylabel("Number of Texts", fontsize=12)
    ax.set_title(
        f"Genericity Score Distribution by {compare_col.capitalize()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Set x-axis ticks at bin centers with bin range labels
    ax.set_xticks(bin_centers)
    bin_labels = [
        f"{edge:.1f}-{bin_edges[i + 1]:.1f}" for i, edge in enumerate(bin_edges[:-1])
    ]
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins="auto", prune=None))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_model_score_distribution_rectangular(
    scored_df: pd.DataFrame,
    save_path: str = None,
    compare_col: str = "model",
    show_std: bool = True,
):
    """
    Plot score distributions in a more rectangular layout for better readability.

    Args:
        scored_df: DataFrame with scored texts
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
        show_std: Whether to show standard deviation in stats box (default: True)
    """
    groups = scored_df[compare_col].unique()
    n_groups = len(groups)
    colors = get_model_colors(groups)

    # Calculate grid dimensions (prefer wider than tall)
    if n_groups <= 2:
        nrows, ncols = 1, n_groups
    elif n_groups <= 4:
        nrows, ncols = 2, 2
    elif n_groups <= 6:
        nrows, ncols = 2, 3
    elif n_groups <= 8:
        nrows, ncols = 2, 4
    else:
        nrows = int(np.ceil(n_groups / 4))
        ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    # Handle single subplot case
    if n_groups == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if hasattr(axes, "__iter__") else [axes]
    else:
        axes = axes.flatten()

    for idx, group in enumerate(groups):
        group_data = scored_df[scored_df[compare_col] == group]["genericity_score"]

        # Create histogram
        axes[idx].hist(
            group_data,
            bins=30,
            alpha=0.7,
            density=True,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add statistics
        mean_score = group_data.mean()
        std_score = group_data.std()
        median_score = group_data.median()

        axes[idx].axvline(
            mean_score, color="red", linestyle="--", linewidth=2, label="Mean"
        )
        axes[idx].axvline(
            median_score, color="orange", linestyle=":", linewidth=2, label="Median"
        )

        axes[idx].set_title(f"{group}\n(n={len(group_data)})", fontweight="bold")
        axes[idx].set_xlabel("Genericity Score")
        axes[idx].set_ylabel("Density")
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()

        # Add text box with statistics
        if show_std:
            stats_text = f"μ = {mean_score:.3f}\nσ = {std_score:.3f}\nmedian = {median_score:.3f}"
        else:
            stats_text = f"μ = {mean_score:.3f}\nmedian = {median_score:.3f}"

        axes[idx].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def plot_length_correlation(
    length_scored_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    save_path: str = None,
    compare_col: str = "model",
):
    """
    Plot correlation between text length and genericity scores.
    Uses pre-computed correlation DataFrame.

    Args:
        length_scored_df: DataFrame with length and score data
        correlation_df: DataFrame with correlation statistics
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
    """
    # Get groups and colors
    groups = correlation_df[compare_col].values
    colors = get_model_colors(groups)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Correlation with number of clauses
    clause_correlations = correlation_df["correlation_clauses"].values
    bars1 = axes[0].bar(
        range(len(groups)), clause_correlations, color=colors, alpha=0.7
    )
    axes[0].set_xlabel(compare_col.capitalize())
    axes[0].set_ylabel("Correlation Coefficient")
    axes[0].set_title("Correlation: Genericity vs Number of Clauses")
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups, rotation=45)
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[0].grid(alpha=0.3)

    # Add correlation values on bars
    for bar, corr in zip(bars1, clause_correlations):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.01 if height >= 0 else -0.02),
            f"{corr:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # Plot 2: Correlation with number of words
    word_correlations = correlation_df["correlation_words"].values
    bars2 = axes[1].bar(range(len(groups)), word_correlations, color=colors, alpha=0.7)
    axes[1].set_xlabel(compare_col.capitalize())
    axes[1].set_ylabel("Correlation Coefficient")
    axes[1].set_title("Correlation: Genericity vs Number of Words")
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups, rotation=45)
    axes[1].set_ylim(-0.3, 0.3)
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].grid(alpha=0.3)

    # Add correlation values on bars
    for bar, corr in zip(bars2, word_correlations):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.01 if height >= 0 else -0.02),
            f"{corr:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def plot_length_distribution(length_scored_df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of text lengths (clauses and words).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Distribution of number of clauses
    axes[0].hist(
        length_scored_df["num_clauses"],
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Number of Clauses")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Text Length (Clauses)")
    axes[0].grid(alpha=0.3)

    # Add statistics
    mean_clauses = length_scored_df["num_clauses"].mean()
    median_clauses = length_scored_df["num_clauses"].median()
    axes[0].axvline(
        mean_clauses, color="red", linestyle="--", linewidth=2, label="Mean"
    )
    axes[0].axvline(
        median_clauses, color="orange", linestyle=":", linewidth=2, label="Median"
    )
    axes[0].legend()

    # Plot 2: Distribution of number of words
    axes[1].hist(
        length_scored_df["num_words"],
        bins=50,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Text Length (Words)")
    axes[1].grid(alpha=0.3)

    # Add statistics
    mean_words = length_scored_df["num_words"].mean()
    median_words = length_scored_df["num_words"].median()
    axes[1].axvline(mean_words, color="red", linestyle="--", linewidth=2, label="Mean")
    axes[1].axvline(
        median_words, color="orange", linestyle=":", linewidth=2, label="Median"
    )
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def plot_length_relationship_lines(
    length_scored_df: pd.DataFrame,
    save_path: str = None,
    compare_col: str = "model",
    show_std: bool = True,
):
    """
    Plot (lines) relationship between text length and genericity using line plots.
    Shows binned averages for better readability, with shaded regions for standard deviation.

    Args:
        length_scored_df: DataFrame with length and score data
        save_path: Path to save the plot
        compare_col: Column name being compared (default: "model")
        show_std: Whether to show standard deviation shaded regions (default: True)
    """
    # Determine dataset split for legend
    if "split" in length_scored_df.columns:
        # If there are multiple splits, use "full", otherwise use the single split
        unique_splits = length_scored_df["split"].unique()
        if len(unique_splits) == 1:
            dataset_split = unique_splits[0]
        else:
            dataset_split = "full"
    else:
        dataset_split = "full"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    groups = length_scored_df[compare_col].unique()
    colors = get_model_colors(groups)  # Use consistent colors
    markers = get_model_markers(groups)  # Use distinct markers

    # Plot 1: Genericity vs Number of Clauses (binned line plot with shaded regions)
    for group, color, marker in zip(groups, colors, markers):
        group_data = length_scored_df[length_scored_df[compare_col] == group]

        # Create bins for number of clauses
        clause_bins = pd.cut(group_data["num_clauses"], bins=10)
        binned_data = group_data.groupby(clause_bins)["genericity_score"].agg(
            ["mean", "std"]
        )

        # Get bin centers, means, and stds, filtering out NaN values
        bin_centers = []
        means = []
        stds = []
        for interval in binned_data.index:
            if pd.notna(interval):
                mean_score = binned_data.loc[interval, "mean"]
                std_score = binned_data.loc[interval, "std"]
                if pd.notna(mean_score):
                    bin_centers.append(interval.mid)
                    means.append(mean_score)
                    stds.append(std_score if pd.notna(std_score) else 0)

        if bin_centers and means:
            # Plot line
            axes[0].plot(
                bin_centers,
                means,
                marker=marker,  # Use distinct marker
                linewidth=2,
                label=group,
                color=color,
                markersize=8,  # Slightly larger markers
                alpha=0.8,
            )
            # Add shaded region for standard deviation
            if show_std:
                axes[0].fill_between(
                    bin_centers,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color,
                    alpha=0.2,
                )

    # Add overall dataset line for clauses with std
    clause_bins_overall = pd.cut(length_scored_df["num_clauses"], bins=10)
    binned_data_overall = length_scored_df.groupby(clause_bins_overall)[
        "genericity_score"
    ].agg(["mean", "std"])

    bin_centers_overall = []
    means_overall = []
    stds_overall = []
    for interval in binned_data_overall.index:
        if pd.notna(interval):
            mean_score = binned_data_overall.loc[interval, "mean"]
            std_score = binned_data_overall.loc[interval, "std"]
            if pd.notna(mean_score):
                bin_centers_overall.append(interval.mid)
                means_overall.append(mean_score)
                stds_overall.append(std_score if pd.notna(std_score) else 0)

    if bin_centers_overall and means_overall:
        # Plot overall line with shaded error region
        axes[0].plot(
            bin_centers_overall,
            means_overall,
            marker="s",  # Different marker (square)
            linewidth=3,
            label=f"Overall ({dataset_split})",
            color="black",
            markersize=8,
            linestyle="--",  # Dashed line to distinguish
        )
        # Add shaded region for standard deviation
        if show_std:
            axes[0].fill_between(
                bin_centers_overall,
                [m - s for m, s in zip(means_overall, stds_overall)],
                [m + s for m, s in zip(means_overall, stds_overall)],
                color="black",
                alpha=0.3,  # Slightly more opaque for the overall line
            )

    axes[0].set_xlabel("Number of Clauses")
    axes[0].set_ylabel("Average Genericity Score")
    axes[0].set_title("Genericity vs Number of Clauses (Binned)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Genericity vs Number of Words (binned line plot with shaded regions)
    for group, color, marker in zip(groups, colors, markers):
        group_data = length_scored_df[length_scored_df[compare_col] == group]

        # Create bins for number of words
        word_bins = pd.cut(group_data["num_words"], bins=10)
        binned_data = group_data.groupby(word_bins)["genericity_score"].agg(
            ["mean", "std"]
        )

        # Get bin centers, means, and stds, filtering out NaN values
        bin_centers = []
        means = []
        stds = []
        for interval in binned_data.index:
            if pd.notna(interval):
                mean_score = binned_data.loc[interval, "mean"]
                std_score = binned_data.loc[interval, "std"]
                if pd.notna(mean_score):
                    bin_centers.append(interval.mid)
                    means.append(mean_score)
                    stds.append(std_score if pd.notna(std_score) else 0)

        if bin_centers and means:
            # Plot line
            axes[1].plot(
                bin_centers,
                means,
                marker=marker,
                linewidth=2,
                label=group,
                color=color,
                markersize=6,
                alpha=0.8,
            )
            # Add shaded region for standard deviation
            if show_std:
                axes[1].fill_between(
                    bin_centers,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color,
                    alpha=0.2,
                )

    # Add overall dataset line for words with std
    word_bins_overall = pd.cut(length_scored_df["num_words"], bins=10)
    binned_data_overall = length_scored_df.groupby(word_bins_overall)[
        "genericity_score"
    ].agg(["mean", "std"])

    bin_centers_overall = []
    means_overall = []
    stds_overall = []
    for interval in binned_data_overall.index:
        if pd.notna(interval):
            mean_score = binned_data_overall.loc[interval, "mean"]
            std_score = binned_data_overall.loc[interval, "std"]
            if pd.notna(mean_score):
                bin_centers_overall.append(interval.mid)
                means_overall.append(mean_score)
                stds_overall.append(std_score if pd.notna(std_score) else 0)

    if bin_centers_overall and means_overall:
        # Plot overall line with shaded error region
        axes[1].plot(
            bin_centers_overall,
            means_overall,
            marker="s",  # Different marker (square)
            linewidth=3,
            label=f"Overall ({dataset_split})",
            color="black",
            markersize=8,
            linestyle="--",  # Dashed line to distinguish
        )
        # Add shaded region for standard deviation
        if show_std:
            axes[1].fill_between(
                bin_centers_overall,
                [m - s for m, s in zip(means_overall, stds_overall)],
                [m + s for m, s in zip(means_overall, stds_overall)],
                color="black",
                alpha=0.3,  # Slightly more opaque for the overall line
            )

    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Average Genericity Score")
    axes[1].set_title("Genericity vs Number of Words (Binned)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    _show_plot()


def sanity_check_histogram_counts(scored_df: pd.DataFrame, compare_col: str = "model"):
    """
    Sanity check: Verify that all texts are accounted for in histogram bins.
    Prints the total texts per group and confirms they match the bin counts.

    Args:
        scored_df: DataFrame with scored texts
        compare_col: Column name being compared (default: "model")
    """
    print("\n" + "=" * 60)
    print("HISTOGRAM SANITY CHECK")
    print("=" * 60)

    groups = scored_df[compare_col].unique()

    for group in groups:
        group_data = scored_df[scored_df[compare_col] == group]["genericity_score"]
        counts, bin_edges = np.histogram(group_data, bins=10, range=(0, 1))

        total_in_bins = counts.sum()
        total_in_data = len(group_data)

        match = "✓" if total_in_bins == total_in_data else "✗"

        print(
            f"{group:20s}: {total_in_data:6d} texts | Bins sum: {total_in_bins:6d} {match}"
        )

        if total_in_bins != total_in_data:
            print(f"  WARNING: Mismatch of {total_in_data - total_in_bins} texts!")

    print("=" * 60 + "\n")
