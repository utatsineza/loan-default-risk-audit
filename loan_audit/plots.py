"""Plots of the audit results."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_agreement_plots(scores_df, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(scores_df["spearman"], bins=20)
    axes[0].set_title("Spearman rank correlation (Model A vs B)")
    axes[0].set_xlabel("Spearman")
    axes[1].hist(scores_df["jaccard_top_k"], bins=10)
    axes[1].set_title("Top-k Jaccard index (Model A vs B)")
    axes[1].set_xlabel("Jaccard")
    for ax in axes:
        ax.set_ylabel("Borrowers")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pairs_plot(pairs_df, path: str) -> None:
    """Headline audit numbers for every model pair, so pair-to-pair spread is visible."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [f"Pair {int(i) + 1}" for i in pairs_df["pair_id"]]
    for ax, column, title in (
            (axes[0], "mean_spearman", "Mean Spearman per pair"),
            (axes[1], "mean_jaccard_top_k", "Mean top-k Jaccard per pair"),
            (axes[2], "pct_divergent", "% divergent cases per pair")):
        ax.bar(labels, pairs_df[column])
        ax.axhline(pairs_df[column].mean(), color="black", linestyle="--", linewidth=1,
                   label="mean across pairs")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
