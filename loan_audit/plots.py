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
