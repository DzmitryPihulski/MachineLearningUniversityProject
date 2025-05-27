import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(jsonl_file):
    """Load results from JSONL file and organize by kernel and model"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            if "nu_scores" in data and data["status"] == 0:  # Only successful results
                model = data["model_name"]
                kernel = data["kernel_name"]
                nu = data["nu"]
                scores = data["nu_scores"]

                results[kernel][model][nu] = scores

    return results


def create_plots(results, output_dir="plots", figure_size=(10, 8), dpi=300):
    """Create plots for each kernel showing model performance across nu values"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Define colors for models (using a beautiful color palette)
    model_colors = {
        "xlm-roberta": "#E74C3C",  # Red
        "labse": "#3498DB",  # Blue
        "distill_bert": "#2ECC71",  # Green
    }
    model_name_mapping = {
        "xlm-roberta": "XLM-RoBERTa",
        "labse": "LaBSE",
        "distill_bert": "Distilled BERT",
    }

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    for kernel_name, kernel_data in results.items():
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)

        for model_name, model_data in kernel_data.items():
            # Extract nu values and scores
            nus = sorted(model_data.keys())
            means = []
            stds = []

            for nu in nus:
                scores = model_data[nu]
                means.append(np.mean(scores))
                stds.append(np.std(scores))

            nus = np.array(nus)
            means = np.array(means)
            stds = np.array(stds)

            color = model_colors.get(
                model_name, "#34495E"
            )  # Default color if not found

            # Plot line with error bars
            ax.errorbar(
                nus,
                means,
                yerr=stds,
                label=model_name_mapping[model_name],
                color=color,
                linewidth=2.5,
                capsize=5,
                capthick=2,
                elinewidth=1.5,
                alpha=0.8,
            )

            # Add scatter points
            ax.scatter(
                nus,
                means,
                color=color,
                s=80,
                alpha=0.9,
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
            )

        # Customize plot
        ax.set_xlabel("Nu Value", fontsize=14, fontweight="bold")
        ax.set_ylabel("F-1 Score", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Model Performance Comparison - {kernel_name.upper()} Kernel",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Customize legend
        legend = ax.legend(
            fontsize=12, frameon=True, fancybox=True, shadow=True, loc="best"
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Customize grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

        # Customize ticks
        ax.tick_params(axis="both", which="major", labelsize=11)

        ax.set_ylim(0, 1)

        # Tight layout
        plt.tight_layout()

        # Save plot
        output_file = Path(output_dir) / f"{kernel_name}_kernel_performance.pdf"
        plt.savefig(
            output_file,
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        print(f"Saved plot for {kernel_name} kernel: {output_file}")
        plt.close()


def plot_summary_statistics(results, output_dir="plots"):
    """Create a summary plot showing best performance for each model-kernel combination"""

    summary_data = []

    for kernel_name, kernel_data in results.items():
        for model_name, model_data in kernel_data.items():
            best_nu = None
            best_mean = -np.inf

            for nu, scores in model_data.items():
                mean_score = np.mean(scores)
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_nu = nu

            summary_data.append(
                {
                    "Kernel": kernel_name,
                    "Model": model_name,
                    "Best_Nu": best_nu,
                    "Best_Score": best_mean,
                }
            )

    # Create summary plot
    df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a pivot table for heatmap
    pivot_df = df.pivot(index="Model", columns="Kernel", values="Best_Score")

    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="RdYlBu_r",
        center=pivot_df.values.mean(),
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Best Performance (Mean Scores)", fontsize=16, fontweight="bold", pad=20
    )
    plt.tight_layout()

    # Save summary plot
    summary_file = Path(output_dir) / "performance_summary_heatmap.pdf"
    plt.savefig(summary_file, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved summary heatmap: {summary_file}")
    plt.close()


def main():
    # Configuration
    jsonl_file = "data/datasets/results_nu.jsonl"  # Change this to your JSONL file path
    output_directory = (
        "data/plots/nu_plots"  # Change this to your desired output directory
    )

    print("Loading results from JSONL file...")
    results = load_results(jsonl_file)

    print(f"Found {len(results)} kernels:")
    for kernel in results.keys():
        print(f"  - {kernel}: {len(results[kernel])} models")

    print("\nCreating plots...")
    create_plots(results, output_dir=output_directory)

    print("\nCreating summary statistics...")
    plot_summary_statistics(results, output_dir=output_directory)

    print(f"\nAll plots saved to: {output_directory}/")


if __name__ == "__main__":
    main()
