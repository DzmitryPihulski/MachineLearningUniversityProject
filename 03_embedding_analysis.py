import matplotlib.pyplot as plt
import numpy as np

from data_loader import CustomDataLoader


class EmbeddingAnalyzer:
    def __init__(self):
        """Initialize the analyzer with embedding data from three models."""
        self.data_loader = CustomDataLoader()

        # Load embeddings from all three models (these are lists of embeddings)
        self.xlm_roberta = self.data_loader.load_train_embeddings("xlm-roberta")
        self.labse = self.data_loader.load_train_embeddings("labse")
        self.distill_bert = self.data_loader.load_train_embeddings("distill_bert")

        # Store model names and data in structured format
        self.models = ["xlm-roberta", "labse", "distill_bert"]
        self.embeddings_by_model = {
            "xlm-roberta": self.xlm_roberta,
            "labse": self.labse,
            "distill_bert": self.distill_bert,
        }

        # Results storage
        self.intra_group_results = {}
        self.inter_group_results = {}
        self.model_similarity_results = {}

    def compute_intra_group_variances(self):
        """
        Compute variance within embeddings for each model.
        For each model, compute variance across dimensions (768) for each embedding,
        then get statistics across all embeddings.
        """
        results = {}

        for model in self.models:
            embeddings = np.array(self.embeddings_by_model[model])

            if len(embeddings) > 0:
                # Compute variance across dimensions for each embedding
                # Then compute statistics across all embeddings
                variances_per_embedding = np.var(
                    embeddings, axis=0
                )  # Variance across dimensions
                print(f"Number of dims: {len(variances_per_embedding)}")

                results[model] = {
                    "mean_variance": np.mean(variances_per_embedding),
                    "std_variance": np.std(variances_per_embedding),
                    "min_variance": np.min(variances_per_embedding),
                    "max_variance": np.max(variances_per_embedding),
                    "all_variances": variances_per_embedding,
                }

        self.intra_group_results = results
        return results

    # def compute_inter_group_distances(self):
    #     """
    #     Compute distances between average embeddings of different models.
    #     """
    #     results = {}

    #     # First compute average embeddings for each model
    #     model_averages = {}
    #     for model in self.models:
    #         embeddings = np.array(self.embeddings_by_model[model])
    #         if len(embeddings) > 0:
    #             model_averages[model] = np.mean(embeddings, axis=0)

    #     # Compute pairwise distances between model averages
    #     model_pairs = [
    #         ("xlm-roberta", "labse"),
    #         ("xlm-roberta", "distill_bert"),
    #         ("labse", "distill_bert"),
    #     ]
    #     model_name_mapping = {
    #         "xlm-roberta": "XLM-RoBERTa",
    #         "labse": "LaBSE",
    #         "distill_bert": "Distilled BERT",
    #     }

    #     for model1, model2 in model_pairs:
    #         if model1 in model_averages and model2 in model_averages:
    #             avg1 = model_averages[model1]
    #             avg2 = model_averages[model2]

    #             # Handle different embedding dimensions
    #             min_dim = min(len(avg1), len(avg2))
    #             avg1_trimmed = avg1[:min_dim]
    #             avg2_trimmed = avg2[:min_dim]

    #             # Compute euclidean distance
    #             distance = np.linalg.norm(avg1_trimmed - avg2_trimmed)

    #             pair_key = (
    #                 f"{model_name_mapping[model1]} vs {model_name_mapping[model2]}"
    #             )
    #             results[pair_key] = {
    #                 "distance": distance,
    #                 "mean_distance": distance,  # Keep for compatibility with report format
    #                 "std_distance": 0.0,  # Single distance, so std is 0
    #                 "min_distance": distance,
    #                 "max_distance": distance,
    #             }

    #     self.inter_group_results = results
    #     return results

    def compute_inter_group_distances(self):
        """
        Compute distances between average embeddings of different models using 10-fold cross-validation.
        """
        from sklearn.model_selection import KFold

        results = {}
        n_folds = 10

        # First compute fold-average embeddings for each model
        model_fold_averages = {}

        for model in self.models:
            embeddings = np.array(self.embeddings_by_model[model])
            if len(embeddings) == 0:
                continue

            # Create 10-fold cross-validation splits
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_averages = []

            for train_idx, _ in kf.split(embeddings):
                # Use 9 folds (train_idx) to compute average embedding
                train_embeddings = embeddings[train_idx]
                fold_average = np.mean(train_embeddings, axis=0)
                fold_averages.append(fold_average)

            model_fold_averages[model] = fold_averages

        # Compute pairwise distances between model averages
        model_pairs = [
            ("xlm-roberta", "labse"),
            ("xlm-roberta", "distill_bert"),
            ("labse", "distill_bert"),
        ]
        model_name_mapping = {
            "xlm-roberta": "XLM-RoBERTa",
            "labse": "LaBSE",
            "distill_bert": "Distilled BERT",
        }

        for model1, model2 in model_pairs:
            if model1 in model_fold_averages and model2 in model_fold_averages:
                fold_averages1 = model_fold_averages[model1]
                fold_averages2 = model_fold_averages[model2]

                # Compute all pairwise distances (10x10 = 100 distances)
                distances = []

                for avg1 in fold_averages1:
                    for avg2 in fold_averages2:
                        # Handle different embedding dimensions
                        min_dim = min(len(avg1), len(avg2))
                        avg1_trimmed = avg1[:min_dim]
                        avg2_trimmed = avg2[:min_dim]

                        # Compute euclidean distance
                        distance = np.linalg.norm(avg1_trimmed - avg2_trimmed)
                        distances.append(distance)

                # Compute statistics
                distances = np.array(distances)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                min_distance = np.min(distances)
                max_distance = np.max(distances)

                pair_key = (
                    f"{model_name_mapping[model1]} vs {model_name_mapping[model2]}"
                )
                results[pair_key] = {
                    "distance": mean_distance,  # Keep for compatibility
                    "mean_distance": mean_distance,
                    "std_distance": std_distance,
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                }

        self.inter_group_results = results
        return results

    def generate_report(
        self, output_file="data/logging/03_embedding_analysis_report.log"
    ):
        """
        Generate the complete analysis report and save to file.
        """
        # Compute all analyses
        self.compute_intra_group_variances()
        self.compute_inter_group_distances()

        # Generate report content
        report_lines = []
        report_lines.append("=== EMBEDDING VARIANCE ANALYSIS REPORT ===")
        report_lines.append("")

        # 1. Intra-group variance analysis
        report_lines.append("1. INTRA-GROUP VARIANCE ANALYSIS")
        report_lines.append("----------------------------------------")
        report_lines.append("")

        for model in self.models:
            model_name = model.upper().replace("-", "-")
            if model == "distill_bert":
                model_name = "DISTILL_BERT"
            elif model == "xlm-roberta":
                model_name = "XLM-ROBERTA"
            elif model == "labse":
                model_name = "LABSE"

            results = self.intra_group_results[model]
            report_lines.append(f"{model_name}:")
            report_lines.append(f"  Mean variance: {results['mean_variance']:.6f}")
            report_lines.append(f"  Std variance:  {results['std_variance']:.6f}")
            report_lines.append(f"  Min variance:  {results['min_variance']:.6f}")
            report_lines.append(f"  Max variance:  {results['max_variance']:.6f}")
            report_lines.append("")

        # 2. Inter-group distance analysis
        report_lines.append("")
        report_lines.append("2. INTER-GROUP DISTANCE ANALYSIS")
        report_lines.append("----------------------------------------")
        report_lines.append("")

        for pair_key, results in self.inter_group_results.items():
            pair_name = pair_key.upper().replace(" VS ", " VS ")
            report_lines.append(f"{pair_name}:")
            report_lines.append(
                f"  Distance: {results['distance']:.6f}, STD: {results['std_distance']:.6f}"
            )
            report_lines.append("")

        # Save report to file
        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Report saved to {output_file}")
        return report_lines

    def create_inter_group_distance_plot(
        self,
        output_path="data/plots/embedding_analysis_plots/inter_group_distances.pdf",
    ):
        """Create bar plot showing inter-group distances between model pairs."""
        if not self.inter_group_results:
            self.compute_inter_group_distances()

        pairs = list(self.inter_group_results.keys())
        distances = [self.inter_group_results[pair]["distance"] for pair in pairs]
        std_distances = [
            self.inter_group_results[pair]["std_distance"] for pair in pairs
        ]

        # Clean up pair names for display
        clean_pairs = [pair.replace(" vs ", "\nvs\n") for pair in pairs]

        plt.figure(figsize=(9.5, 6.5))
        bars = plt.bar(
            clean_pairs,
            distances,
            yerr=std_distances,
            color=["#E74C3C", "#3498DB", "#2ECC71"],
            alpha=0.8,
        )

        plt.title(
            "Inter-Group Distances Between Model Pairs", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Model Pairs", fontsize=12)
        plt.ylabel("Euclidean Distance", fontsize=12)
        plt.xticks(rotation=0)

        # Add value labels on bars
        for i, (bar, dist_val) in enumerate(zip(bars, distances)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(distances) * 0.02,
                f"{dist_val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Inter-group distance plot saved to {output_path}")

    def create_intra_group_variance_plot(
        self,
        output_path="data/plots/embedding_analysis_plots/intra_group_variances.pdf",
    ):
        """Create bar plot showing intra-group variances for each model."""
        if not self.intra_group_results:
            self.compute_intra_group_variances()

        models = ["XLM-RoBERTa", "LaBSE", "Distilled BERT"]
        model_reverse_mapping = {
            "XLM-RoBERTa": "xlm-roberta",
            "LaBSE": "labse",
            "Distilled BERT": "distill_bert",
        }
        mean_variances = [
            self.intra_group_results[model_reverse_mapping[model]]["mean_variance"]
            for model in models
        ]
        std_variances = [
            self.intra_group_results[model_reverse_mapping[model]]["std_variance"]
            for model in models
        ]

        plt.figure(figsize=(9.5, 6.5))
        bars = plt.bar(
            models,
            mean_variances,
            yerr=std_variances,
            capsize=5,
            color=["#E74C3C", "#3498DB", "#2ECC71"],
            alpha=0.8,
        )

        plt.title(
            "Average per dimention embeddings variances\n by Model",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Mean Variance", fontsize=12)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(
            zip(bars, mean_variances, std_variances)
        ):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_val + max(mean_variances) * 0.02,
                f"{mean_val:.6f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Intra-group variance plot saved to {output_path}")

    def run_complete_analysis(
        self,
        report_file="data/logging/03_embedding_analysis_report.log",
        plot_dir="plots",
    ):
        """
        Run the complete analysis and generate all outputs.
        """
        # Create output directory for plots if it doesn't exist

        # Generate report
        print("Generating analysis report...")
        self.generate_report(report_file)

        # Create all plots
        print("Creating visualizations...")
        self.create_inter_group_distance_plot()
        self.create_intra_group_variance_plot()

        print("Analysis complete!")


# Usage example
if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = EmbeddingAnalyzer()
    analyzer.run_complete_analysis()
