import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import CustomDataLoader

logging.basicConfig(
    filename="data/logging/04_PCA.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class_mapping = {
    1: "Political texts",
    0: "Outliers",
}

warnings.filterwarnings("ignore")


class PCAResearchFramework:
    def __init__(self, embeddings, labels):
        """
        Initialize the research framework

        Args:
            embeddings: List of lists, shape (2698, 768)
            labels: List of labels (0 or 1)
        """
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)
        self.pca_models = {}
        self.svm_results = {}

        logging.info(f"Data shape: {self.embeddings.shape}")
        logging.info(f"Class distribution: {np.bincount(self.labels)}")

    def research_1_variance_analysis(
        self, loader, max_components=600, output_path="data/plots/PCA/"
    ):
        """
        Research Question 1: How much variance do we need to retain?
        Compares variance analysis across multiple models.
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 1: VARIANCE ANALYSIS")
        logging.info("=" * 60)

        models = ["xlm-roberta", "labse", "distill_bert"]
        colors = ["#E74C3C", "#3498DB", "#2ECC71"]
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]

        # Store results for all models
        all_cumvar = {}
        all_individual_var = {}
        all_optimal_components = {}

        # Process each model
        for model_name in models:
            logging.info(f"\nProcessing model: {model_name}")

            # Load embeddings for current model
            train_embeds = loader.load_train_embeddings(model_name)
            test_embeds, test_labels = loader.load_test_embeddings(model_name)

            # Combine embeddings
            combined_embeddings = np.concatenate((train_embeds, test_embeds), axis=0)

            # Standardize data
            embeddings_scaled = StandardScaler().fit_transform(combined_embeddings)

            # Fit PCA with maximum components
            pca_full = PCA(
                n_components=min(max_components, combined_embeddings.shape[1])
            )
            pca_full.fit(embeddings_scaled)

            # Calculate cumulative variance
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            all_cumvar[model_name] = cumvar
            all_individual_var[model_name] = pca_full.explained_variance_ratio_

            # Find components for different variance thresholds
            optimal_components = {}
            for threshold in thresholds:
                n_comp = np.argmax(cumvar >= threshold) + 1
                optimal_components[threshold] = n_comp
                logging.info(
                    f"{model_name} - Components needed for {threshold * 100:.0f}% variance: {n_comp}"
                )

            all_optimal_components[model_name] = optimal_components

        # Plot 1: Cumulative variance explained for all models
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(models):
            cumvar = all_cumvar[model_name]
            plt.plot(
                range(1, len(cumvar) + 1),
                cumvar,
                color=colors[i],
                marker="o",
                alpha=0.7,
                markersize=3,
                label=f"{model_name}",
                linewidth=2,
            )

        # Add threshold lines
        for threshold in thresholds:
            plt.axhline(
                y=threshold, color="gray", linestyle="--", alpha=0.4, linewidth=1
            )
            plt.text(
                max_components * 0.8,
                threshold + 0.005,
                f"{threshold * 100:.0f}%",
                fontsize=9,
                alpha=0.7,
            )

        plt.xlabel("Number of Components", fontsize=12)
        plt.ylabel("Cumulative Explained Variance Ratio", fontsize=12)
        plt.title(
            "Cumulative Variance Explained by PCA Components - Model Comparison",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(
            output_path + "Cumulative_explained_var_comparison.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

        # Plot 2: Individual component variance (first 50 components)
        plt.figure(figsize=(10, 6))

        for i, model_name in enumerate(models):
            individual_var = all_individual_var[model_name]
            n_components_to_show = min(50, len(individual_var))
            plt.plot(
                range(1, n_components_to_show + 1),
                individual_var[:n_components_to_show],
                color=colors[i],
                marker="o",
                alpha=0.7,
                markersize=3,
                label=f"{model_name}",
                linewidth=2,
            )

        plt.xlabel("Component Number", fontsize=12)
        plt.ylabel("Individual Explained Variance Ratio", fontsize=12)
        plt.title(
            "Individual Component Variance (First 50) - Model Comparison", fontsize=14
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(
            output_path + "Individual_var_comparison.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

        # Print summary table
        logging.info("\n" + "=" * 80)
        logging.info("VARIANCE THRESHOLD SUMMARY")
        logging.info("=" * 80)
        logging.info(
            f"{'Model':<15} {'80%':<8} {'85%':<8} {'90%':<8} {'95%':<8} {'99%':<8}"
        )
        logging.info("-" * 80)

        for model_name in models:
            components = all_optimal_components[model_name]
            logging.info(
                f"{model_name:<15} {components[0.80]:<8} {components[0.85]:<8} "
                f"{components[0.90]:<8} {components[0.95]:<8} {components[0.99]:<8}"
            )

        return all_optimal_components

    def research_2_visualization_analysis(self, output_path="data/plots/PCA/"):
        """
        Research Question 2: How do classes separate in reduced dimensional space?
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 2: CLASS SEPARATION VISUALIZATION")
        logging.info("=" * 60)

        embeddings_scaled = StandardScaler().fit_transform(self.embeddings)

        # Create PCA models for 2D and 3D
        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)

        embeddings_2d = pca_2d.fit_transform(embeddings_scaled)
        embeddings_3d = pca_3d.fit_transform(embeddings_scaled)

        # Store for later use
        self.pca_models["2d"] = pca_2d
        self.pca_models["3d"] = pca_3d

        plt.figure(figsize=(7, 6))
        # 2D Plot
        colors = ["red", "blue"]
        for i, label in enumerate([0, 1]):
            mask = self.labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[i],
                alpha=0.4,
                label=class_mapping[label],
                s=1,
            )
        plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.3f} variance)")
        plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.3f} variance)")
        plt.title("2D PCA Projection of Text Embeddings")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            output_path + "2D_PCA_Projection.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

        # 3D Plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        for i, label in enumerate([0, 1]):
            mask = self.labels == label
            ax.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=colors[i],
                alpha=0.3,
                label=class_mapping[label],
                s=1,
            )

        ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})")
        ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})")
        ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})")
        ax.set_title("3D PCA Projection of Text Embeddings")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            output_path + "3D_PCA_Projection.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        # Calculate separation metrics
        logging.info(
            f"2D Variance captured: {sum(pca_2d.explained_variance_ratio_):.4f}"
        )
        logging.info(
            f"3D Variance captured: {sum(pca_3d.explained_variance_ratio_):.4f}"
        )

    def research_3_component_analysis(
        self, n_components=50, output_path="data/plots/PCA/"
    ):
        """
        Research Question 3: What do the principal components represent?
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 3: PRINCIPAL COMPONENT ANALYSIS")
        logging.info("=" * 60)

        embeddings_scaled = StandardScaler().fit_transform(self.embeddings)

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit_transform(embeddings_scaled)

        # Analyze component contributions
        logging.info(f"Analyzing first {n_components} components...")
        logging.info(
            f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}"
        )

        # Analyze class discrimination in PCA space
        self.analyze_class_discrimination_pca(pca, embeddings_scaled)

        return pca

    def analyze_class_discrimination_pca(
        self, pca, embeddings_scaled, output_path="data/plots/PCA/"
    ):
        """Analyze how well PCA components discriminate between classes"""
        embeddings_pca = pca.transform(embeddings_scaled)

        # Calculate mean difference between classes for each component
        class_means = {}
        for label in [0, 1]:
            mask = self.labels == label
            class_means[label] = np.mean(embeddings_pca[mask], axis=0)

        mean_diff = np.abs(class_means[1] - class_means[0])

        # Find most discriminative components
        discriminative_components = np.argsort(mean_diff)[::-1][:10]

        logging.info("\nMost discriminative components (by class mean difference):")
        for i, comp_idx in enumerate(discriminative_components):
            logging.info(
                f"Component {comp_idx + 1}: Mean difference = {mean_diff[comp_idx]:.4f}"
            )

        # Show distribution of most discriminative component
        plt.figure(figsize=(6, 4))
        most_disc_comp = discriminative_components[0]
        for label in [0, 1]:
            mask = self.labels == label
            plt.hist(
                embeddings_pca[mask, most_disc_comp],
                alpha=0.6,
                label=class_mapping[label],
                bins=30,
                density=True,
            )
        plt.xlabel(f"PC{most_disc_comp + 1} Values")
        plt.ylabel("Density")
        plt.title(
            f"Distribution of Most Discriminative Component (PC{most_disc_comp + 1})"
        )
        plt.legend()

        plt.savefig(
            output_path + "distr_most_dicriminative_comp.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

    def run_complete_research(self, loader):
        """Run all research analyses"""
        logging.info("STARTING COMPREHENSIVE PCA RESEARCH ON TEXT EMBEDDINGS")
        logging.info("=" * 80)

        # Research 1: Variance Analysis
        optimal_components = self.research_1_variance_analysis(loader)

        # Research 2: Visualization
        self.research_2_visualization_analysis()

        # Research 3: Component Analysis
        self.research_3_component_analysis(50)

        logging.info("\n" + "=" * 80)
        logging.info("RESEARCH COMPLETE!")
        logging.info("=" * 80)

        return {
            "optimal_components": optimal_components,
        }


def main():
    loader = CustomDataLoader()
    train_embeds = loader.load_train_embeddings("xlm-roberta")
    embeddings, labels = loader.load_test_embeddings("xlm-roberta")

    framework = PCAResearchFramework(
        np.concatenate((train_embeds, embeddings), axis=0),
        np.array([1] * len(train_embeds) + list(labels)),
    )
    framework.run_complete_research(loader)


if __name__ == "__main__":
    main()
