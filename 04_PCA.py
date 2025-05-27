import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import CustomDataLoader

logging.basicConfig(
    filename="data/logging/pca.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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
        self.scaler = StandardScaler()
        self.pca_models = {}
        self.svm_results = {}

        logging.info(f"Data shape: {self.embeddings.shape}")
        logging.info(f"Class distribution: {np.bincount(self.labels)}")

    def research_1_variance_analysis(self, max_components=200):
        """
        Research Question 1: How much variance do we need to retain?
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 1: VARIANCE ANALYSIS")
        logging.info("=" * 60)

        # Standardize data
        embeddings_scaled = self.scaler.fit_transform(self.embeddings)

        # Fit PCA with maximum components
        pca_full = PCA(n_components=min(max_components, self.embeddings.shape[1]))
        pca_full.fit(embeddings_scaled)

        # Calculate cumulative variance
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        # Find components for different variance thresholds
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        optimal_components = {}

        for threshold in thresholds:
            n_comp = np.argmax(cumvar >= threshold) + 1
            optimal_components[threshold] = n_comp
            logging.info(
                f"Components needed for {threshold * 100:.0f}% variance: {n_comp}"
            )

        # Plot variance explained
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(cumvar) + 1), cumvar, "bo-", alpha=0.7)
        for threshold in thresholds:
            n_comp = optimal_components[threshold]
            plt.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
            plt.axvline(x=n_comp, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Variance Explained by PCA Components")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, min(51, len(pca_full.explained_variance_ratio_) + 1)),
            pca_full.explained_variance_ratio_[:50],
            "ro-",
            alpha=0.7,
        )
        plt.xlabel("Component Number")
        plt.ylabel("Individual Explained Variance Ratio")
        plt.title("Individual Component Variance (First 50)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return optimal_components

    def research_2_visualization_analysis(self):
        """
        Research Question 2: How do classes separate in reduced dimensional space?
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 2: CLASS SEPARATION VISUALIZATION")
        logging.info("=" * 60)

        embeddings_scaled = self.scaler.fit_transform(self.embeddings)

        # Create PCA models for 2D and 3D
        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)

        embeddings_2d = pca_2d.fit_transform(embeddings_scaled)
        embeddings_3d = pca_3d.fit_transform(embeddings_scaled)

        # Store for later use
        self.pca_models["2d"] = pca_2d
        self.pca_models["3d"] = pca_3d

        # Plot 2D and 3D visualizations
        fig = plt.figure(figsize=(15, 6))

        # 2D Plot
        ax1 = fig.add_subplot(1, 2, 1)
        colors = ["red", "blue"]
        for i, label in enumerate([0, 1]):
            mask = self.labels == label
            ax1.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[i],
                alpha=0.6,
                label=f"Class {label}",
                s=20,
            )
        ax1.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.3f} variance)")
        ax1.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.3f} variance)")
        ax1.set_title("2D PCA Projection of Text Embeddings")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 3D Plot
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        for i, label in enumerate([0, 1]):
            mask = self.labels == label
            ax2.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=colors[i],
                alpha=0.6,
                label=f"Class {label}",
                s=20,
            )
        ax2.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]:.3f})")
        ax2.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]:.3f})")
        ax2.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]:.3f})")
        ax2.set_title("3D PCA Projection of Text Embeddings")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Calculate separation metrics
        logging.info(
            f"2D Variance captured: {sum(pca_2d.explained_variance_ratio_):.4f}"
        )
        logging.info(
            f"3D Variance captured: {sum(pca_3d.explained_variance_ratio_):.4f}"
        )

    # def research_3_oneclass_svm_comparison(
    #     self, components_list=[50, 100, 150, 200, 768]
    # ):
    #     """
    #     Research Question 3: How does PCA dimensionality affect OneClassSVM performance?
    #     """
    #     logging.info("\n" + "=" * 60)
    #     logging.info(
    #         "RESEARCH 3: ONECLASS SVM PERFORMANCE WITH DIFFERENT PCA DIMENSIONS"
    #     )
    #     logging.info("=" * 60)

    #     embeddings_scaled = self.scaler.fit_transform(self.embeddings)

    #     # Split data for evaluation
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         embeddings_scaled,
    #         self.labels,
    #         test_size=0.3,
    #         random_state=42,
    #         stratify=self.labels,
    #     )

    #     results = []

    #     for n_components in components_list:
    #         logging.info(f"\nTesting with {n_components} components...")

    #         if n_components == 768:
    #             # Use original dimensions
    #             X_train_pca = X_train
    #             X_test_pca = X_test
    #             variance_retained = 1.0
    #         else:
    #             # Apply PCA
    #             pca = PCA(n_components=min(n_components, X_train.shape[1]))
    #             X_train_pca = pca.fit_transform(X_train)
    #             X_test_pca = pca.transform(X_test)
    #             variance_retained = sum(pca.explained_variance_ratio_)

    #         # Train OneClassSVM on class 1 only (assuming this is the normal class)
    #         # You can modify this based on which class should be considered "normal"
    #         normal_class = 1  # Change this if needed
    #         X_train_normal = X_train_pca[y_train == normal_class]

    #         # Grid search for best parameters
    #         param_grid = {
    #             "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    #             "nu": [0.01, 0.05, 0.1, 0.2, 0.5],
    #         }

    #         svm = OneClassSVM()
    #         grid_search = GridSearchCV(
    #             svm, param_grid, cv=3, scoring="accuracy", n_jobs=-1
    #         )
    #         grid_search.fit(X_train_normal)

    #         best_svm = grid_search.best_estimator_

    #         # Predict on test set
    #         y_pred = best_svm.predict(X_test_pca)
    #         y_pred_binary = (y_pred == 1).astype(int)  # Convert to 0/1 labels

    #         # Calculate metrics
    #         # For OneClassSVM: 1 = normal (inlier), -1 = anomaly (outlier)
    #         # We need to map this to our binary classification problem
    #         y_test_binary = (y_test == normal_class).astype(int)

    #         from sklearn.metrics import (
    #             accuracy_score,
    #             f1_score,
    #             precision_score,
    #             recall_score,
    #         )

    #         accuracy = accuracy_score(y_test_binary, y_pred_binary)
    #         precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    #         recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    #         f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

    #         result = {
    #             "n_components": n_components,
    #             "variance_retained": variance_retained,
    #             "best_params": grid_search.best_params_,
    #             "accuracy": accuracy,
    #             "precision": precision,
    #             "recall": recall,
    #             "f1_score": f1,
    #             "model": best_svm,
    #         }

    #         results.append(result)

    #         logging.info(f"Variance retained: {variance_retained:.4f}")
    #         logging.info(f"Best params: {grid_search.best_params_}")
    #         logging.info(f"Accuracy: {accuracy:.4f}")
    #         logging.info(f"F1-score: {f1:.4f}")

    #     # Store results
    #     self.svm_results = results

    #     # Plot comparison
    #     self.plot_svm_comparison(results)

    #     return results

    def plot_svm_comparison(self, results):
        """Plot OneClassSVM performance comparison"""
        df_results = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ["accuracy", "precision", "recall", "f1_score"]
        titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            ax.plot(
                df_results["n_components"],
                df_results[metric],
                "bo-",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel("Number of PCA Components")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs PCA Components")
            ax.grid(True, alpha=0.3)

            # Highlight best performance
            best_idx = df_results[metric].idxmax()
            best_comp = df_results.loc[best_idx, "n_components"]
            best_score = df_results.loc[best_idx, metric]
            ax.scatter(best_comp, best_score, color="red", s=100, zorder=5)
            ax.annotate(
                f"Best: {best_comp} comp\n{best_score:.4f}",
                xy=(best_comp, best_score),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        plt.tight_layout()
        plt.show()

        # Print summary
        logging.info("\nSUMMARY OF RESULTS:")
        logging.info("-" * 50)
        for result in results:
            logging.info(
                f"Components: {result['n_components']:3d} | "
                f"Variance: {result['variance_retained']:.3f} | "
                f"F1: {result['f1_score']:.4f} | "
                f"Accuracy: {result['accuracy']:.4f}"
            )

    def research_4_component_analysis(self, n_components=50):
        """
        Research Question 4: What do the principal components represent?
        """
        logging.info("\n" + "=" * 60)
        logging.info("RESEARCH 4: PRINCIPAL COMPONENT ANALYSIS")
        logging.info("=" * 60)

        embeddings_scaled = self.scaler.fit_transform(self.embeddings)

        # Fit PCA
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings_scaled)

        # Analyze component contributions
        logging.info(f"Analyzing first {n_components} components...")
        logging.info(
            f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}"
        )

        # Plot component importance
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.bar(range(1, min(21, n_components + 1)), pca.explained_variance_ratio_[:20])
        plt.xlabel("Component Number")
        plt.ylabel("Explained Variance Ratio")
        plt.title("First 20 Components Variance")
        plt.xticks(range(1, min(21, n_components + 1)))

        plt.subplot(1, 3, 2)
        component_norms = np.linalg.norm(pca.components_, axis=1)
        plt.plot(range(1, len(component_norms) + 1), component_norms, "go-")
        plt.xlabel("Component Number")
        plt.ylabel("Component Vector Norm")
        plt.title("Component Vector Magnitudes")

        plt.subplot(1, 3, 3)
        # Show loadings heatmap for first few components
        n_show = min(10, n_components)
        loadings = pca.components_[:n_show, :50]  # First 50 original features
        sns.heatmap(
            loadings,
            cmap="RdBu_r",
            center=0,
            xticklabels=range(1, 51),
            yticklabels=range(1, n_show + 1),
        )
        plt.title("Component Loadings (First 50 features)")
        plt.xlabel("Original Feature Index")
        plt.ylabel("Principal Component")

        plt.tight_layout()
        plt.show()

        # Analyze class discrimination in PCA space
        self.analyze_class_discrimination_pca(pca, embeddings_scaled, n_components)

        return pca

    def analyze_class_discrimination_pca(self, pca, embeddings_scaled, n_components):
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

        logging.info(f"\nMost discriminative components (by class mean difference):")
        for i, comp_idx in enumerate(discriminative_components):
            logging.info(
                f"Component {comp_idx + 1}: Mean difference = {mean_diff[comp_idx]:.4f}"
            )

        # Plot discriminative power
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, min(21, len(mean_diff) + 1)), mean_diff[:20])
        plt.xlabel("Component Number")
        plt.ylabel("Absolute Mean Difference Between Classes")
        plt.title("Discriminative Power of Components")

        # Show distribution of most discriminative component
        plt.subplot(1, 2, 2)
        most_disc_comp = discriminative_components[0]
        for label in [0, 1]:
            mask = self.labels == label
            plt.hist(
                embeddings_pca[mask, most_disc_comp],
                alpha=0.6,
                label=f"Class {label}",
                bins=30,
                density=True,
            )
        plt.xlabel(f"PC{most_disc_comp + 1} Values")
        plt.ylabel("Density")
        plt.title(
            f"Distribution of Most Discriminative Component (PC{most_disc_comp + 1})"
        )
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run_complete_research(self):
        """Run all research analyses"""
        logging.info("STARTING COMPREHENSIVE PCA RESEARCH ON TEXT EMBEDDINGS")
        logging.info("=" * 80)

        # Research 1: Variance Analysis
        optimal_components = self.research_1_variance_analysis()

        # Research 2: Visualization
        self.research_2_visualization_analysis()

        # # Research 3: OneClassSVM Performance
        # suggested_components = [
        #     50,
        #     100,
        #     optimal_components[0.90],
        #     optimal_components[0.95],
        #     768,
        # ]
        # svm_results = self.research_3_oneclass_svm_comparison(suggested_components)

        # Research 4: Component Analysis
        self.research_4_component_analysis(50)

        logging.info("\n" + "=" * 80)
        logging.info("RESEARCH COMPLETE!")
        logging.info("=" * 80)

        return {
            "optimal_components": optimal_components,
        }


# Example usage:
loader = CustomDataLoader()
embeddings, labels = loader.load_test_embeddings("xlm-roberta")
framework = PCAResearchFramework(embeddings, labels)
results = framework.run_complete_research()
