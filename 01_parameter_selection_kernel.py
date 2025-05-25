import logging

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM

from data_loader import CustomDataLoader


def evaluate_kernel(
    model_name, train_embeddings, val_embeddings, val_labels, kernel_list
):
    train_embeddings = train_embeddings.astype(np.float32)
    val_embeddings = val_embeddings.astype(np.float32)

    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)
    kf_val = KFold(n_splits=10, shuffle=True, random_state=42)

    kernel_scores = {kernel: [] for kernel in kernel_list}

    for kernel in kernel_list:
        logging.info(f"Evaluating kernel: {kernel}")
        for fold_idx, ((train_idx, _), (val_train_idx, _)) in enumerate(
            zip(kf_train.split(train_embeddings), kf_val.split(val_embeddings))
        ):
            X_train = train_embeddings[train_idx]
            X_val = val_embeddings[val_train_idx]
            y_val = val_labels[val_train_idx]

            model = OneClassSVM(kernel=kernel)
            model.fit(X_train)

            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred == 1).astype(int)

            f1 = f1_score(y_val, y_pred_binary)
            kernel_scores[kernel].append(f1)
            logging.info(f"Fold {fold_idx + 1} - Kernel {kernel} - F1-score: {f1:.4f}")

    return kernel_scores


def statistical_analysis(kernel_scores):
    logging.info("Starting Kruskal-Wallis test for all kernel scores.")
    score_lists = list(kernel_scores.values())
    stat, p_value = kruskal(*score_lists)

    logging.info(f"Kruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        logging.info("Significant difference found, performing post-hoc Dunn test.")

        # Prepare data as a long-form pandas DataFrame
        data = []
        for kernel, scores in kernel_scores.items():
            for score in scores:
                data.append({"kernel": kernel, "accuracy": score})
        df = pd.DataFrame(data)

        # Dunn test
        dunn_result = sp.posthoc_dunn(
            df, val_col="accuracy", group_col="kernel", p_adjust="bonferroni"
        )
        logging.info("Dunn's test results (Bonferroni corrected):\n%s", dunn_result)

        # Select best kernel by median accuracy
        median_scores = {k: np.median(v) for k, v in kernel_scores.items()}
        best_kernel = max(median_scores, key=median_scores.get)
        logging.info(
            f"Selected best kernel: {best_kernel} with median accuracy: {median_scores[best_kernel]:.4f}"
        )
    else:
        logging.info("No significant difference found between kernels.")
        best_kernel = None

    return best_kernel


def main():
    loader = CustomDataLoader()
    model_names = ["xlm-roberta", "labse", "distill_bert"]
    best_kernels = list()
    for model_name in model_names:
        train_embeddings = loader.load_train_embeddings(model_name)
        val_embeddings, val_labels = loader.load_val_embeddings(model_name)
        # Set up logging
        logging.basicConfig(
            filename="data/logging/01_parameter_selection_kernel.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Testing for: {model_name}")
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        kernel_scores = evaluate_kernel(
            model_name, train_embeddings, val_embeddings, val_labels, kernels
        )
        best_kernel = statistical_analysis(kernel_scores)

        logging.info("Final kernel F1-scores per fold:")
        for kernel, scores in kernel_scores.items():
            logging.info(f"{kernel}: {scores}")

        if best_kernel:
            logging.info(f"Best performing kernel: {best_kernel}")
            best_kernels.append(best_kernel)
        else:
            logging.info("No single best kernel found.")
    logging.info(f"\n\n\n BEST KERNELS: {best_kernels}")


if __name__ == "__main__":
    main()
