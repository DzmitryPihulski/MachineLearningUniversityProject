import json
import logging

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data_loader import CustomDataLoader


def evaluate_nu(kernel_name, train_embeddings, test_embeddings, test_labels, nu):
    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)

    # Create n equal parts of test set, where n = len(nu_list)
    test_indices = np.arange(len(test_embeddings))
    np.random.seed(42)  # For reproducible partitioning
    np.random.shuffle(test_indices)

    nu_scores = list()

    logging.info(f"Evaluating nu: {nu}")

    # Create KFold for this test partition
    kf_test_partition = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation on both training set and test set
    statuses = 0
    for fold_idx, ((train_idx, _), (test_train_idx, _)) in enumerate(
        zip(
            kf_train.split(train_embeddings),
            kf_test_partition.split(test_embeddings),
        )
    ):
        X_train = train_embeddings[train_idx]
        X_test = test_embeddings[test_train_idx]
        y_test = test_labels[test_train_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = OneClassSVM(
            kernel=kernel_name, nu=nu, max_iter=2000000, degree=2, tol=0.01
        )
        model.fit(X_train_scaled)

        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred == 1).astype(int)

        f1 = f1_score(y_test, y_pred_binary)
        nu_scores.append(f1)
        logging.info(
            f"Fold {fold_idx + 1} - Nu {nu} - F1-score: {f1:.4f} - Number of inter: {model.n_iter_} - Fit status (0 if correctly fitted): {model.fit_status_}"
        )
        statuses += model.fit_status_

    return nu_scores, statuses


def main():
    loader = CustomDataLoader()
    model_names = ["xlm-roberta", "labse", "distill_bert"]
    kernel_names = [
        "linear",
        "rbf",
        "poly",
        "sigmoid",
    ]

    # Set up logging
    logging.basicConfig(
        filename="data/logging/01_analysis_of_threshold.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Prepare results file
    results_file = "data/datasets/results_nu.jsonl"
    nus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for model_name in model_names:
        logging.info(f"Loading embeddings for model: {model_name}")
        train_embeddings = loader.load_train_embeddings(model_name)
        val_embeddings, val_labels = loader.load_test_embeddings(model_name)

        for kernel_name in kernel_names:
            for nu in nus:
                logging.info(f"Testing model: {model_name}, kernel: {kernel_name}")

                try:
                    # Evaluate this model-kernel combination
                    nu_scores_dict, status = evaluate_nu(
                        kernel_name, train_embeddings, val_embeddings, val_labels, nu
                    )

                    # Prepare result record
                    result = {
                        "model_name": model_name,
                        "kernel_name": kernel_name,
                        "nu": nu,
                        "status": status,
                        "nu_scores": nu_scores_dict,  # Dict with nu as key, list of 10 values as value
                    }

                    # Write to JSONL file
                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")

                    logging.info(
                        f"Completed {model_name}-{kernel_name}: {len(nu_scores_dict)} nus tested, each with 10 values"
                    )

                except Exception as e:
                    logging.error(f"Error testing {model_name}-{kernel_name}: {str(e)}")
                    # Optionally write error record to JSONL
                    error_result = {
                        "model_name": model_name,
                        "kernel_name": kernel_name,
                        "nu": nu,
                        "error": str(e),
                    }
                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result) + "\n")

    logging.info("All model-kernel combinations tested. Results saved to results.jsonl")
    print(f"Testing completed. Results saved to {results_file}")


if __name__ == "__main__":
    main()
