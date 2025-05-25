import logging

import numpy as np
import scikit_posthocs as sp
from scipy.stats import kruskal
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data_loader import CustomDataLoader


def evaluate_tol(model_name, train_embeddings, val_embeddings, val_labels, tol_list):
    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)
    kf_val = KFold(n_splits=10, shuffle=True, random_state=42)

    tol_scores = {tol: [] for tol in tol_list}

    for tol in tol_list:
        logging.info(f"Evaluating tol: {tol}")
        for fold_idx, ((train_idx, _), (val_train_idx, _)) in enumerate(
            zip(kf_train.split(train_embeddings), kf_val.split(val_embeddings))
        ):
            X_train = train_embeddings[train_idx]
            X_val = val_embeddings[val_train_idx]
            y_val = val_labels[val_train_idx]

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale", tol=tol)
            model.fit(X_train_scaled)

            y_pred = model.predict(X_val_scaled)
            y_pred_binary = (y_pred == 1).astype(int)

            f1 = f1_score(y_val, y_pred_binary)
            tol_scores[tol].append(f1)
            logging.info(f"Fold {fold_idx + 1} - Tol {tol} - F1-score: {f1:.4f}")

    return tol_scores


def statistical_analysis(tol_scores):
    logging.info("Starting Kruskal-Wallis test for all tol scores.")
    score_lists = list(tol_scores.values())
    stat, p_value = kruskal(*score_lists)

    logging.info(f"Kruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        logging.info("Significant difference found, performing post-hoc Dunn test.")
        all_scores = []
        groups = []

        for tol, scores in tol_scores.items():
            all_scores.extend(scores)
            groups.extend([tol] * len(scores))

        dunn_result = sp.posthoc_dunn([all_scores, groups], p_adjust="bonferroni")
        logging.info("Dunn's test results (Bonferroni corrected):")
        logging.info(f"\n{dunn_result}")

        # Select best tol based on highest median score
        median_scores = {k: np.median(v) for k, v in tol_scores.items()}
        best_tol = max(median_scores, key=median_scores.get)
        logging.info(
            f"Selected best tol: {best_tol} with median accuracy: {median_scores[best_tol]:.4f}"
        )
    else:
        logging.info("No significant difference found between tols.")
        best_tol = None

    return best_tol


def main():
    loader = CustomDataLoader()
    model_names = ["xlm-roberta", "labse", "distill_bert"]
    best_tols = list()
    for model_name in model_names:
        train_embeddings = loader.load_train_embeddings(model_name)
        val_embeddings, val_labels = loader.load_val_embeddings(model_name)
        # Set up logging
        logging.basicConfig(
            filename="data/logging/04_parameter_selection_tol.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Testing for: {model_name}")
        tols = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        tol_scores = evaluate_tol(
            model_name, train_embeddings, val_embeddings, val_labels, tols
        )
        best_tol = statistical_analysis(tol_scores)

        logging.info("Final tol F1-scores per fold:")
        for tol, scores in tol_scores.items():
            logging.info(f"{tol}: {scores}")

        if best_tol:
            logging.info(f"Best performing tol: {best_tol}")
            best_tols.append(best_tol)
        else:
            logging.info("No single best tol found.")
    logging.info(f"\n\n\n BEST TOLS: {best_tols}")


if __name__ == "__main__":
    main()
