import logging

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM

from data_loader import CustomDataLoader


def evaluate_gamma(
    model_name, train_embeddings, val_embeddings, val_labels, gamma_list
):
    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)
    kf_val = KFold(n_splits=10, shuffle=True, random_state=42)

    gamma_scores = {gamma: [] for gamma in gamma_list}

    for gamma in gamma_list:
        logging.info(f"Evaluating gamma: {gamma}")
        for fold_idx, ((train_idx, _), (val_train_idx, _)) in enumerate(
            zip(kf_train.split(train_embeddings), kf_val.split(val_embeddings))
        ):
            X_train = train_embeddings[train_idx]
            X_val = val_embeddings[val_train_idx]
            y_val = val_labels[val_train_idx]

            model = OneClassSVM(kernel="rbf", nu=0.1, gamma=gamma)
            model.fit(X_train)

            y_pred = model.predict(X_val)
            # OneClassSVM returns +1 for inliers, -1 for outliers
            y_pred_binary = (y_pred == 1).astype(int)

            f1 = f1_score(y_val, y_pred_binary)
            gamma_scores[gamma].append(f1)
            logging.info(f"Fold {fold_idx + 1} - Gamma {gamma} - F1-score: {f1:.4f}")

    return gamma_scores


def statistical_analysis(gamma_scores):
    logging.info("Starting statistical analysis for Gamma scores.")

    gamma_keys = list(gamma_scores.keys())
    if len(gamma_keys) != 2:
        logging.warning(
            "Wilcoxon test is only applicable for 2 groups. Found: %d", len(gamma_keys)
        )
        return None

    gamma1, gamma2 = gamma_keys
    scores1 = gamma_scores[gamma1]
    scores2 = gamma_scores[gamma2]

    # Check equal length
    if len(scores1) != len(scores2):
        logging.error(
            "Gamma score lists are not of equal length! Cannot perform Wilcoxon test."
        )
        return None

    stat, p_value = wilcoxon(scores1, scores2)
    logging.info(f"Wilcoxon test statistic: {stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        median_scores = {gamma1: np.median(scores1), gamma2: np.median(scores2)}
        best_gamma = max(median_scores, key=median_scores.get)
        logging.info(
            f"Significant difference found. Best gamma: {best_gamma} with median accuracy: {median_scores[best_gamma]:.4f}"
        )
    else:
        logging.info("No significant difference found between gamma values.")
        best_gamma = None

    return best_gamma


def main():
    loader = CustomDataLoader()
    model_names = ["xlm-roberta", "labse", "distill_bert"]
    best_gammas = list()
    for model_name in model_names:
        train_embeddings = loader.load_train_embeddings(model_name)
        val_embeddings, val_labels = loader.load_val_embeddings(model_name)
        # Set up logging
        logging.basicConfig(
            filename="data/logging/03_parameter_selection_gamma.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Testing for: {model_name}")
        gammas = ["scale", "auto"]
        gamma_scores = evaluate_gamma(
            model_name, train_embeddings, val_embeddings, val_labels, gammas
        )
        best_gamma = statistical_analysis(gamma_scores)

        logging.info("Final gamma F1-scores per fold:")
        for gamma, scores in gamma_scores.items():
            logging.info(f"{gamma}: {scores}")

        if best_gamma:
            logging.info(f"Best performing gamma: {best_gamma}")
            best_gammas.append(best_gamma)
        else:
            logging.info("No single best gamma found.")
    logging.info(f"\n\n\n BEST GAMMAS: {best_gammas}")


if __name__ == "__main__":
    main()
