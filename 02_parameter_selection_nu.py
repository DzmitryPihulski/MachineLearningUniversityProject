import logging

import numpy as np
import scikit_posthocs as sp
from scipy.stats import kruskal
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from data_loader import CustomDataLoader


def evaluate_nu(model_name, train_embeddings, val_embeddings, val_labels, nu_list):
    kf_train = KFold(n_splits=10, shuffle=True, random_state=42)
    kf_val = KFold(n_splits=10, shuffle=True, random_state=42)

    nu_scores = {nu: [] for nu in nu_list}

    for nu in nu_list:
        logging.info(f"Evaluating nu: {nu}")
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

            model = OneClassSVM(kernel="rbf", nu=nu)
            model.fit(X_train_scaled)

            y_pred = model.predict(X_val_scaled)
            y_pred_binary = (y_pred == 1).astype(int)

            f1 = f1_score(y_val, y_pred_binary)
            nu_scores[nu].append(f1)
            logging.info(f"Fold {fold_idx + 1} - Nu {nu} - F1-score: {f1:.4f}")

    return nu_scores


def statistical_analysis(nu_scores):
    logging.info("Starting Kruskal-Wallis test for all nu scores.")
    score_lists = list(nu_scores.values())
    stat, p_value = kruskal(*score_lists)

    logging.info(f"Kruskal-Wallis H-statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        logging.info("Significant difference found, performing post-hoc Dunn test.")
        all_scores = []
        groups = []

        for nu, scores in nu_scores.items():
            all_scores.extend(scores)
            groups.extend([nu] * len(scores))

        dunn_result = sp.posthoc_dunn([all_scores, groups], p_adjust="bonferroni")
        logging.info("Dunn's test results (Bonferroni corrected):")
        logging.info(f"\n{dunn_result}")

        # Select best nu based on highest median score
        median_scores = {k: np.median(v) for k, v in nu_scores.items()}
        best_nu = max(median_scores, key=median_scores.get)
        logging.info(
            f"Selected best nu: {best_nu} with median accuracy: {median_scores[best_nu]:.4f}"
        )
    else:
        logging.info("No significant difference found between nus.")
        best_nu = None

    return best_nu


def main():
    loader = CustomDataLoader()
    model_names = ["xlm-roberta", "labse", "distill_bert"]
    best_nus = list()
    for model_name in model_names:
        train_embeddings = loader.load_train_embeddings(model_name)
        val_embeddings, val_labels = loader.load_val_embeddings(model_name)
        # Set up logging
        logging.basicConfig(
            filename="data/logging/02_parameter_selection_nu.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Testing for: {model_name}")
        nus = np.arange(0.1, 1.0, 0.1)
        nu_scores = evaluate_nu(
            model_name, train_embeddings, val_embeddings, val_labels, nus
        )
        best_nu = statistical_analysis(nu_scores)

        logging.info("Final nu F1-scores per fold:")
        for nu, scores in nu_scores.items():
            logging.info(f"{nu}: {scores}")

        if best_nu:
            logging.info(f"Best performing nu: {best_nu}")
            best_nus.append(best_nu)
        else:
            logging.info("No single best nu found.")
    logging.info(f"\n\n\n BEST NUS: {best_nus}")


if __name__ == "__main__":
    main()
