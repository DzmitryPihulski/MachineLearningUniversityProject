import json
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM

train_file = "data/train.jsonl"
test_file = "data/test.jsonl"

models = ["xlm-roberta", "labse", "distill_bert"]

logging.basicConfig(
    filename="data/logging/final_evaluation_on_test_set.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_train_embeddings(file_path, model_name):
    embeddings = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if model_name in sample:
                    embeddings.append(sample[model_name])
    return np.array(embeddings)


def load_test_embeddings(file_path, model_name):
    embeddings = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if model_name in sample:
                    embeddings.append(sample[model_name])
                    labels.append(sample["target"])
    return np.array(embeddings), np.array(labels)


def train_oneclass_svm(train_data):
    logging.info("Trenowanie modelu One-Class SVM...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale", tol=1e-3)
    ocsvm.fit(train_data)
    logging.info("Trenowanie zakończone.")
    return ocsvm


def evaluate_model(model, test_data, test_labels):
    if test_data.shape[0] == 0:
        logging.error("Zbiór testowy jest pusty. Pomijanie oceny.")
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}
    logging.info("Ocena modelu na zbiorze testowym...")
    predictions = model.predict(test_data)
    normalized_predictions = (predictions == 1).astype(int)
    accuracy = accuracy_score(test_labels, normalized_predictions)
    precision = precision_score(test_labels, normalized_predictions)
    recall = recall_score(test_labels, normalized_predictions)
    f1 = f1_score(test_labels, normalized_predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


results = {}

for model_name in models:
    logging.info(f"\n=== Przetwarzanie dla modelu: {model_name} ===")
    train_embeddings = load_train_embeddings(train_file, model_name)
    test_embeddings, test_labels = load_test_embeddings(test_file, model_name)

    logging.info(f"Rozmiar zbioru treningowego:{train_embeddings.shape}")
    logging.info(f"Rozmiar zbioru testowego:{test_embeddings.shape}")

    if train_embeddings.size == 0 or test_embeddings.size == 0:
        logging.error(f"Pominięcie modelu {model_name} – brak danych.")
        continue

    ocsvm = train_oneclass_svm(train_embeddings)
    metrics = evaluate_model(ocsvm, test_embeddings, test_labels)
    results[model_name] = metrics

logging.info("\n=== Podsumowanie wyników ===")
for name, m in results.items():
    logging.info(f"Wyniki dla modelu: {name}:")
    logging.info(f"Accuracy: {m['accuracy']:.2f}")
    logging.info(f"Precision: {m['precision']:.2f}")
    logging.info(f"Recall: {m['recall']:.2f}")
    logging.info(f"F1-Score: {m['f1']:.2f}")
    logging.info("-" * 40)
