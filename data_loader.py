import json

import numpy as np


class CustomDataLoader:
    """Custom class for loading the data into cpu memory."""

    def __init__(self, models=["xlm-roberta", "labse", "distill_bert"]):
        self.models = models

    def load_train_embeddings(self, model_name, file_path="data/datasets/train.jsonl"):
        embeddings = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if model_name in sample:
                        embeddings.append(sample[model_name])
        return np.array(embeddings)

    def load_test_embeddings(self, model_name, file_path="data/datasets/test.jsonl"):
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
