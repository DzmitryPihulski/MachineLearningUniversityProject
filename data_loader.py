import json

import numpy as np


class CustomDataLoader:
    def __init__(self, models=["xlm-roberta", "labse", "distill_bert"]):
        self.models = models

    def load_train_embeddings(self, model_name, file_path="data/train.jsonl"):
        embeddings = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if model_name in sample:
                        embeddings.append(sample[model_name])
        return np.array(embeddings)

    def load_val_embeddings(self, model_name, file_path="data/val.jsonl"):
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

    def load_test_embeddings(self, model_name, file_path="data/test.jsonl"):
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
