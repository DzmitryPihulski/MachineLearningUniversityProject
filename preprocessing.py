import json
import os
import re
from transformers import BertTokenizer, AutoTokenizer

with open("Desktop/extracted_text_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
for articles in data.values():
    for article in articles:
        if 'text' in article:
            texts.append(article['text'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_texts = [clean_text(t) for t in texts]

#Tokenization
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
distillbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def tokenize_texts(texts, tokenizer, model_name):
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    os.makedirs(f"results/tokenized/{model_name}", exist_ok=True)
    
    input_ids = encoded['input_ids'].tolist()
    attention_mask = encoded['attention_mask'].tolist()

    with open(f"results/tokenized/{model_name}/input_ids.json", "w", encoding="utf-8") as f:
        json.dump(input_ids, f, ensure_ascii=False, indent=2)

    with open(f"results/tokenized/{model_name}/attention_mask.json", "w", encoding="utf-8") as f:
        json.dump(attention_mask, f, ensure_ascii=False, indent=2)

    print(f"Tokenizacja zakończona dla {model_name} i zapisana w: results/tokenized/{model_name}/")

tokenize_texts(cleaned_texts, bert_tokenizer, "bert")
tokenize_texts(cleaned_texts, labse_tokenizer, "labse")
tokenize_texts(cleaned_texts, distillbert_tokenizer, "distillbert")

os.makedirs("results/preprocessing", exist_ok=True)
with open("results/preprocessing/cleaned_texts.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_texts, f, ensure_ascii=False, indent=2)

print("Preprocessing zakończony")