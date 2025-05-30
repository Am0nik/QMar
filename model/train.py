import os
import json
import numpy as np
from model.text_encoder import TextEncoder
from model.gru_model import build_model
from sklearn.model_selection import train_test_split
from utils.loader import load_jsonl_texts, load_txt_pairs
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'vocab.json')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'model.keras')
TXT_PATH = os.path.join(BASE_DIR, 'data', 'raw_text')
JSONL_DIR = os.path.join(BASE_DIR, 'data', 'corpora_jsonl')



num_epochs = 20
max_length = 100

def train_model():
    print("[train.py] Загрузка .txt и .jsonl файлов...")
    jsonl_texts = load_jsonl_texts(JSONL_DIR)
    txt_pairs = load_txt_pairs(TXT_PATH)
    txt_texts = [q for q, _ in txt_pairs] + [a for _, a in txt_pairs]
    
    texts = jsonl_texts + txt_texts
    print(f"[train.py] Загружено текстов: {len(texts)}")
    print(f"[train.py] Пример текста: {texts[0]}")

    encoder = TextEncoder()
    print("[train.py] Фитинг токенизатора...")
    encoder.fit(texts)

    print("[train.py] Подготовка обучающих последовательностей...")
    X, y = [], []
    for text in texts:
        tokens = encoder.encode(text, max_length + 1)
        if len(tokens) < 2:
            continue
        X.append(tokens[:-1])
        y.append(tokens[1:])

    X = np.array(X)
    y = np.array(y)
    print(f"[train.py] Размерность X: {X.shape}, y: {y.shape}")

    print("[train.py] Разделение на train и val...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    print("[train.py] Построение модели...")
    vocab_size = len(encoder.tokenizer.word_index) + 1
    model = build_model(vocab_size=vocab_size)

    print("[train.py] Компиляция и обучение...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    model.save(CHECKPOINT_PATH)
    print(f"[train.py] Модель сохранена в {CHECKPOINT_PATH}")

    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        f.write(encoder.tokenizer.to_json())
    print(f"[train.py] Токенизатор сохранён в {VOCAB_PATH}")

if __name__ == "__main__":
    train_model()
