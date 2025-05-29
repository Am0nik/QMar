#preprocess_jsonl.py
import json

def extract_texts_from_jsonl(filepath):
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    texts.append(obj['text'])
            except json.JSONDecodeError as e:
                print(f"[Ошибка JSONL] Файл: {filepath}, строка {i}: {e}")
    return texts
