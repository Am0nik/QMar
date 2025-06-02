#preprocess_jsonl.py
import json

def extract_texts_from_jsonl(filepath):
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if text:
                    # Добавляем специальные токены, если они нужны
                    if not text.endswith("<eos>"):
                        text += " <eos>"
                    texts.append(f"<user> {text}")
            except json.JSONDecodeError:
                continue  # пропускаем повреждённые строки
    return texts
