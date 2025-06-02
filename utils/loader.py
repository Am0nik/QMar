#loaders.py
import os
from .preprocess_txt import parse_txt_file
from .preprocess_jsonl import extract_texts_from_jsonl


def load_txt_pairs(txt_dir):
    all_pairs = []
    print(f"[loaders.py] Чтение файлов из: {txt_dir}")
    for filename in os.listdir(txt_dir):
        #print(f"[loaders.py] Найден файл: {filename}")
        if not filename.endswith('.txt'):
            print(f"[loaders.py] Пропущен (не .txt): {filename}")
            continue
        filepath = os.path.join(txt_dir, filename)
        #print(f"[loaders.py] Обработка файла: {filepath}")
        try:
            pairs = parse_txt_file(filepath)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"[loaders.py] Ошибка чтения {filename}: {e}")
    print(f"[loaders.py] Всего пар загружено: {len(all_pairs)}")
    return all_pairs


def load_jsonl_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(folder_path, filename)
            extracted = extract_texts_from_jsonl(filepath)
            texts.extend(extracted)
    return texts
