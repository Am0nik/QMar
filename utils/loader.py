#loaders.py
import os
from .preprocess_txt import parse_txt_file
from .preprocess_jsonl import extract_texts_from_jsonl

def load_txt_pairs(directory):
    all_pairs = []
    for fname in os.listdir(directory):
        if fname.endswith('.txt'):
            all_pairs.extend(parse_txt_file(os.path.join(directory, fname)))
    return all_pairs

def load_jsonl_texts(directory):
    all_texts = []
    for fname in os.listdir(directory):
        if fname.endswith('.jsonl'):
            all_texts.extend(extract_texts_from_jsonl(os.path.join(directory, fname)))
    return all_texts
