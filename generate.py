# generate.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Пути
MODEL_PATH = 'checkpoints/model.keras'
TOKENIZER_PATH = 'data/processed/vocab.json'
MAX_LEN = 100  # такой же, как при обучении

print("[generate.py] Загрузка модели...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[generate.py] Модель загружена")

print("[generate.py] Загрузка токенизатора...")
with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(f.read())
print("[generate.py] Токенизатор загружен!")

special_tokens = ['<user>', '<assistant>', '<eos>']
for token in special_tokens:
    if token not in tokenizer.word_index:
        print(f"Внимание: токен {token} отсутствует в словаре токенизатора!")
    else:
        print(f"Токен {token} присутствует в словаре, индекс: {tokenizer.word_index[token]}")

def sample_with_temperature(logits, temperature=0.4, top_k=20):
    logits = logits.astype(np.float64)
    logits = logits / temperature
    
    # Запрещаем выбирать токен 0 (padding)
    #logits[0] = -np.inf
    
    top_k = min(top_k, len(logits))
    top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
    top_k_logits = logits[top_k_indices]

    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
    probs = exp_logits / np.sum(exp_logits)

    return np.random.choice(top_k_indices, p=probs)


def generate_reply(input_text, max_len=100, temperature=0.7):#температура по умолчанию 0.6
    # добавляем <user> и <assistant>
    if not input_text.startswith('<user>'):
        input_text = '<user> ' + input_text
    input_text += ' <assistant>'

    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    output_seq = input_seq.copy()

    for _ in range(max_len):
        #длина к нужному размеру
        padded = pad_sequences([output_seq], maxlen=MAX_LEN, padding='pre')

        preds = model.predict(padded, verbose=0)

        # НЕ с нулевого токена
        last_logits = preds[0, -1]

        next_token_id = sample_with_temperature(last_logits, temperature, top_k=40)  # выбираем токен с учетом температуры
        print(f"Следующий токен: {next_token_id} ({tokenizer.index_word.get(next_token_id, '')})")

        # конец 
        if next_token_id == 0 or tokenizer.index_word.get(next_token_id, '') in ['<eos>','<user>']:
            break

        output_seq.append(next_token_id)

    # преобразуем только сгенерированную часть
    generated_words = [tokenizer.index_word.get(i, '') for i in output_seq[len(input_seq):]]
    reply_text = ' '.join(generated_words).strip()

    return reply_text

if __name__ == "__main__":
    print("Введите 'выход' для завершения")
    
    dialog_history = []  # Сохраняем сообщения поочерёдно: <user> ... <assistant> ...
    
    while True:
        user_input = input("Ты: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break

        dialog_history.append(f"<user> {user_input} <assistant>")

        # создаём общую строку из всей истории диалога
        full_prompt = ' '.join(dialog_history)

        print(f"[generate.py] Ввод пользователя(токен): {user_input} ({tokenizer.texts_to_sequences([user_input])[0]})")
        response = generate_reply(full_prompt,temperature=0.4)  # теперь модель видит весь контекст!

        print("ИИ:", response.replace('<assistant>', '').strip())
        dialog_history.append(response + " <eos>")  # сохраняем ответ
