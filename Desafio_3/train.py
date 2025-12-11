import numpy as np
import pickle
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger

import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Input, TimeDistributed, CategoryEncoding, SimpleRNN, Dense, LSTM, GRU

from tensorflow.keras.models import Sequential

CONFIG = {
  "project_name": "pln-char-generation",
  "model_type": "GRU",
  "max_context_size": 100,
  "batch_size": 256,
  "epochs": 100,
  "hidden_units": 200,
  "dropout": 0.1,
  "optimizer": "rmsprop",
  "val_split": 0.1
}


def load_prepared_data():
  """Carga los archivos generados por preparar_datos.py"""
  if not os.path.exists('vocab.pkl') or not os.path.exists('corpus_tokenizado.npy'):
    raise FileNotFoundError("No se encontraron los datos. Ejecuta primero 'preparar_datos.py'")

  with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

  tokenized_text = np.load('corpus_tokenizado.npy')

  return tokenized_text, vocab_data


def create_dataset(tokenized_text, max_context_size, val_split):
  """Crea X, y usando sliding window"""
  num_val = int(np.ceil(len(tokenized_text) * val_split / max_context_size))

  train_text = tokenized_text[:-num_val * max_context_size]
  val_text = tokenized_text[-num_val * max_context_size:]

  # Secuencias para validación (callback)
  tokenized_sentences_val = [val_text[init * max_context_size:init * (max_context_size + 1)]
                             for init in range(num_val)]

  # Secuencias para entrenamiento
  tokenized_sentences_train = [train_text[init:init + max_context_size + 1]
                               for init in range(len(train_text) - max_context_size)]

  train_sequences = np.array(tokenized_sentences_train)

  X_train = train_sequences[:, :-1]
  y_train = train_sequences[:, 1:]

  X_train = np.expand_dims(X_train, axis=-1)  # (N, 100, 1)

  return X_train, y_train, tokenized_sentences_val


def build_simple_model(vocab_size, config):
  """El modelo exacto de la notebook (SimpleRNN)"""
  model = Sequential()

  # 1. Input -> OneHot
  model.add(TimeDistributed(
    CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot"),
    input_shape=(None, 1)
  ))

  # 2. Celda de Elman
  model.add(SimpleRNN(
    config['hidden_units'],
    return_sequences=True,
    dropout=config['dropout'],
    recurrent_dropout=config['dropout']
  ))

  # 3. Salida
  model.add(Dense(vocab_size, activation='softmax'))

  model.compile(loss='sparse_categorical_crossentropy', optimizer=config['optimizer'])
  return model

def build_lstm_model(vocab_size, config):
    model = Sequential()

    # 1. InputSTM( -> OneHot (Igual que antes)
    model.add(TimeDistributed(
      CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot"),
      input_shape=(None, 1)
    ))

    # 2. Capa LSTM
    # return_sequences=True es vital para que devuelva una secuencia del mismo largo
    model.add(
      config['hidden_units'],
      return_sequences=True,
      dropout=config['dropout'],
      recurrent_dropout=config['dropout']
    )

    # 3. Salida (Igual que antes)
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=config['optimizer'])
    return model


def build_gru_model(vocab_size, config):
  model = Sequential()

  # 1. Input -> OneHot
  model.add(TimeDistributed(
    CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot"),
    input_shape=(None, 1)
  ))

  # 2. Capa GRU (Gated Recurrent Unit)
  # GRU es más eficiente que LSTM y mejor que SimpleRNN para memoria a largo plazo
  model.add(GRU(
    config['hidden_units'],
    return_sequences=True,
    dropout=config['dropout'],
    recurrent_dropout=config['dropout']
  ))

  # 3. Salida
  model.add(Dense(vocab_size, activation='softmax'))

  model.compile(loss='sparse_categorical_crossentropy', optimizer=config['optimizer'])
  return model

class WandbPplCallback(tf.keras.callbacks.Callback):
  """Calcula perplejidad y reporta a W&B"""

  def __init__(self, val_data, max_context_size, patience=5):
    self.val_data = val_data
    self.max_context_size = max_context_size
    self.target = []
    self.padded = []
    self.info = []
    self.min_score = np.inf
    self.patience_counter = 0
    self.patience = patience

    # Preprocesamiento batch
    count = 0
    for seq in self.val_data:
      len_seq = len(seq)
      subseq = [seq[:i] for i in range(1, len_seq)]
      self.target.extend([seq[i] for i in range(1, len_seq)])
      if len(subseq) != 0:
        self.padded.append(pad_sequences(subseq, maxlen=self.max_context_size, padding='pre'))
        self.info.append((count, count + len_seq - 1))
        count += (len_seq - 1)
    self.padded = np.vstack(self.padded)
    self.padded = np.expand_dims(self.padded, axis=-1)

  def on_epoch_end(self, epoch, logs=None):
    scores = []
    predictions = self.model.predict(self.padded, verbose=0)

    for start, end in self.info:
      probs = [predictions[idx_seq, -1, idx_vocab]
               for idx_seq, idx_vocab in zip(range(start, end), self.target[start:end])]
      scores.append(np.exp(-np.sum(np.log(np.array(probs) + 1e-10)) / (end - start)))

    current_score = np.mean(scores)
    print(f'\nEpoch {epoch + 1}: Mean Perplexity: {current_score:.4f}')

    wandb.log({"val_perplexity": current_score, "epoch": epoch})

    if current_score < self.min_score:
      self.min_score = current_score
      self.model.save("best_model.keras")
      print("Saved new best model!")
      self.patience_counter = 0
    else:
      self.patience_counter += 1
      if self.patience_counter >= self.patience:
        self.model.stop_training = True
        print("Early stopping triggered.")


def main():
  wandb_key = os.environ.get("WANDB_API_KEY")

  env_model = os.environ.get("MODEL")

  if env_model:
    print(f"-> Sobreescribiendo modelo desde variable de entorno: {env_model}")
    CONFIG["model_type"] = env_model
  else:
    print(f"-> Usando modelo por defecto: {CONFIG['model_type']}")


  if wandb_key:
    wandb.login(key=wandb_key)
  else:
    # Fallo explícito si no hay key en el entorno
    raise RuntimeError("Error Crítico: WANDB_API_KEY no encontrada en variables de entorno. ")

  wandb.init(project=CONFIG["project_name"], config=CONFIG)
  config = wandb.config

  # 1. Cargar Datos
  print("Cargando datos pre-procesados...")
  tokenized_text, vocab_data = load_prepared_data()
  vocab_size = vocab_data['vocab_size']
  print(f"Vocabulario cargado: {vocab_size} caracteres.")

  # 2. Crear Dataset
  print("Generando secuencias...")
  X_train, y_train, val_sentences = create_dataset(tokenized_text, config.max_context_size, config.val_split)

  print(f"Construyendo modelo tipo: {config.model_type}")

  # 5. Construir el Modelo Seleccionado
  print(f"Construyendo arquitectura: {config.model_type}")

  if config.model_type.upper() == 'LSTM':
    model = build_lstm_model(vocab_size, config)
  elif config.model_type.upper() == 'SIMPLERNN':
    model = build_simple_model(vocab_size, config)
  elif config.model_type.upper() == 'GRU':
    model = build_gru_model(vocab_size, config)
  else:
    raise ValueError(f"Modelo '{config.model_type}' no reconocido. Usa 'GRU', 'LSTM' o 'SimpleRNN'.")


  model.summary()
  # 4. Entrenar
  ppl_callback = WandbPplCallback(val_sentences, config.max_context_size)

  model.fit(
    X_train, y_train,
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=[ppl_callback, WandbMetricsLogger(log_freq="batch")]
  )

  wandb.finish()


if __name__ == "__main__":
  main()