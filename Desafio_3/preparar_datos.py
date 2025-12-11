import urllib.request
import bs4 as bs
import pickle
import os
import numpy as np


def download_book(url):
  """Descarga y limpia el HTML del libro."""
  try:
    raw_html = urllib.request.urlopen(url).read()
  except Exception as e:
    raise Exception(f"Error descargando: {e}")

  article_html = bs.BeautifulSoup(raw_html, 'lxml')
  article_paragraphs = article_html.find_all('p')

  article_text = ''
  for para in article_paragraphs:
    article_text += para.text + ' '

  return article_text.lower()


def process_corpus(text):
  """
  Lógica pura: Recibe texto, devuelve tokens y vocabulario.
  Esta es la función que vamos a testear.
  """
  # Crear Vocabulario
  chars_vocab = sorted(list(set(text)))
  vocab_size = len(chars_vocab)

  char2idx = {char: idx for idx, char in enumerate(chars_vocab)}
  idx2char = {idx: char for idx, char in enumerate(chars_vocab)}

  # Tokenizar
  tokenized_text = [char2idx[ch] for ch in text]

  vocab_data = {
    'char2idx': char2idx,
    'idx2char': idx2char,
    'vocab_size': vocab_size
  }

  return np.array(tokenized_text), vocab_data


def main():
  print("--- Iniciando preparación de datos ---")

  # 1. Descargar
  print("1. Descargando corpus...")
  url = 'https://www.textos.info/julio-verne/la-vuelta-al-mundo-en-80-dias/ebook'
  article_text = download_book(url)
  print(f"   Corpus descargado. Longitud: {len(article_text)} caracteres.")

  # 2. Procesar
  print("2. Procesando y tokenizando...")
  tokenized_text, vocab_data = process_corpus(article_text)
  print(f"   Tamaño del vocabulario: {vocab_data['vocab_size']}")

  # 3. Guardar
  print("3. Guardando archivos...")
  with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab_data, f)

  np.save('corpus_tokenizado.npy', tokenized_text)

  print("--- ¡Listo! Archivos generados. ---")


if __name__ == "__main__":
  main()