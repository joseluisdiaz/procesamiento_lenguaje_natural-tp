import unittest
from preparar_datos import process_corpus


class TestDataPrep(unittest.TestCase):

  def setUp(self):
    # Texto de prueba con caracteres especiales y espacios
    self.sample_text = "hola mundo! 123"
    self.tokens, self.vocab_data = process_corpus(self.sample_text)

  def test_granularity_is_char(self):
    """
    VALIDACIÓN CRÍTICA:
    Verifica que el dataset esté tokenizado por CARACTERES y no por palabras.
    """
    char2idx = self.vocab_data['char2idx']

    for char in char2idx.keys():
      # Si esto falla, el modelo está tokenizando palabras o subwords
      self.assertEqual(len(char), 1,
                       f"Error: El token '{char}' tiene longitud > 1. Tokenización por caracteres fallida.")

  def test_vocab_consistency(self):
    """Verifica que idx2char y char2idx sean espejos exactos."""
    char2idx = self.vocab_data['char2idx']
    idx2char = self.vocab_data['idx2char']

    self.assertEqual(len(char2idx), len(idx2char))

    for char, idx in char2idx.items():
      self.assertEqual(idx2char[idx], char)

  def test_tokenization_integrity(self):
    """Verifica que el texto se pueda reconstruir (Round-trip)."""
    reconstructed = "".join([self.vocab_data['idx2char'][idx] for idx in self.tokens])
    self.assertEqual(reconstructed, self.sample_text)

  def test_vocab_size_is_reasonable(self):
    """
    Un modelo de caracteres debería tener un vocabulario pequeño (< 300 aprox).
    Un modelo de palabras tendría miles.
    """
    vocab_size = self.vocab_data['vocab_size']
    # El texto de prueba tiene 11 caracteres únicos: 'h','o','l','a',' ','m','u','n','d','!','1','2','3'
    unique_chars = len(set(self.sample_text))
    self.assertEqual(vocab_size, unique_chars)

    # En un escenario real, si el vocabulario es > 1000, algo anda mal para char-level
    self.assertLess(vocab_size, 1000, "El vocabulario es sospechosamente grande para ser por caracteres.")


if __name__ == '__main__':
  unittest.main()