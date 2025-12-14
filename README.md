# Procesamiento de Lenguaje Natural (NLP) - Trabajos Prácticos

Este repositorio contiene una colección de desafíos y proyectos prácticos enfocados en técnicas de **Procesamiento de Lenguaje Natural (NLP)**. A través de estos trabajos, se explora la evolución de los modelos de lenguaje, desde algoritmos probabilísticos clásicos hasta arquitecturas de redes neuronales recurrentes y modelos Seq2Seq.

## 📋 Contenido

El repositorio está organizado en cuatro desafíos incrementales:

### 1. Clasificación de Texto con Naïve Bayes
**[Ver Notebook](https://github.com/joseluisdiaz/procesamiento_lenguaje_natural-tp/blob/main/Desafio_1.ipynb)**

Implementación de un modelo de clasificación de texto utilizando el algoritmo probabilístico **Naïve Bayes**.
* **Objetivo:** Clasificación de documentos/texto en categorías predefinidas.
* **Técnicas:** Preprocesamiento de texto (tokenización, stop words), vectorización (Bag of Words / TF-IDF) y modelado con Naïve Bayes.

### 2. Representación Vectorial con Word2Vec
**[Ver Notebook](https://github.com/joseluisdiaz/procesamiento_lenguaje_natural-tp/blob/main/Desafio_2.ipynb)**

Exploración de técnicas de **Word Embeddings** para capturar relaciones semánticas entre palabras.
* **Objetivo:** Crear y visualizar representaciones vectoriales de palabras.
* **Técnicas:** Entrenamiento de modelos **Word2Vec** (CBOW/Skip-gram), visualización de espacios vectoriales y análisis de similitud semántica.

### 3. Modelos de Lenguaje con RNNs (SimpleRNN, GRU y LSTM)
**[Ver Carpeta](https://github.com/joseluisdiaz/procesamiento_lenguaje_natural-tp/tree/main/Desafio_3)**

Comparativa y experimentación con diferentes arquitecturas de Redes Neuronales Recurrentes para la generación de texto o predicción de secuencias.
* **Objetivo:** Analizar el rendimiento de distintas celdas recurrentes y estructurar un flujo de trabajo profesional.
* **Arquitecturas:** SimpleRNN, GRU (Gated Recurrent Unit) y LSTM (Long Short-Term Memory).
* **Highlight:** Se implementó una separación explícita entre el **entrenamiento del modelo** y la **inferencia interactiva** para mejorar la modularidad del código.

### 4. Traductor Automático (Seq2Seq)
**[Ver Notebook](https://github.com/joseluisdiaz/procesamiento_lenguaje_natural-tp/blob/main/Desafio_4.ipynb)**

Desarrollo de un sistema de traducción automática utilizando una arquitectura **Sequence-to-Sequence (Seq2Seq)**.
* **Objetivo:** Traducir oraciones de un idioma origen a un idioma destino (Inglés - Español).
* **Técnicas:** Arquitectura Encoder-Decoder, manejo de secuencias de longitud variable, capas de Embedding y capas recurrentes profundas.