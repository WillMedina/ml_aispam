import os
import re
import pandas as pd
import numpy as np
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuración ---
PATH_TRAIN_CSV = 'train.csv'
PATH_TEST_CSV = 'test.csv'
OUTPUT_DIR = 'modelo_assets' # Nuevo directorio para los nuevos artefactos
MAX_FEATURES = 7500  # Aumentamos ligeramente las características

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Inicialización de Componentes de PLN ---
try:
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))


def procesar_texto(texto: str) -> str:
    # (Esta función no necesita cambios)
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    tokens = word_tokenize(texto, language='spanish')
    tokens_procesados = [
        stemmer.stem(palabra) for palabra in tokens
        if palabra not in stop_words and len(palabra) > 1
    ]
    return " ".join(tokens_procesados)


# --- Flujo Principal de Preprocesamiento ---
if __name__ == '__main__':
    print("[+] Iniciando preprocesamiento OPTIMIZADO...")

    # Carga y procesamiento de texto (sin cambios)
    df_train = pd.read_csv(PATH_TRAIN_CSV)
    df_test = pd.read_csv(PATH_TEST_CSV)
    df_train['texto_procesado'] = df_train['mensaje'].apply(procesar_texto)
    df_test['texto_procesado'] = df_test['mensaje'].apply(procesar_texto)
    df_train['etiqueta'] = df_train['tipo'].apply(lambda x: 1 if x == 'spam' else 0)
    df_test['etiqueta'] = df_test['tipo'].apply(lambda x: 1 if x == 'spam' else 0)

    # --- Vectorización TF-IDF Optimizada ---
    print(f"[+] Creando y ajustando TfidfVectorizer con N-Gramos...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 4),  # <-- INCLUIR TRIGRAMAS
        min_df=5,            # <-- IGNORAR TÉRMINOS MUY RAROS
        max_df=0.95          # <-- IGNORAR TÉRMINOS MUY COMUNES
    )
    
    X_train = vectorizer.fit_transform(df_train['texto_procesado']).toarray()
    y_train = df_train['etiqueta'].values
    
    X_test = vectorizer.transform(df_test['texto_procesado']).toarray()
    y_test = df_test['etiqueta'].values

    print(f"    - Dimensiones de la nueva matriz de entrenamiento: {X_train.shape}")

    # Guardado de artefactos
    print("[+] Guardando artefactos optimizados...")
    vectorizer_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer_optimizado.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"    - Vectorizador guardado en: {vectorizer_path}")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    print("    - Arrays de datos (NumPy) guardados.")

    print("\n[SUCCESS] Preprocesamiento optimizado completado.")