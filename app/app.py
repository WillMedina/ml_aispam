import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import numpy as np
import tf_keras as keras
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger('202506180511_MLAISPAM')

MODEL_DIR = 'modelo_assets'
PORT = int(os.getenv('PORT', 5000))


# ============================================================================
# COMPONENTES DE PREPROCESAMIENTO Y CARGA DE RECURSOS
# ============================================================================
try:
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)
    stemmer = SnowballStemmer('spanish')
    stop_words = set(stopwords.words('spanish'))


def procesar_texto(texto: str) -> str:
    """Función de preprocesamiento de texto. Idéntica a la de entrenamiento."""
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    tokens = word_tokenize(texto, language='spanish')
    tokens_procesados = [stemmer.stem(p) for p in tokens if p not in stop_words and len(p) > 1]
    return " ".join(tokens_procesados)


def load_resources():
    """Carga el modelo Keras (.h5) y el vectorizador Tfidf (.joblib)."""
    try:
        # Cargar modelo
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        if not model_files: raise FileNotFoundError("No se encontró archivo de modelo .h5")
        model_path = os.path.join(MODEL_DIR, sorted(model_files, reverse=True)[0])
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Modelo cargado desde: {model_path}")

        # Cargar vectorizador
        vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer_optimizado.joblib')
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Vectorizer cargado desde: {vectorizer_path}")
        
        return model, vectorizer, sorted(model_files, reverse=True)[0]

    except Exception as e:
        logger.error(f"Error fatal al cargar los recursos: {e}", exc_info=True)
        return None, None, None


model, vectorizer, MODEL_NAME = load_resources()


# ============================================================================
# APLICACIÓN FLASK Y ENDPOINTS
# ============================================================================
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    if not all([model, vectorizer]):
        return jsonify({'status': 'error', 'message': 'Servicio no disponible'}), 503
    
    try:
        data = request.get_json(force=True)
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'status': 'error', 'message': 'El campo "message" es requerido.'}), 400
        
        processed_text = procesar_texto(message)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        
        prediction = model.predict(vectorized_text, verbose=0)
        prob_spam = float(prediction[0][0])
        label = 'SPAM' if prob_spam > 0.5 else 'HAM'
        
        return jsonify({'status': 'success', 'prediction': label, 'probability_spam': round(prob_spam, 4)})

    except Exception as e:
        logger.error(f'Error en /predict: {e}', exc_info=True)
        return jsonify({'status': 'error', 'message': 'Error interno del servidor.'}), 500


@app.route('/health')
def health_check():
    """Endpoint para verificar la salud del servicio."""
    healthy = bool(model and vectorizer)
    status_code = 200 if healthy else 503
    return jsonify({
        'status': 'healthy' if healthy else 'unhealthy',
        'model_loaded': MODEL_NAME if healthy else 'NOT_LOADED',
        'deployment_version': '202506180511_MLAISPAM'
    }), status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)