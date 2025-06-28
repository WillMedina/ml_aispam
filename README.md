# 📧 Clasificador de Correos SPAM/HAM

Un sistema completo de machine learning para la clasificación automática de correos electrónicos como SPAM o HAM (correo legítimo), con API REST y despliegue en la nube.

## 🏗️ Arquitectura del Proyecto

```
├── app/                    # API REST y aplicación web
│   ├── main.py            # FastAPI application
│   ├── models/            # Modelos entrenados (.joblib, .h5)
│   ├── requirements.txt   # Dependencias de la API
│   └── Dockerfile         # Configuración del contenedor
├── data/                  # Datasets de entrenamiento y prueba
│   ├── train/            # Datos de entrenamiento
│   └── test/             # Datos de prueba
├── preprocessing.py       # Script de preprocesamiento de datos
├── train.py              # Script de entrenamiento del modelo
├── requirements.txt      # Dependencias del proyecto
└── README.md            # Este archivo
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.9+
- Docker (para despliegue)
- Cuenta en DigitalOcean u otro proveedor de nube (opcional)

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone <tu-repositorio>
cd ml_aispam
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 📊 Entrenamiento del Modelo

### Paso 1: Preprocesamiento de Datos

Ejecuta el script de preprocesamiento para limpiar y preparar los datos:

```bash
python preprocessing.py
```

Este script:
- Limpia y normaliza el texto de los correos
- Elimina caracteres especiales y stopwords
- Tokeniza y vectoriza el contenido
- Divide los datos en conjuntos de entrenamiento y validación

### Paso 2: Entrenamiento

Entrena el modelo de clasificación:

```bash
python train.py
```

El entrenamiento generará:
- `model.joblib`: Modelo de machine learning serializado
- `vectorizer.joblib`: Vectorizador de texto entrenado
- `model.h5`: Modelo de deep learning (si aplica)

### Entrenamiento en Google Colab

Para entrenar en Google Colab:

1. Sube los archivos del proyecto a Google Drive
2. Abre un nuevo notebook en Colab
3. Monta Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Navega al directorio del proyecto:
```python
%cd /content/drive/MyDrive/spam-classifier
```

5. Instala dependencias y ejecuta:
```python
!pip install -r requirements.txt
!python preprocessing.py
!python train.py
```

6. Descarga los modelos entrenados a tu máquina local

## 🐳 Despliegue con Docker

### Construcción Local

1. **Construir la imagen Docker**
```bash
cd app/
docker build -t ml_aispam .
```

2. **Ejecutar el contenedor localmente**
```bash
docker run -p 8000:8000 ml_aispam
```

La API estará disponible en `http://localhost:8000`

### Despliegue en DigitalOcean

#### Opción 1: Docker en Droplet

1. **Crear un Droplet en DigitalOcean**
   - Selecciona Ubuntu 20.04+
   - Instala Docker en el servidor

2. **Subir la imagen**
```bash
# Etiquetar imagen
docker tag ml_aispam tu_usuario_de_docker/ml_aispam

# Subir a registry (Docker Hub, DigitalOcean Container Registry)
docker push tu_usuario_de_docker/ml_aispam
```

3. **Desplegar en el servidor**
```bash
ssh root@TU_IP_DE_DROPLET_O_SERVER
docker pull tu_usuario_de_docker/ml_aispam
docker run -d -p 80:8000 --name ml_aispam tu_usuario_de_docker/ml_aispam
```

#### Opción 2: DigitalOcean App Platform

1. Conecta tu repositorio de GitHub/GitLab
2. Configura el build con el Dockerfile en `/app`
3. Establece el puerto 8000
4. Despliega automáticamente

## 🔌 API Endpoints

### Predicción Individual
```http
POST /predict
Content-Type: application/json

{
    "text": "Congratulations! You've won $1000! Click here to claim your prize!"
}
```

**Respuesta:**
```json
{
    "prediction": "spam",
    "confidence": 0.95,
    "timestamp": "2024-01-15T10:30:00Z"
}
```


### Estado de Salud
```http
GET /health
```

## 📱 Integración con Aplicaciones

### Aplicación Web (Frontend)

```javascript
// Ejemplo con JavaScript
async function classifyEmail(emailText) {
    const response = await fetch('https://TU_IP_DE_DROPLET_O_SERVER/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText })
    });
    
    const result = await response.json();
    return result;
}
```

### Aplicación Móvil (React Native)

```javascript
// Ejemplo para React Native
import axios from 'axios';

export const SpamClassifier = {
    async classify(emailText) {
        try {
            const response = await axios.post('https://TU_IP_DE_DROPLET_O_SERVER/predict', {
                text: emailText
            });
            return response.data;
        } catch (error) {
            console.error('Error clasificando email:', error);
            throw error;
        }
    }
};
```

### Aplicación Android (Kotlin)

```kotlin
// Ejemplo para Android
class SpamClassifierService {
    private val client = OkHttpClient()
    private val gson = Gson()
    
    suspend fun classifyEmail(emailText: String): ClassificationResult {
        val requestBody = EmailRequest(emailText)
        val json = gson.toJson(requestBody)
        
        val request = Request.Builder()
            .url("https://TU_IP_DE_DROPLET_O_SERVER/predict")
            .post(json.toRequestBody("application/json".toMediaType()))
            .build()
            
        val response = client.newCall(request).execute()
        return gson.fromJson(response.body?.string(), ClassificationResult::class.java)
    }
}
```

## 📊 Monitoreo y Métricas

### Métricas Disponibles
```http
GET /metrics
```

Retorna métricas como:
- Número total de predicciones
- Tiempo promedio de respuesta
- Distribución de clasificaciones
- Uso de memoria y CPU

### Logging

Los logs están disponibles en:
- Contenedor: `docker logs spam-api`
- Archivo: `/var/log/ML_AISPAM_LOG.log`

## 🔧 Configuración

### Variables de Entorno

```bash
# .env file
MODEL_PATH=/app/models/model.joblib
VECTORIZER_PATH=/app/models/vectorizer.joblib
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=10000
RATE_LIMIT=100  # requests per minute
```

### Configuración de Producción

Para producción, considera:
- Usar un servidor ASGI como Gunicorn
- Implementar balanceador de carga
- Configurar HTTPS con certificados SSL
- Establecer límites de rate limiting
- Configurar monitoreo con Prometheus/Grafana

## 🧪 Testing

```bash
# Ejecutar tests unitarios
python -m pytest tests/

# Test de la API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a test email"}'
```

## 📈 Mejoras Futuras

- [ ] Implementar reentrenamiento automático
- [ ] Agregar soporte para múltiples idiomas
- [ ] Implementar explicabilidad del modelo (LIME/SHAP)
- [ ] Añadir interfaz web para administración
- [ ] Implementar A/B testing para modelos
- [ ] Agregar detección de phishing avanzada

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request