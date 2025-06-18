import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tf_keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tf_keras.models import Sequential
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
from tf_keras.metrics import Precision, Recall, BinaryAccuracy
from tf_keras.layers import InputLayer, Dense, Dropout, BatchNormalization


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
ASSETS_DIR = 'modelo_assets'
LR = 1e-4
EPOCHS = 20
BATCH_SZ = 32


# ============================================================================
# FUNCIONES AUXILIARES: MÉTRICAS Y GRÁFICOS
# ============================================================================

class F1Score(keras.metrics.Metric):
    """Métrica F1-Score personalizada."""
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision(name='p_f1')
        self.recall = Recall(name='r_f1')
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def reset_state(self):
        self.precision.reset_state(); self.recall.reset_state()
    def result(self):
        p = self.precision.result(); r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))

def generar_reporte_grafico(history, y_test, y_pred_proba, output_dir):
    """Genera un reporte visual completo del entrenamiento y evaluación."""
    y_pred = (y_pred_proba > 0.5).astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Reporte de Rendimiento del Modelo (TF-IDF + DNN)', fontsize=22, weight='bold')

    ax = axes[0, 0]
    ax.plot(history.history.get('accuracy'), marker='o', label='Train Acc'); ax.plot(history.history.get('val_accuracy'), marker='x', linestyle='--', label='Val Acc'); ax.plot(history.history.get('loss'), marker='o', label='Train Loss'); ax.plot(history.history.get('val_loss'), marker='x', linestyle='--', label='Val Loss'); ax.set_title('Accuracy & Loss', fontsize=16); ax.legend(); ax.grid(True)
    ax = axes[0, 1]
    ax.plot(history.history.get('precision'), marker='o', label='Train Prec'); ax.plot(history.history.get('val_precision'), marker='x', linestyle='--', label='Val Prec'); ax.plot(history.history.get('recall'), marker='o', label='Train Rec'); ax.plot(history.history.get('val_recall'), marker='x', linestyle='--', label='Val Rec'); ax.plot(history.history.get('f1_score'), marker='o', label='Train F1'); ax.plot(history.history.get('val_f1_score'), marker='x', linestyle='--', label='Val F1'); ax.set_title('Métricas de Clasificación', fontsize=16); ax.legend(); ax.grid(True)
    ax = axes[1, 0]; cm = confusion_matrix(y_test, y_pred); cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]; labels = [f"{v1}\n({v2:.2%})" for v1, v2 in zip(cm.flatten(), cm_percent.flatten())]; labels = np.asarray(labels).reshape(2, 2); sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax, cbar=False); ax.set_title('Matriz de Confusión', fontsize=16); ax.set_xlabel('Predicción'); ax.set_ylabel('Real'); ax.set_xticklabels(['HAM', 'SPAM']); ax.set_yticklabels(['HAM', 'SPAM'], rotation=0)
    ax = axes[1, 1]; fpr, tpr, _ = roc_curve(y_test, y_pred_proba); roc_auc = auc(fpr, tpr); ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})'); ax.plot([0, 1], [0, 1], color='navy', linestyle='--'); ax.set_title('Curva ROC', fontsize=16); ax.legend(loc="lower right"); ax.grid(True)
    
    report_path = os.path.join(output_dir, 'reporte_rendimiento.png'); plt.savefig(report_path, dpi=150); plt.close()
    print(f"Reporte gráfico guardado en: {report_path}")

# ============================================================================
# DEFINICIÓN DEL MODELO
# ============================================================================

def crear_modelo_dnn(input_dim):
    """Construye el modelo de Red Neuronal Densa (DNN)."""
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(192, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model


# ============================================================================
# FLUJO PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

if __name__ == '__main__':
    print("[+] Iniciando entrenamiento con Red Neuronal Densa...")

    # Carga de datos
    X_train = np.load(os.path.join(ASSETS_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(ASSETS_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(ASSETS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(ASSETS_DIR, 'y_test.npy'))

    # Creación y compilación
    model = crear_modelo_dnn(input_dim=X_train.shape[1])
    METRICAS = [BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), F1Score()]
    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=METRICAS)
    model.summary()

    # Entrenamiento
    print("\n[+] Comenzando entrenamiento...")
    early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SZ,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluación y guardado
    print("\n[+] Evaluación final y guardado de artefactos...")
    y_pred_proba = model.predict(X_test).flatten()
    generar_reporte_grafico(history, y_test, y_pred_proba, ASSETS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(ASSETS_DIR, f"modelo_dnn_spam_{timestamp}.h5")
    model.save(model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

    print("\n[SUCCESS] Entrenamiento completado.")