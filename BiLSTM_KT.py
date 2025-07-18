import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras_tuner as kt  # Importación nueva para hiperoptimización
from tensorflow.keras.optimizers import Adam, RMSprop
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar GPU
tf.config.optimizer.set_jit(True)  # Activa XLA para todas las operaciones


# Configuración de directorios
DIRECTORIO_PROCESADOS = Path("dataset/procesados_lstm")
DIRECTORIO_RESULTADOS = Path("resultados/bilstm_models_kt")
DIRECTORIO_TUNER = Path("trials/bilstm")
DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
DIRECTORIO_TUNER.mkdir(parents=True, exist_ok=True)

# Parámetros globales
USAR_TUNER = True  # Cambiar a False para usar modelo base sin optimización
PARAMS_BASE = {
    'embedding_dim': 128,
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 128,
    'epochs': 10,
    'patience': 3
}

def construir_hipermodelo(hp, vocab_size, max_len):
    """Construye modelo BiLSTM compatible con cuDNN para máximo rendimiento"""
    model = Sequential()
    
    # 1. Embedding layer
    embedding_dim = hp.Choice('embedding_dim', [64])
    model.add(Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        input_length=max_len
    ))
    
    # 2. Primera capa BiLSTM - COMPATIBLE CON cuDNN
    lstm_units1 = hp.Choice('lstm_units1', [64, 128])
    model.add(Bidirectional(LSTM(
        lstm_units1,
        return_sequences=True
    )))
    
    # 3. Dropout manual después de LSTM
    dropout1 = hp.Choice('dropout1', [0.3, 0.4])
    model.add(Dropout(dropout1))
    
    # 4. Segunda capa BiLSTM - COMPATIBLE CON cuDNN
    lstm_units2 = hp.Choice('lstm_units2', [32, 64])
    model.add(Bidirectional(LSTM(
        lstm_units2
    )))
    
    # 5. Dropout manual
    model.add(Dropout(dropout1))
    
    # 6. Capa densa intermedia
    dense_units = hp.Choice('dense_units', [32])
    model.add(Dense(dense_units, activation='relu'))
    
    # 7. Dropout final
    dense_dropout = hp.Choice('dense_dropout', [0.3])
    model.add(Dropout(dense_dropout))
    
    # 8. Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    # 9. Optimización
    optimizer_name = hp.Choice('optimizer', ['adam'])  # Solo Adam
    learning_rate = hp.Choice('lr', [1e-3])
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model



def tuner_hyperparameter(X_train, y_train, X_test, y_test, vocab_size, max_len):
    """Ejecuta la optimización de hiperparámetros con Keras Tuner"""
    tuner = kt.Hyperband(
        lambda hp: construir_hipermodelo(hp, vocab_size, max_len),
        objective='val_accuracy',
        max_epochs=10,        # Aumentar épocas
        factor=3,             # Factor conservador
        hyperband_iterations=2,  # Más iteraciones
        directory=DIRECTORIO_TUNER,
        project_name='bilstm',
        overwrite=True
    )
    
    # Configurar parámetros fijos para todos los trials
    tuner.search_space_summary()
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=PARAMS_BASE['epochs'],
        batch_size=PARAMS_BASE['batch_size'],
        callbacks=[
            EarlyStopping(patience=PARAMS_BASE['patience'], restore_best_weights=True)
        ],
        verbose=1
    )
    
    # Obtener mejor modelo
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_model, best_hps.values

def construir_modelo_base(vocab_size, max_len):
    """Construye modelo LSTM con parámetros base (sin optimizar)"""
    model = Sequential()
    model.add(Embedding(
        input_dim=vocab_size + 1,
        output_dim=PARAMS_BASE['embedding_dim'],
        input_length=max_len
    ))
    model.add(Dropout(PARAMS_BASE['dropout_rate']))
    model.add(LSTM(PARAMS_BASE['lstm_units']))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def entrenar_evaluar_modelo(dataset_dir):
    """Pipeline principal de entrenamiento y evaluación"""
    datos = cargar_datos_procesados(dataset_dir)
    if datos is None:
        return None
    
    X_train, X_test, y_train, y_test, max_len, vocab_size = datos
    
    if USAR_TUNER:
        print("\n🔍 Iniciando optimización de hiperparámetros...")
        model, best_params = tuner_hyperparameter(
            X_train, y_train, X_test, y_test, vocab_size, max_len
        )
        print("✅ Mejores hiperparámetros encontrados:", best_params)
    else:
        print("\n🏗️ Construyendo modelo base...")
        model = construir_modelo_base(vocab_size, max_len)
    
    # Configurar callbacks comunes
    nombre_modelo = f"lstm_{'tuned' if USAR_TUNER else 'base'}_{dataset_dir.name}"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PARAMS_BASE['patience'], restore_best_weights=True),
        ModelCheckpoint(
            filepath=DIRECTORIO_RESULTADOS / f"{nombre_modelo}_best.h5",
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Entrenamiento final
    print(f"\n🏋️ Entrenamiento final: {nombre_modelo}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=PARAMS_BASE['epochs'],
        batch_size=PARAMS_BASE['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluación
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    return {
        'dataset': dataset_dir.name,
        'modelo': 'LSTM_Tuned' if USAR_TUNER else 'LSTM_Base',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'best_epoch': len(history.history['loss']),
        'val_loss': min(history.history['val_loss']),
        'hiperparams': str(best_params) if USAR_TUNER else str(PARAMS_BASE)
    }, history

def cargar_datos_procesados(dataset_path):
    """Carga los datos preprocesados para LSTM"""
    try:
        datos = joblib.load(dataset_path / "datos.joblib")
        return (
            datos['X_train'], 
            datos['X_test'],
            datos['y_train'],
            datos['y_test'],
            datos['max_len'],
            datos['vocab_size']
        )
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return None

def generar_graficos_entrenamiento(history, dataset_nombre):
    """Genera y guarda gráficos de entrenamiento individuales"""
    plt.figure(figsize=(12, 6))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title(f'Precisión - {dataset_nombre}')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title(f'Pérdida - {dataset_nombre}')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(DIRECTORIO_RESULTADOS / f"graficos_{dataset_nombre}.png")
    plt.close()

def generar_comparativa_general():
    """Genera gráfico comparativo y tabla de resultados desde el CSV"""
    csv_path = DIRECTORIO_RESULTADOS / "metricas_lstm.csv"
    
    if not csv_path.exists():
        print("⚠️ No se encontraron resultados para generar la comparativa")
        return
    
    df = pd.read_csv(csv_path)
    
    # 1. Gráfico de barras comparativo
    num_datasets = len(df)
    columnas = 3
    filas = (num_datasets + columnas - 1) // columnas
    
    plt.figure(figsize=(18, 6 * filas))
    for i, (_, row) in enumerate(df.iterrows(), 1):
        plt.subplot(filas, columnas, i)
        plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], 
                [row['accuracy'], row['precision'], row['recall'], row['f1']],
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f"{row['dataset']}\n(Epochs: {row['best_epoch']})")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(DIRECTORIO_RESULTADOS / "comparativa_general.png")
    plt.close()
    
    # 2. Tabla comparativa con valores numéricos
    plt.figure(figsize=(16, max(6, len(df)*0.5)))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Preparar datos para la tabla
    df_tabla = df[['dataset', 'accuracy', 'precision', 'recall', 'f1', 'best_epoch', 'val_loss']].copy()
    df_tabla['val_loss'] = df_tabla['val_loss'].round(4)
    df_tabla[['accuracy', 'precision', 'recall', 'f1']] = df_tabla[['accuracy', 'precision', 'recall', 'f1']].round(4)
    
    # Crear tabla
    tabla = plt.table(
        cellText=df_tabla.values,
        colLabels=df_tabla.columns,
        colColours=['#f0f0f0']*len(df_tabla.columns),
        cellLoc='center',
        loc='center'
    )
    
    # Formatear tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)  # Ajustar tamaño de celdas
    
    # Añadir título
    plt.title("Comparativa de Métricas LSTM", y=0.95, fontsize=14)
    
    # Guardar tabla
    plt.tight_layout()
    plt.savefig(DIRECTORIO_RESULTADOS / "comparativa_tabla.png", bbox_inches='tight')
    plt.close()
    
    print("\n📊 Tabla comparativa generada: comparativa_tabla.png")
    print("📈 Gráfico comparativo actualizado: comparativa_general.png")


def guardar_resultados(resultados_actuales):
    """Guarda resultados en CSV combinando con existentes"""
    csv_path = DIRECTORIO_RESULTADOS / "metricas_lstm.csv"
    
    # Cargar resultados existentes si hay
    if csv_path.exists():
        df_existente = pd.read_csv(csv_path)
        df_nuevo = pd.DataFrame(resultados_actuales)
        df_combinado = pd.concat([df_existente, df_nuevo])
        
        # Eliminar duplicados manteniendo la última versión
        df_combinado = df_combinado.drop_duplicates(subset='dataset', keep='last')
    else:
        df_combinado = pd.DataFrame(resultados_actuales)
    
    df_combinado.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados guardados/actualizados en: {csv_path}")



# Procesar todos los datasets
resultados_nuevos = []
csv_path = DIRECTORIO_RESULTADOS / "metricas_lstm.csv"

# Cargar resultados existentes si existen
if csv_path.exists():
    df_existente = pd.read_csv(csv_path)
    print("\n📂 Resultados existentes cargados")
else:
    df_existente = pd.DataFrame()

for dataset_dir in DIRECTORIO_PROCESADOS.glob("G*"):
    nombre_dataset = dataset_dir.name

    if nombre_dataset == "G1_grupo_combinado_1" or nombre_dataset == "G2_grupo_combinado_2":
    
        # Saltar si ya está procesado
        if not df_existente.empty and nombre_dataset in df_existente['dataset'].values:
            print(f"\n⏩ Saltando {nombre_dataset} (ya procesado)")
            continue
        
        print(f"\n{'='*40}\nProcesando: {nombre_dataset}\n{'='*40}")
        
        # Entrenar y evaluar modelo
        try:
            metricas, history = entrenar_evaluar_modelo(dataset_dir)
            if metricas:
                resultados_nuevos.append(metricas)
                generar_graficos_entrenamiento(history, nombre_dataset)
        except Exception as e:
            print(f"❌ Error crítico procesando {nombre_dataset}: {str(e)}")

# Actualizar resultados y generar gráficos
if resultados_nuevos:
    guardar_resultados(resultados_nuevos)

# Generar comparativa general siempre (incluso si no hay nuevos datos)
generar_comparativa_general()

print("\n✅ Proceso completado exitosamente!")

