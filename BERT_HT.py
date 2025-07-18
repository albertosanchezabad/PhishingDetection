import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
from pathlib import Path
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Optimizaciones de CPU
os.environ["TF_USE_LEGACY_KERAS"] = "1"     # Para compatibilidad


# Configuración
DIRECTORIO_PROCESADOS = Path("dataset/procesados_bert")
DIRECTORIO_RESULTADOS = Path("resultados/bert_models_tuned")
DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Parámetros BERT
MAX_LENGTH = 256
BERT_MODEL_NAME = 'bert-base-uncased'

def cargar_datos_procesados(dataset_path):
    """Carga datos tokenizados para BERT"""
    try:
        train_data = np.load(dataset_path / 'train_data.npz')
        test_data = np.load(dataset_path / 'test_data.npz')
        return (
            {'input_ids': train_data['input_ids'], 'attention_mask': train_data['attention_mask']},
            train_data['labels'],
            {'input_ids': test_data['input_ids'], 'attention_mask': test_data['attention_mask']},
            test_data['labels']
        )
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return None


def hp_space(trial):
    """Define el espacio de búsqueda de hiperparámetros"""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 2e-5, 4e-5, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8]),
        'epochs': trial.suggest_int('epochs', 2, 3),
        'decay': trial.suggest_float('decay', 0.01, 0.05, log=True) 
    }

def construir_modelo_bert(config, num_labels=2):
    """Construye modelo BERT con hiperparámetros dinámicos"""
    model = TFBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, 
        num_labels=num_labels
    )
    
    optimizer = Adam(
        learning_rate=config['learning_rate'],
        decay=config['decay'],
        global_clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def objective(trial, dataset_dir):
    """Función objetivo para Optuna"""
    try:
        # Cargar datos
        datos = cargar_datos_procesados(dataset_dir)
        if datos is None:
            return float('nan')
        
        X_train, y_train, X_test, y_test = datos
        
        # Obtener hiperparámetros
        params = hp_space(trial)
        
        train_inputs = {**X_train, 'labels': y_train}
        test_inputs  = {**X_test,  'labels': y_test}

        train_dataset = (
            tf.data.Dataset
              .from_tensor_slices(train_inputs)
              .shuffle(1000)
              .batch(params['batch_size'])
              .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = (
            tf.data.Dataset
              .from_tensor_slices(test_inputs)
              .batch(params['batch_size'])
              .prefetch(tf.data.AUTOTUNE)
        )
        
        # Construir modelo
        model = construir_modelo_bert(params)
        
        # Crear directorio del modelo
        model_path = DIRECTORIO_RESULTADOS / f"bert_{dataset_dir.name}_trial_{trial.number}"
        model_path.mkdir(exist_ok=True)

        # Guardar configuración ANTES del entrenamiento
        model.config.save_pretrained(model_path)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=False),
            ModelCheckpoint(
                filepath=str(model_path / "weights.h5"),
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=True
            )
        ]
        
        print(f"\n🏋️ Trial {trial.number} - Entrenando BERT para {dataset_dir.name}")
        print(f"Parámetros: LR={params['learning_rate']:.2e}, BS={params['batch_size']}, Epochs={params['epochs']}, Decay={params['decay']:.3f}")

        # Entrenamiento
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Cargar los mejores pesos
        model.load_weights(str(model_path / "weights.h5"))
        
        # Evaluación con el mejor modelo
        y_pred = model.predict(test_dataset, verbose=1)
        y_pred_labels = np.argmax(y_pred.logits, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels, zero_division=0)
        recall = recall_score(y_test, y_pred_labels, zero_division=0)
        f1 = f1_score(y_test, y_pred_labels, zero_division=0)
        
        # Guardar métricas en el trial
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("best_epoch", np.argmax(history.history['val_accuracy']) + 1)
        trial.set_user_attr("history", history.history)  # Guardar historial completo

        print(f"✅ Trial {trial.number} completado - F1: {f1:.4f}")
        
        return f1
        
    except Exception as e:
        print(f"❌ Error en trial {trial.number}: {str(e)}")
        return float('nan')

def optimizar_hiperparametros(dataset_dir, n_trials=8):
    """Ejecuta la optimización para un dataset"""
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=2),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=1,  # Empezar pruning antes
            n_min_trials=3     # Mínimo 3 trials antes de pruning
        )
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_dir),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Guardar estudio completo
    joblib.dump(study, DIRECTORIO_RESULTADOS / f"study_{dataset_dir.name}.joblib")
    
    return study

def generar_graficos_mejor_trial(dataset_name, best_trial):
    """Genera gráficos de entrenamiento del mejor trial"""
    if 'history' not in best_trial.user_attrs:
        print(f"⚠️ No se encontró historial para {dataset_name}")
        return
    
    history = best_trial.user_attrs['history']
    
    # Crear array de épocas
    epochs = list(range(1, len(history['accuracy']) + 1))
    
    plt.figure(figsize=(14, 6))
    
    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b-o', linewidth=2, markersize=6, label='Entrenamiento')
    plt.plot(epochs, history['val_accuracy'], 'r-o', linewidth=2, markersize=6, label='Validación')
    plt.title(f'Precisión - {dataset_name} (Mejor Trial)', fontsize=14, fontweight='bold')
    plt.ylabel('Precisión', fontsize=12)
    plt.xlabel('Época', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 1.05)
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b-o', linewidth=2, markersize=6, label='Entrenamiento')
    plt.plot(epochs, history['val_loss'], 'r-o', linewidth=2, markersize=6, label='Validación')
    plt.title(f'Pérdida - {dataset_name} (Mejor Trial)', fontsize=14, fontweight='bold')
    plt.ylabel('Pérdida', fontsize=12)
    plt.xlabel('Época', fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(DIRECTORIO_RESULTADOS / f"graficos_mejor_{dataset_name}.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Gráfico del mejor trial guardado para {dataset_name}")


def generar_tabla_comparativa():
    """Genera tabla comparativa profesional con los mejores resultados"""
    metricas = []
    
    for study_file in DIRECTORIO_RESULTADOS.glob("study_*.joblib"):
        try:
            study = joblib.load(study_file)
            best_trial = study.best_trial
            
            if best_trial.value is not None:
                dataset_name = study_file.stem.replace("study_", "")
                
                metricas.append({
                    'Dataset': dataset_name,
                    'F1-Score': best_trial.value,
                    'Accuracy': best_trial.user_attrs.get('accuracy', 0),
                    'Precision': best_trial.user_attrs.get('precision', 0),
                    'Recall': best_trial.user_attrs.get('recall', 0),
                    'Best Epoch': best_trial.user_attrs.get('best_epoch', 0),
                    'Learning Rate': best_trial.params['learning_rate'],
                    'Batch Size': best_trial.params['batch_size'],
                    'Decay': best_trial.params['decay'],
                    'Epochs': best_trial.params['epochs']
                })
                
                # Generar gráfico del mejor trial
                generar_graficos_mejor_trial(dataset_name, best_trial)
                
        except Exception as e:
            print(f"⚠️ Error cargando {study_file}: {str(e)}")
    
    if not metricas:
        print("❌ No se encontraron métricas válidas")
        return
    
    df = pd.DataFrame(metricas)
    df = df.sort_values('F1-Score', ascending=False)
    
    # Guardar CSV con hiperparámetros incluidos
    csv_path = DIRECTORIO_RESULTADOS / "metricas_finales.csv"
    df.to_csv(csv_path, index=False)
    
    # Generar tabla visual SIN colores verdes
    plt.figure(figsize=(18, max(6, len(df)*0.6)))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Formatear valores
    df_display = df.copy()
    for col in ['F1-Score', 'Accuracy', 'Precision', 'Recall']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    df_display['Learning Rate'] = df_display['Learning Rate'].apply(lambda x: f"{x:.2e}")
    df_display['Decay'] = df_display['Decay'].apply(lambda x: f"{x:.3f}")
    
    # Crear tabla con anchos de columna personalizados
    col_widths = [0.2, 0.1, 0.1, 0.1, 0.1, 0.08, 0.12, 0.08, 0.08, 0.06]  # Dataset más ancho
    
    table = plt.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f3f3f3']*len(df.columns),
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Estilizar SOLO cabeceras (sin colores verdes en celdas)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Colorear filas alternadas (sin resaltar mejores valores)
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Resultados Optimización Hiperparámetros BERT', fontsize=16, pad=20, fontweight='bold')
    plt.savefig(DIRECTORIO_RESULTADOS / "tabla_comparativa_final.png", 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n📊 Tabla comparativa guardada en: {DIRECTORIO_RESULTADOS / 'tabla_comparativa_final.png'}")
    print(f"📈 CSV con métricas e hiperparámetros guardado en: {csv_path}")


# Procesar todos los datasets
for dataset_dir in DIRECTORIO_PROCESADOS.glob("G*"):
    nombre_dataset = dataset_dir.name

    if nombre_dataset in ["G1_grupo_combinado_1", "G2_grupo_combinado_2"]:
        print(f"\n{'='*40}\nOptimizando: {dataset_dir.name}\n{'='*40}")
        try:
            study = optimizar_hiperparametros(dataset_dir)
            if study.best_trial.value is not None:
                print(f"✅ Mejor F1-Score para {dataset_dir.name}: {study.best_value:.4f}")
                print(f"📋 Mejores parámetros: {study.best_params}")
            else:
                print(f"⚠️ No se encontraron trials válidos para {dataset_dir.name}")
        except Exception as e:
            print(f"❌ Error optimizando {dataset_dir.name}: {str(e)}")

# Generar reporte final
try:
    generar_tabla_comparativa()
    print("\n✅ Optimización completada!")
except Exception as e:
    print(f"❌ Error generando reporte final: {str(e)}")
