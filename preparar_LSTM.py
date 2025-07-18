import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Configuración
DIRECTORIO_DATASETS = Path("dataset/solo_ingles")
DIRECTORIO_RESULTADOS = Path("dataset/procesados_lstm")
DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Parámetros ajustables
VOCAB_SIZE = 20000    # Tamaño máximo del vocabulario
OOV_TOKEN = "<OOV>"   # Token para palabras fuera del vocabulario
TEST_SIZE = 0.2       # Proporción para test
RANDOM_STATE = 42     # Semilla para reproducibilidad
PERCENTIL_PADDING = 95# Percentil para calcular longitud máxima

def procesar_dataset(nombre_archivo, grupo):
    """Proceso completo de preprocesamiento para LSTM"""
    print(f"\n{'='*40}\nProcesando: {nombre_archivo} (Grupo {grupo})\n{'='*40}")
    
    try:
        # 1. Cargar datos
        df = pd.read_csv(DIRECTORIO_DATASETS / nombre_archivo)
        textos = preparar_texto(df, grupo)
        etiquetas = df['label'].values
        
        # 2. Dividir en train/test ANTES de tokenizar
        X_train, X_test, y_train, y_test = train_test_split(
            textos, 
            etiquetas, 
            test_size=TEST_SIZE,
            stratify=etiquetas,
            random_state=RANDOM_STATE
        )
        
        # 3. Tokenización (solo con datos de entrenamiento)
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
        tokenizer.fit_on_texts(X_train)
        
        # 4. Convertir textos a secuencias
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        # 5. Calcular longitud máxima basada en el percentil de train
        longitudes_train = [len(seq) for seq in X_train_seq]
        max_len = int(np.percentile(longitudes_train, PERCENTIL_PADDING))
        print(f"📏 Longitud máxima (percentil {PERCENTIL_PADDING}): {max_len}")
        
        # 6. Aplicar padding con la longitud calculada
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
        
        # 7. Guardar todos los recursos
        guardar_recursos(nombre_archivo, grupo, tokenizer, X_train_pad, X_test_pad, y_train, y_test, max_len)
        
    except Exception as e:
        print(f"❌ Error procesando {nombre_archivo}: {str(e)}")

def preparar_texto(df, grupo):
    """Combina metadatos y texto según el grupo del dataset"""
    # Grupo 1: Metadatos estructurados
    if grupo == 1:
        df['texto_completo'] = (
            "[SENDER] " + df['sender'].fillna('desconocido').str.lower() + 
            " [RECEIVER] " + df['receiver'].fillna('desconocido').str.lower() + 
            " [DATE] " + df['date'].fillna('').astype(str).str.lower() + 
            " [URL] " + df['urls'].fillna(0).astype(str) + 
            " [TEXTO] " + df['texto_completo'].fillna('')
        )
    # Grupo 2: Solo texto preprocesado
    else:
        df['texto_completo'] = df['texto_completo'].fillna('')
    
    # Limpieza final de espacios múltiples
    return df['texto_completo'].str.replace(r'\s+', ' ', regex=True).values

def guardar_recursos(nombre_archivo, grupo, tokenizer, X_train, X_test, y_train, y_test, max_len):
    """Guarda todos los recursos necesarios para el entrenamiento"""
    # Crear directorio para el dataset
    nombre_base = nombre_archivo.split('.')[0]
    dir_dataset = DIRECTORIO_RESULTADOS / f"G{grupo}_{nombre_base}"
    dir_dataset.mkdir(exist_ok=True)
    
    # Guardar tokenizador
    joblib.dump(tokenizer, dir_dataset / "tokenizador.joblib")
    
    # Guardar datos en formato comprimido
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'max_len': max_len,
        'vocab_size': len(tokenizer.word_index)
    }, dir_dataset / "datos.joblib")
    
    print(f"💾 Recursos guardados en: {dir_dataset}")

# Listas de archivos para cada grupo
GRUPOS = {
    1: [
        "CEAS_08.csv", "SpamAssasin.csv", "Nazario.csv", 
        "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
        "TREC_05.csv", "TREC_06.csv", "TREC_07.csv", "grupo_combinado_1.csv"
    ],
    2: [
        "CEAS_08.csv", "SpamAssasin.csv", "Nazario.csv", 
        "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
        "TREC_05.csv", "TREC_06.csv", "TREC_07.csv",
        "Enron.csv", "Ling.csv", "grupo_combinado_2.csv"
    ]
}

if __name__ == "__main__":
    for grupo, archivos in GRUPOS.items():
        for archivo in archivos:
            procesar_dataset(archivo, grupo)
