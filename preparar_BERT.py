import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import joblib

# Configuración
DIRECTORIO_DATASETS = Path("dataset/solo_ingles")
DIRECTORIO_RESULTADOS = Path("dataset/procesados_bert")  # Nuevo directorio para BERT
DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Parámetros BERT
MAX_LENGTH = 512  # Longitud máxima fija para BERT
BERT_MODEL_NAME = 'bert-base-uncased'  # Modelo pre-entrenado

def procesar_dataset_bert(nombre_archivo, grupo):
    """Proceso completo de preprocesamiento para BERT"""
    print(f"\n{'='*40}\nProcesando: {nombre_archivo} (Grupo {grupo})\n{'='*40}")
    
    try:
        # 1. Cargar y preparar datos
        df = pd.read_csv(DIRECTORIO_DATASETS / nombre_archivo)
        textos = preparar_texto(df, grupo)  # Misma función que en LSTM
        etiquetas = df['label'].values
        
        # 2. Dividir en train/test (misma lógica que LSTM)
        X_train, X_test, y_train, y_test = train_test_split(
            textos, 
            etiquetas, 
            test_size=0.2,
            stratify=etiquetas,
            random_state=42
        )
        
        # 3. Tokenización BERT (no necesita entrenamiento)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        # 4. Convertir textos a formato BERT
        train_encodings = tokenizer(
            X_train.tolist(),
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        test_encodings = tokenizer(
            X_test.tolist(),
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # 5. Guardar recursos en formato BERT
        guardar_recursos_bert(
            nombre_archivo,
            grupo,
            train_encodings,
            test_encodings,
            y_train,
            y_test
        )
        
    except Exception as e:
        print(f"❌ Error procesando {nombre_archivo}: {str(e)}")

def guardar_recursos_bert(nombre_archivo, grupo, train_encodings, test_encodings, y_train, y_test):
    """Guarda los datos procesados para BERT"""
    nombre_base = nombre_archivo.split('.')[0]
    dir_dataset = DIRECTORIO_RESULTADOS / f"G{grupo}_{nombre_base}"
    dir_dataset.mkdir(exist_ok=True)
    
    # Guardar inputs y etiquetas en formato comprimido
    np.savez(dir_dataset / 'train_data.npz',
             input_ids=train_encodings['input_ids'],
             attention_mask=train_encodings['attention_mask'],
             labels=y_train)
    
    np.savez(dir_dataset / 'test_data.npz',
             input_ids=test_encodings['input_ids'],
             attention_mask=test_encodings['attention_mask'],
             labels=y_test)
    
    print(f"💾 Datos BERT guardados en: {dir_dataset}")

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
    for grupo, archivos in GRUPOS.items():  # Usar misma estructura GRUPOS
        for archivo in archivos:
            procesar_dataset_bert(archivo, grupo)
