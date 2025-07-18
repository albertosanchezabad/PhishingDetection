import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Configuración de directorios
DIRECTORIO_ENTRADA = Path("dataset/preprocesados_ML")
DIRECTORIO_SALIDA = Path("dataset/tfidf_models")
DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVOS_CSV = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario_5.csv", "Nigerian_5.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv", "grupo_combinado_1.csv", "grupo_combinado_2.csv"
]

def aplicar_tfidf(archivo):
    """Aplica TF-IDF y guarda los resultados"""
    try:
        # 1. Cargar datos preprocesados
        df = pd.read_csv(DIRECTORIO_ENTRADA / archivo, low_memory=False)
        
        # 2. Seleccionar columna de texto preprocesado
        textos = df['texto_preprocesado'].fillna('')
        labels = df['label'] if 'label' in df.columns else None
        
        # 3. Configurar TF-IDF (parámetros clave)
        tfidf = TfidfVectorizer(
            max_features=5000,          # Limitar a 5000 términos más relevantes
            min_df=2,                   # Ignorar términos en <2 documentos
            max_df=0.95,                # Ignorar términos en >95% documentos
            ngram_range=(1, 2),         # Considerar unigramas y bigramas
            token_pattern=r'(?u)\b[a-zA-Z0-9_]{3,}\b'  # Tokens de al menos 3 caracteres
        )
        
        # 4. Aplicar TF-IDF
        X = tfidf.fit_transform(textos)
        
        # 5. Dividir en train/test (si hay labels)
        if labels is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, stratify=labels, random_state=42
            )
            
            # Guardar splits
            joblib.dump((X_train, X_test, y_train, y_test), 
                       DIRECTORIO_SALIDA / f"{archivo.split('.')[0]}_split.joblib")
        
        # 6. Guardar modelo TF-IDF y datos transformados
        joblib.dump(tfidf, DIRECTORIO_SALIDA / f"{archivo.split('.')[0]}_tfidf.joblib")
        joblib.dump(X, DIRECTORIO_SALIDA / f"{archivo.split('.')[0]}_features.joblib")
        
        print(f"✅ {archivo} procesado (dimensiones: {X.shape})")
        
    except Exception as e:
        print(f"❌ Error en {archivo}: {str(e)}")

# Ejecutar para todos los datasets
print("=== Iniciando vectorización TF-IDF ===")
for archivo in ARCHIVOS_CSV:
    aplicar_tfidf(archivo)

print("\n=== Proceso completado ===")
print(f"Modelos y datos guardados en: {DIRECTORIO_SALIDA}")
