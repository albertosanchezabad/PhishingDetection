import pandas as pd
from pathlib import Path
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import nltk

# Descargar recursos necesarios de NLTK
nltk.download(['punkt', 'averaged_perceptron_tagger_eng', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger', 'wordnet'])

# Configuración de directorios
DIRECTORIO_ENTRADA = Path("dataset/solo_ingles")
DIRECTORIO_SALIDA = Path("dataset/preprocesados_ML")
DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVOS_CSV = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]

def obtener_pos_etiqueta(etiqueta):
    """Mapea etiquetas POS de NLTK a formato WordNet para lematización precisa"""
    mapeo = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return mapeo.get(etiqueta[0], wordnet.NOUN)

def lematizar_texto(texto):
    """
    Lematización avanzada con análisis contextual:
    1. Expande contracciones
    2. Limpia caracteres no alfabéticos
    3. Etiquetado gramatical (POS)
    4. Lematización basada en contexto
    """
    try:
        # Paso 1: Expansión de contracciones ("don't" -> "do not")
        texto = contractions.fix(texto)
        
        # Paso 2: Eliminar URLs, emails y caracteres especiales
        texto = re.sub(r'http\S+', '', texto)
        texto = re.sub(r'\S+@\S+', '', texto)
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        
        # Paso 3: Tokenización
        tokens = word_tokenize(texto.lower())
        
        # Paso 4: Etiquetado gramatical
        tokens_tag = pos_tag(tokens)
        
        # Paso 5: Lematización contextual
        lematizador = WordNetLemmatizer()
        tokens_lemma = [
            lematizador.lemmatize(token, obtener_pos_etiqueta(etiqueta)) 
            for token, etiqueta in tokens_tag
        ]
        
        # Paso 6: Eliminar stopwords después de lematizar
        stop_words = set(stopwords.words('english'))
        tokens_filtrados = [token for token in tokens_lemma if token not in stop_words]
        
        return ' '.join(tokens_filtrados)
    
    except Exception as e:
        print(f"Error procesando texto: {str(e)}")
        return ''

def combinar_campos_texto(df):
    """
    Combina 'subject' y 'body' de forma robusta, manejando:
    - Campos existentes
    - Valores nulos
    - Tipos de datos inconsistentes
    """
    texto_completo = pd.Series('', index=df.index)
    
    if 'subject' in df.columns:
        texto_completo = texto_completo.str.cat(df['subject'].fillna(''), sep=' ')
    
    if 'body' in df.columns:
        texto_completo = texto_completo.str.cat(df['body'].fillna(''), sep=' ')
    
    return texto_completo.str.strip()


def procesar_dataset(archivo):
    """Procesa un archivo CSV completo con lematización avanzada"""
    try:
        # Leer archivo
        df = pd.read_csv(DIRECTORIO_ENTRADA / archivo)
        
        # Combinar campos de texto
        df['texto_completo'] = combinar_campos_texto(df)
        
        # Aplicar lematización
        df['texto_preprocesado'] = df['texto_completo'].apply(lematizar_texto)
        
        # Guardar resultados
        df.to_csv(DIRECTORIO_SALIDA / archivo, index=False)
        print(f"Procesado: {archivo}")
        
    except Exception as e:
        print(f"Error procesando {archivo}: {str(e)}")

# Ejecutar procesamiento
print("\n=== Iniciando preprocesamiento ===")
for archivo in ARCHIVOS_CSV:
    procesar_dataset(archivo)

print("\n=== Proceso completado ===")
print(f"Archivos guardados en: {DIRECTORIO_SALIDA}")
