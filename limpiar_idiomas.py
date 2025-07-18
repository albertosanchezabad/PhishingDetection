import pandas as pd
from pathlib import Path
import numpy as np
import csv
import sys
from colorama import Fore, Style, init
from langdetect import detect, DetectorFactory

# Configuración inicial
init(autoreset=True)
csv.field_size_limit(100000000)
DetectorFactory.seed = 0  # Para resultados consistentes

# Configuración de rutas
DIRECTORIO_DATASETS = Path("dataset")
CARPETA_SALIDA = DIRECTORIO_DATASETS / "solo_ingles"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]

def es_ingles(texto):
    """Detecta si el texto está en inglés con manejo de errores"""
    try:
        # Verificar texto válido
        if not texto or len(texto.strip()) < 10:
            return False
            
        # Detectar idioma
        return detect(texto) == 'en'
    except:
        return False

def procesar_dataset(archivo):
    """Procesa un dataset y elimina registros no ingleses"""
    ruta = DIRECTORIO_DATASETS / "preprocesamiento_general" / archivo
    try:
        df = pd.read_csv(ruta, engine='python', on_bad_lines='warn')
    except Exception as e:
        print(f"{Fore.RED}Error leyendo {archivo}: {str(e)}{Style.RESET_ALL}")
        return None

    # Verificar campos necesarios
    columnas = df.columns.str.lower()
    tiene_subject = 'subject' in columnas
    tiene_body = 'body' in columnas

    if not tiene_subject and not tiene_body:
        print(f"{Fore.YELLOW}{archivo}: No tiene 'subject' ni 'body'{Style.RESET_ALL}")
        return None

    # Concatenar subject y body
    subject_col = next((col for col in df.columns if col.lower() == 'subject'), '')
    body_col = next((col for col in df.columns if col.lower() == 'body'), '')
    
    df['texto_completo'] = df[subject_col].fillna('') + ' ' + df[body_col].fillna('')
    
    # Filtrar por idioma
    registros_iniciales = len(df)
    mascara = df['texto_completo'].apply(es_ingles)
    df_limpio = df[mascara].copy()
    registros_finales = len(df_limpio)
    
    # Guardar resultado
    ruta_salida = CARPETA_SALIDA / archivo
    df_limpio.to_csv(ruta_salida, index=False)

    return {
        'archivo': archivo,
        'inicial': registros_iniciales,
        'final': registros_finales,
        'eliminados': registros_iniciales - registros_finales,
        'texto_ejemplo': df['texto_completo'].iloc[0][:100] + '...' if not df.empty else ''
    }

def generar_reporte(resultado):
    """Genera un reporte detallado en consola"""
    print(f"\n{Fore.CYAN}=== {resultado['archivo']} ===")
    print(f"{Fore.WHITE}Registros iniciales: {resultado['inicial']}")
    print(f"Registros en inglés: {resultado['final']}")
    print(f"Porcentaje conservado: {(resultado['final']/resultado['inicial'])*100:.2f}%")
    print(f"{Fore.YELLOW}Registros eliminados: {resultado['eliminados']}")
    print(f"{Fore.GREEN}Ejemplo de texto analizado: {resultado['texto_ejemplo']}")
    print(Style.RESET_ALL + "-"*60)

# Procesar todos los datasets
print(f"{Fore.BLUE}\nIniciando filtrado por idioma inglés...\n")
for dataset in DATASETS:
    resultado = procesar_dataset(dataset)
    if resultado:
        generar_reporte(resultado)

print(f"{Fore.GREEN}\nProceso completado. Datos en inglés en: {CARPETA_SALIDA}")
