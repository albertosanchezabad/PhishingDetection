import pandas as pd
from pathlib import Path
import numpy as np
import csv
import sys
from colorama import Fore, Style, init

# Inicializar colorama para colores en consola
init(autoreset=True)
csv.field_size_limit(100000000)

# Configuración de rutas
DIRECTORIO_DATASETS = Path("dataset/iniciales")
CARPETA_SALIDA = DIRECTORIO_DATASETS / "preprocesamiento_general"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "CEAS_08.csv", "Enron.csv", "Ling.csv", "SpamAssasin.csv",
    "Nazario.csv", "Nazario_5.csv", "Nigerian_5.csv", "Nigerian_Fraud.csv",
    "TREC_05.csv", "TREC_06.csv", "TREC_07.csv"
]

def limpiar_etiqueta(valor):
    """
    Normaliza los valores de etiqueta a 0 o 1 enteros.
    Convierte 0, 0.0, '0', '0.0' a 0; 1, 1.0, '1', '1.0' a 1.
    Cualquier otro valor se convierte en np.nan.
    """
    try:
        # Primero convertir a float (para manejar '0.0', 0.0, etc.)
        num = float(valor)
        if num == 0.0:
            return 0
        elif num == 1.0:
            return 1
        else:
            return np.nan
    except:
        return np.nan

def procesar_dataset(archivo):
    """Procesa un único dataset y devuelve estadísticas"""
    ruta = DIRECTORIO_DATASETS / archivo
    try:
        df = pd.read_csv(ruta, engine='python', on_bad_lines='warn')
    except Exception as e:
        print(f"{Fore.RED}Error leyendo {archivo}: {str(e)}{Style.RESET_ALL}")
        return None

    if 'label' not in df.columns:
        print(f"{Fore.YELLOW}{archivo}: No tiene columna 'label'{Style.RESET_ALL}")
        return None

    registros_iniciales = len(df)

    # Limpieza y normalización
    df['label'] = df['label'].apply(limpiar_etiqueta)

    # Filtrar registros válidos
    df_limpio = df[df['label'].isin([0, 1])].copy()
    df_limpio['label'] = df_limpio['label'].astype(int)

    registros_finales = len(df_limpio)

    # Detección de valores inválidos
    invalidos = df[~df['label'].isin([0, 1])]
    conteo_invalidos = len(invalidos)

    # Guardar resultado
    ruta_salida = CARPETA_SALIDA / archivo
    df_limpio.to_csv(ruta_salida, index=False)

    # Calcular distribución
    distribucion = df_limpio['label'].value_counts(normalize=True).mul(100).round(2)

    return {
        'archivo': archivo,
        'inicial': registros_iniciales,
        'final': registros_finales,
        'invalidos': conteo_invalidos,
        'distribucion': distribucion,
        'ejemplos_invalidos': invalidos['label'].value_counts().head(3).to_dict()
    }

def generar_reporte(resultado):
    """Genera un reporte detallado en consola"""
    print(f"\n{Fore.CYAN}=== {resultado['archivo']} ===")
    print(f"{Fore.WHITE}Registros iniciales: {resultado['inicial']}")
    print(f"Registros válidos: {resultado['final']}")
    print(f"Porcentaje conservado: {(resultado['final']/resultado['inicial'])*100:.2f}%")
    
    if resultado['invalidos'] > 0:
        print(f"{Fore.YELLOW}Registros con labels inválidos: {resultado['invalidos']}")
        for valor, cantidad in resultado['ejemplos_invalidos'].items():
            print(f"  - Valor inválido {valor}: {cantidad} registros")
    
    print(f"{Fore.GREEN}Distribución final:")
    for valor, porcentaje in resultado['distribucion'].items():
        print(f"  - Label {valor}: {porcentaje}%")
    print(Style.RESET_ALL + "-"*60)

# Procesar todos los datasets
print(f"{Fore.BLUE}\nIniciando procesamiento de {len(DATASETS)} datasets...\n")
for dataset in DATASETS:
    resultado = procesar_dataset(dataset)
    if resultado:
        generar_reporte(resultado)

print(f"{Fore.GREEN}\nProceso completado. Datos limpios en: {CARPETA_SALIDA}")
