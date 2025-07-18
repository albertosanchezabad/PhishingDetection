import pandas as pd
from pathlib import Path

# Configurar la ruta del directorio donde están los datasets
DIRECTORIO_DATASETS = Path("dataset/solo_ingles")

# Lista de archivos de datasets
lista_datasets = [
    DIRECTORIO_DATASETS / "CEAS_08.csv",
    DIRECTORIO_DATASETS / "Enron.csv",
    DIRECTORIO_DATASETS / "Ling.csv",
    DIRECTORIO_DATASETS / "SpamAssasin.csv",
    DIRECTORIO_DATASETS / "Nazario.csv",
    DIRECTORIO_DATASETS / "Nazario_5.csv",
    DIRECTORIO_DATASETS / "Nigerian_5.csv",
    DIRECTORIO_DATASETS / "Nigerian_Fraud.csv",
    DIRECTORIO_DATASETS / "TREC_05.csv",
    DIRECTORIO_DATASETS / "TREC_06.csv",
    DIRECTORIO_DATASETS / "TREC_07.csv"
]

# 1. Leer los campos de cada dataset y almacenarlos en un diccionario
campos_por_dataset = dict()

for ruta_dataset in lista_datasets:
    try:
        # Leer solo la cabecera del CSV para obtener los nombres de los campos
        df = pd.read_csv(ruta_dataset, nrows=0)
        campos = set(df.columns)
        campos_por_dataset[ruta_dataset.name] = campos
    except Exception as e:
        print(f"Error leyendo {ruta_dataset.name}: {e}")

# 2. Identificar todos los conjuntos únicos de campos presentes en los datasets
conjuntos_campos_unicos = []
for campos in campos_por_dataset.values():
    if campos not in conjuntos_campos_unicos:
        conjuntos_campos_unicos.append(campos)

# 3. Para cada conjunto de campos, asignar todos los datasets que contienen al menos esos campos
grupos = dict()
for i, campos_grupo in enumerate(conjuntos_campos_unicos, start=1):
    nombre_grupo = f"Grupo_{i}"
    grupos[nombre_grupo] = {
        'campos': campos_grupo,
        'datasets': []
    }
    for nombre_dataset, campos_dataset in campos_por_dataset.items():
        if campos_grupo.issubset(campos_dataset):
            grupos[nombre_grupo]['datasets'].append(nombre_dataset)

# 4. Mostrar los resultados de la clasificación
for nombre_grupo, info_grupo in grupos.items():
    print(f"{nombre_grupo}:")
    print(f"  Campos del grupo: {sorted(info_grupo['campos'])}")
    print(f"  Datasets en el grupo: {info_grupo['datasets']}\n")
