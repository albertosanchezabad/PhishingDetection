import csv
import re
import random
import math
import enchant  # Biblioteca para verificar ortografía
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logging
from email.utils import parseaddr
import dateutil.parser
import sys
import os
import pandas as pd

# Configura el logging para mostrar mensajes DEBUG (cuando se hagan pruebas con pocos samples) o WARNING
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

# Aumentar límite de tamaño de campo CSV
csv.field_size_limit(100000000)

# Inicializar diccionario en inglés para verificar ortografía
dict_en = enchant.Dict("en_GB-ise")

# Diccionario de zonas horarias comunes (se puede ampliarlo según tus necesidades)
tzinfos = {
    "UT": 0, "UTC": 0, "GMT": 0,
    "EST": -5*3600, "EDT": -4*3600,
    "CST": -6*3600, "CDT": -5*3600,
    "MST": -7*3600, "MDT": -6*3600,
    "PST": -8*3600, "PDT": -7*3600,
    "HST": -10*3600, "AKST": -9*3600, "AKDT": -8*3600,
    "CET": 1*3600, "CEST": 2*3600,
    "EET": 2*3600, "EEST": 3*3600,
    "MET": 1*3600, "MESZ": 2*3600,
}

# Listas negras para URLs
dominios_sospechosos = [
    "bit.ly", "t.co", "tinyurl.com", "is.gd", "soo.gd", "ow.ly", "buff.ly",
    "rebrand.ly", "shorte.st", "cutt.ly", "bl.ink", "snip.ly", "cli.re",
    "goo.gl", "1drv.ms", "fb.me", "lnkd.in", "t.me", "wa.me",
    "adf.ly", "bc.vc", "u.to", "qr.net", "v.gd", "tr.im", "x.co", "rb.gy", "t2m.io"
]

dominios_utilizados_phishing = [
    # Dominios de alto riesgo y frecuentemente usados en phishing
    r'[\w-]+\.onion',  # Dark web / Tor
    r'[\w-]+\.ru',     # Rusia
    r'[\w-]+\.cn',     # China
    r'[\w-]+\.kp',     # Corea del Norte
    r'[\w-]+\.ir',     # Irán
    r'[\w-]+\.su',     # Antiguo dominio soviético, aún activo
    r'[\w-]+\.ng',     # Nigeria (asociado a fraudes)
    
    # Dominios genéricos usados en phishing por ser baratos o anónimos
    r'[\w-]+\.xyz', r'[\w-]+\.top', r'[\w-]+\.club', r'[\w-]+\.online',
    r'[\w-]+\.site', r'[\w-]+\.buzz', r'[\w-]+\.space', r'[\w-]+\.live',
    r'[\w-]+\.info', r'[\w-]+\.vip', r'[\w-]+\.best', r'[\w-]+\.gq',
    r'[\w-]+\.ml', r'[\w-]+\.cf', r'[\w-]+\.ga', r'[\w-]+\.tk',  # Dominio gratuitos de Freenom

    # Dominios de correo desechable usados para fraudes
    r'temp-mail\.org', r'mohmal\.com', r'10minutemail\.com',
    r'mytemp\.email', r'guerrillamail\.com', r'throwawaymail\.com',

    # Dominios fraudulentos de bancos y pagos (versiones falsas de servicios reales)
    r'payp[a@]l-?secure\.com', r'paypal-verification\.com',
    r'login-bankofamerica\.com', r'bankofamerica-security\.com',
    r'chaselogin\.com', r'citi-verification\.com',

    # Versiones sospechosas de grandes plataformas (Google, Microsoft, etc.)
    r'googl[e3]\.verification\.com', r'microsoft-support\.site',
    r'appl[e3]-id-verification\.com', r'fac3b00k-login\.com'
]

# Palabras clave refinadas basadas en investigaciones recientes
palabras_asunto = [
    "urgent", "action required", "verify now", "account locked", "suspended",
    "important notice", "security update", "compromised", "unusual activity",
    "payment required", "login attempt", "confirm identity", "verify identity"
]

palabras_cuerpo = [
    "click here", "failure to comply", "account will be suspended",
    "update your account details", "confirm your information", "verify your identity",
    "validate your account", "login now to secure", "unauthorized access",
    "click the link below", "security purposes", "limited time", "free"
]

# Frases sospechosas relacionadas con estafas específicas (basadas en investigaciones)
frases_estafa = [
    "lottery winner", "inheritance fund", "prince needs help", "million dollars",
    "urgent transfer request", "claim your prize", "overseas transaction",
    "business proposal", "unclaimed funds", "investment opportunity"
]

# Términos financieros sospechosos
terminos_financieros = [
    "usd", "euro", "pound", "transfer", "bank account", "wire transfer", 
    "western union", "moneygram", "bitcoin", "cryptocurrency", "payment"
]

# Expresión regular generalizada para detectar ofuscaciones
patron_obfuscacion_generico = re.compile(
    r'\b'  # Delimitar palabras
    r'(?:'  # Iniciar grupo no capturante
    r'[a-z]+[A-Z][a-zA-Z]*|'  # Minúsculas seguidas de mayúsculas (no al inicio)
    r'[a-z]+\d|'  # Minúsculas seguidas de números
    r'[a-z]+[@%$#!]|'  # Minúsculas seguidas de caracteres especiales
    r'[A-Z]+\d|'  # Mayúsculas seguidas de números
    r'[A-Z]+[@%$#!]|'  # Mayúsculas seguidas de caracteres especiales
    r'(?:[a-zA-Z]*[@%$#!]+[0-9]*){3,}'  # Combinación de al menos tres tipos diferentes
    r')\b'  # Delimitar palabras
)

def detectar_obfuscacion(texto):
    # 1.1) Sanear y tokenizar (alphanum + símbolos)
    texto = texto.replace('\r',' ').replace('\n',' ')
    tokens = re.findall(r'\b[^\s]{1,50}\b', texto)  # palabras de hasta 50 chars

    obfus = []
    for tok in tokens:
        # 1.2) Ignora tokens demasiado cortos
        if len(tok) < 4:
            continue
        # 1.3) Aplica patrón “light” a cada token
        if patron_obfuscacion_generico.search(tok):
            obfus.append(tok)
    return obfus


# Función para detectar palabras mal escritas
def detectar_palabras_mal_escritas(texto, dict_en):
    # Dividir el texto en palabras sin restricción de longitud
    palabras = re.findall(r'\b[a-zA-Z]{3,}\b', texto.lower())
    
    # Filtrar palabras mal escritas
    palabras_mal_escritas = [palabra for palabra in palabras if not dict_en.check(palabra)]
    
    return palabras_mal_escritas

# Función heurística mejorada basada en investigaciones recientes
def correo_sospechoso(grupo, asunto, cuerpo, remitente="", destinatario="", fecha="", urls=""):
    puntuacion = 0  # Inicializar puntuación heurística

    logging.debug(f"\nAnalizando correo: Grupo {grupo}")
    logging.debug(f" - Asunto: {asunto}")
    logging.debug(f" - Remitente: {remitente}")
    logging.debug(f" - Destinatario: {destinatario}")
    logging.debug(f" - Fecha: {fecha}") 
    logging.debug(f" - Cuerpo: {cuerpo}...")

    texto_completo = (asunto + " " + cuerpo).lower()
    
    # 1. Análisis de asunto (mayor peso según investigaciones)
    if any(palabra in asunto.lower() for palabra in palabras_asunto):
        puntuacion += 4
        logging.debug("   +4: Asunto contiene palabras clave sospechosas.")
    
    # 2. Frecuencia de palabras clave sospechosas en cuerpo (técnica validada en estudios)
    suma = sum(1 for palabra in palabras_cuerpo if palabra in cuerpo.lower())
    puntuacion += suma
    logging.debug("   +" + str(suma) + ": Cuerpo contiene palabras clave sospechosas.")

    # 3. Detección de estafas específicas (alto indicador de phishing)
    if any(frase in cuerpo.lower() for frase in frases_estafa):
        puntuacion += 4
        logging.debug("   +4: Estafa específica.")
    
    # 4. Términos financieros en contexto sospechoso
    if any(termino in cuerpo.lower() for termino in terminos_financieros):
        puntuacion += 2
        logging.debug("   +2: Términos financieros sospechosos.")
    
    # Análisis de URLs (factor crítico según investigaciones)
    urls = re.findall(r'https?://[^\s]+', cuerpo.lower())

    dominios_analizados = set()  # Conjunto para registrar dominios analizado

    urls = set(urls) # Convertir la lista de URLs en un set para eliminar duplicados
    
    for url in urls:
        # Extraer dominio
        dominio_match = re.findall(r'https?://(?:www\.)?([^/]+)', url)
        if dominio_match:
            dominio = dominio_match[0]

            # Analizar dominio **solo si no ha sido procesado antes**
            if dominio not in dominios_analizados:
                dominios_analizados.add(dominio)  # Marcar dominio como analizado
            
                # 5. Dominio acortador
                if any(d in dominio for d in dominios_sospechosos):
                    puntuacion += 2
                    logging.debug("   +2: Dominio de URL de un acortador. " + dominio)

                # 6. Dominios phishings
                for patron in dominios_utilizados_phishing:
                    if re.fullmatch(patron, dominio):
                        puntuacion += 4
                        logging.debug("   +4: Dominio utilizados phishing. " + dominio)
                
                # 7. Dominio de URL muy largo
                if len(dominio) > 30:
                    puntuacion += 2
                    logging.debug("   +2: Dominio de URL muy largo. " + dominio)

                # 8. Dominio con más subdominios
                if dominio.count('.') == 3:
                    puntuacion += 2
                    logging.debug("   +2: Dominio de URL con 2 subdominios. " + dominio)

                elif dominio.count('.') == 4:
                    puntuacion += 3
                    logging.debug("   +3: Dominio de URL con 3 subdominios. " + dominio)

                elif dominio.count('.') > 5:
                    puntuacion += 4
                    logging.debug("   +4: Dominio de URL con más de 4 subdominios. " + dominio)

            # 9. URL con carácteres raros
            if re.search(r'[@\$#!*]', url):
                puntuacion += 1
                logging.debug("   +1: URL con carácteres raros." + url)

            # 10. URL con punycode
            if 'xn--' in url:
                    puntuacion += 2
                    logging.debug(f"   +2: URL con Punycode ({url}).")
    
    # 11. Detección de palabras mal escritas (técnica heurística avanzada)
    palabras_mal_escritas = detectar_palabras_mal_escritas(texto_completo, dict_en)
    num_palabras_mal_escritas = len(palabras_mal_escritas)
    if num_palabras_mal_escritas > 0:
        puntuacion += min(math.ceil(num_palabras_mal_escritas/2),2)
        logging.debug("   +" + str(min(math.ceil(num_palabras_mal_escritas/2),2)) +": Palabras mal escritas." + str(palabras_mal_escritas))
    
    # 12. Patrones de obfuscación
    obfuscaciones = detectar_obfuscacion(cuerpo)
    num_obfuscaciones = len(obfuscaciones)
    if num_obfuscaciones > 0:
        puntuacion += min(math.ceil(num_obfuscaciones/3),1)
        logging.debug("   +" +str(min(math.ceil(num_obfuscaciones/3),1)) +": Patrones de obfuscacion." + str(obfuscaciones))
    
    # 13. Uso excesivo de mayúsculas (indicador secundario)
    if re.search(r'\b[A-Z]{4,}\b', cuerpo):
        puntuacion += 2
        logging.debug("   +2: Excesivo uso de mayúsculas.")
    
    # Umbral basado en investigaciones para grupo 2
    if grupo == 2:
        return puntuacion > 5

    else:
        
        # 14. Si se muestra el nombre completo en el Remitente
        name, email_addr = parseaddr(remitente)
        if name:
            puntuacion += 1
            logging.debug("   +1: Display name en remitente.")


        if email_addr and '@' in email_addr:

            # 15. Punycode en remitente
            if 'xn--' in email_addr:
                puntuacion += 2
                logging.debug(f"   +2: Punycode en remitente ({dom_r}).")

            dom_r = email_addr.split('@')[-1].lower()

            # 16. Dominio remitente phishings 
            for patron in dominios_utilizados_phishing:
                if re.fullmatch(patron, dom_r):
                    puntuacion += 4
                    logging.debug(f"   +4: Dominio de remitente utilizados phishing ({dom_r}).")

        # 17. Varios destinatarios
        recs = re.split(r'[;,]', destinatario)
        recs = [r.strip() for r in recs if r.strip()]
        if len(recs) > 1:
            puntuacion += 1
            logging.debug("   +1: Varios destinatarios.")

        fin_sem, fuera_h = False, False
        try:
            dt = dateutil.parser.parse(fecha, tzinfos=tzinfos)

            # 18. Correo enviado en fin de semana
            if dt.weekday() >= 5:
                fin_sem = True
                puntuacion += 2
                logging.debug("   +2: Correo enviado en fin de semana.")

            # 19. Correo enviado fuera de horario laboral
            elif dt.hour < 8 or dt.hour > 18:
                fuera_h = True
                puntuacion += 1
                logging.debug("   +1: Correo fuera de horario laboral.")
        except Exception:
            logging.debug("   0: No se pudo parsear la fecha.")

        # Umbral grupo 1
        return puntuacion > 6

# Dataset de ejemplo (Grupo 1)
grupo_1 = [
    r"dataset\solo_ingles\CEAS_08.csv",
    r"dataset\solo_ingles\SpamAssasin.csv",
    r"dataset\solo_ingles\Nazario.csv",
    r"dataset\solo_ingles\Nazario_5.csv",
    r"dataset\solo_ingles\Nigerian_5.csv",
    r"dataset\solo_ingles\Nigerian_Fraud.csv",
    r"dataset\solo_ingles\TREC_05.csv",
    r"dataset\solo_ingles\TREC_06.csv",
    r"dataset\solo_ingles\TREC_07.csv",
    r"dataset\solo_ingles\grupo_combinado_1.csv"
]

# Dataset de ejemplo (Grupo 2)
grupo_2 = [
    r"dataset\solo_ingles\CEAS_08.csv",
    r"dataset\solo_ingles\Enron.csv",
    r"dataset\solo_ingles\Ling.csv",
    r"dataset\solo_ingles\SpamAssasin.csv",
    r"dataset\solo_ingles\Nazario.csv",
    r"dataset\solo_ingles\Nazario_5.csv",
    r"dataset\solo_ingles\Nigerian_5.csv",
    r"dataset\solo_ingles\Nigerian_Fraud.csv",
    r"dataset\solo_ingles\TREC_05.csv",
    r"dataset\solo_ingles\TREC_06.csv",
    r"dataset\solo_ingles\TREC_07.csv",
    r"dataset\solo_ingles\grupo_combinado_2.csv"
]

def analizar_dataset(archivo,grupo):
    print(f"\nAnalizando dataset: {archivo}")

    # Primera pasada rápida para contar registros reales (CSV)
    try:
        with open(archivo, newline='', encoding='utf-8') as f_count:
            lector_count = csv.DictReader(f_count)
            total_rows = sum(1 for _ in lector_count)
    except FileNotFoundError:
        print(f"Error: Archivo {archivo} no encontrado.")
        return

    
    y_true, y_pred = [], []
    total, phishing, legitimo = 0, 0, 0
    
    try:

        df = pd.read_csv(
            archivo,
            engine='python',
            sep=',',
            on_bad_lines='skip',      # descarta las líneas que no encajan
            quoting=csv.QUOTE_MINIMAL,  # lee sin esperar cierre excesivo
        )
        

        filas_procesadas = 0
        barra_long = 40  # ancho de la barra en caracteres

        df['subject'] = df['subject'].fillna('')
        df['body']    = df['body'].fillna('')

        if grupo == 1:
            df['sender']    = df['sender'].fillna('')
            df['receiver']    = df['receiver'].fillna('')
            df['date']    = df['date'].fillna('')   
            df['urls']    = df['urls'].fillna(0)

        for idx, fila in df.iterrows():

            filas_procesadas += 1

            #if filas_procesadas != 295 and filas_procesadas != 294:
            #    continue

            # Progress bar
            porcentaje = filas_procesadas / total_rows
            llenado = int(porcentaje * barra_long)
            barra = '[' + '=' * llenado + ' ' * (barra_long - llenado) + ']'
            sys.stdout.write(f"\r {barra} {filas_procesadas}/{total_rows} ({porcentaje:.0%})")
            sys.stdout.flush()

            asunto = fila['subject']
            cuerpo = fila['body'].replace('\r',' ').replace('\n',' ')

            etiqueta_real = int(fila.get('label', '0'))
            
            if grupo == 1:
                remitente = fila['sender']
                destinatario = fila['receiver']
                fecha = fila['date']
                urls = fila['urls']

                sospechoso = correo_sospechoso(grupo, asunto, cuerpo, remitente, destinatario, fecha, urls)

            else:
                sospechoso = correo_sospechoso(grupo, asunto, cuerpo)

            logging.debug(f" - Clasificado como {'Phishing' if sospechoso else 'Legítimo'}")
            logging.debug(f" - Es un {'Phishing' if etiqueta_real==1 else 'Legítimo'}")
            logging.debug("----------------------")

            y_true.append(etiqueta_real)
            y_pred.append(1 if sospechoso else 0)

            total += 1
            if etiqueta_real == 1:
                phishing += 1
            else:
                legitimo += 1

        print()  # Salto tras la barra

        # Métricas de evaluación
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) else 0

        # Mostrar resultados integrados claramente
        print("Distribución de clases:")
        print(f" - Total correos: {total}")
        print(f" - Phishing: {phishing} ({phishing/total:.2%})")
        print(f" - Legítimos: {legitimo} ({legitimo/total:.2%})")

        print("\nResultados heurísticos:")
        print(f" - Precision: {precision:.2f}")
        print(f" - Recall (Tasa detección): {recall:.2f}")
        print(f" - F1-Score: {f1:.2f}")
        print(f" - Accuracy: {accuracy:.2f}")
        print(f" - False Positive Rate (FPR): {fpr:.2f}")

    except FileNotFoundError:
        print(f"Error: Archivo {archivo} no encontrado.")

    return {
        "dataset": os.path.basename(archivo),
        "grupo": grupo,
        "total": total,
        "phishing": phishing,
        "legitimo": legitimo,
        "pct_phishing": phishing/total,
        "pct_legitimo": legitimo/total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr
    }

# Fichero de salida
resultados_csv = "resultados/heuristicas/resultados_heuristicas.csv"
campos = [
    "dataset", "grupo", "total", "phishing", "legitimo",
    "pct_phishing", "pct_legitimo",
    "precision", "recall", "f1", "accuracy", "fpr"
]

# Escribir cabecera (sobrescribe si ya existía)
with open(resultados_csv, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=campos)
    writer.writeheader()


# Ejecución con Grupo 1
for archivo in grupo_1:
    resultado = analizar_dataset(archivo,1)
    if resultado:
        with open(resultados_csv, "a", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=campos)
            writer.writerow(resultado)
        print(f"→ Resultados volcados en {resultados_csv}\n")


# Ejecución con Grupo 2
for archivo in grupo_2:
    resultado = analizar_dataset(archivo,2)
    if resultado:
        with open(resultados_csv, "a", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=campos)
            writer.writerow(resultado)
        print(f"→ Resultados volcados en {resultados_csv}\n")

