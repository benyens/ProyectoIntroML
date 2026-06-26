import os
import pandas as pd
import zipfile
from tqdm import tqdm

# Configuración de rutas locales relativas
DATA_DIR = "../data/"
METADATA_ZIP = os.path.join(DATA_DIR, "fma_metadata.zip")
AUDIO_ZIP = os.path.join(DATA_DIR, "fma_medium.zip")
AUDIO_EXTRACT_DIR = os.path.join(DATA_DIR, "audio_jazz/")

os.makedirs(AUDIO_EXTRACT_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "fma_metadata"), exist_ok=True)

def descargar_y_filtrar():
    # 1. Descargar Metadatos si no existen
    if not os.path.exists(METADATA_ZIP):
        print("[+] Descargando fma_metadata.zip...")
        os.system(f"curl -o {METADATA_ZIP} https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")
        
    # Descomprimir metadatos nativamente con Python 
    if not os.path.exists(os.path.join(DATA_DIR, 'fma_metadata/tracks.csv')):
        print("[+] Descomprimiendo metadatos...")
        with zipfile.ZipFile(METADATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

    # 2. Leer metadatos para obtener los IDs de Jazz
    print("[+] Identificando pistas de Jazz...")
    tracks = pd.read_csv(os.path.join(DATA_DIR, 'fma_metadata/tracks.csv'), index_col=0, header=[0, 1])
    jazz_tracks = tracks[tracks[('track', 'genre_top')] == 'Jazz']
    jazz_track_ids = jazz_tracks.index.tolist()
    
    # 3. Descargar el zip de audio de 22 GB (si no existe)
    if not os.path.exists(AUDIO_ZIP):
        print("[+] Descargando fma_medium.zip (22 GB)...")
        print("[!] Esto demora un rato por el tamaño del archivo.")
        os.system(f"curl -o {AUDIO_ZIP} https://os.unil.cloud.switch.ch/fma/fma_medium.zip")

    # 4. Extracción con Python 
    print("[+] Iniciando extracción de MP3s de Jazz...")
    
    # Preparamos las rutas de los archivos que queremos pescar dentro del Zip
    archivos_objetivo = set()
    for track_id in jazz_track_ids:
        id_str = str(track_id).zfill(6)
        carpeta = id_str[:3]
        archivos_objetivo.add(f"fma_medium/{carpeta}/{id_str}.mp3")

    archivos_extraidos = 0
    try:
        with zipfile.ZipFile(AUDIO_ZIP, 'r') as zip_ref:
            for archivo_en_zip in tqdm(zip_ref.namelist(), desc="Buscando audios"):
                if archivo_en_zip in archivos_objetivo:
                    # Extraer solo el mp3 y ponerlo directo en nuestra carpeta (sin subcarpetas)
                    nombre_final = os.path.basename(archivo_en_zip)
                    ruta_destino = os.path.join(AUDIO_EXTRACT_DIR, nombre_final)
                    
                    if not os.path.exists(ruta_destino):
                        with zip_ref.open(archivo_en_zip) as source, open(ruta_destino, "wb") as target:
                            target.write(source.read())
                    archivos_extraidos += 1
                    
        print(f"\nÉXITO: Se extrajeron {archivos_extraidos} audios de Jazz en: {AUDIO_EXTRACT_DIR}")
    except Exception as e:
        print(f"Hubo un problema al abrir el zip de audios: {e}")
        print("Asegurarse de que la descarga de fma_medium.zip se completó al 100%.")

if __name__ == "__main__":
    descargar_y_filtrar()