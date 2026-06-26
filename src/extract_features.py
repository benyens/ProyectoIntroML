import os
import ast
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "../data/"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_jazz/")

def cargar_mapeo_generos():
    """Carga el diccionario para transformar IDs numéricos a nombres reales."""
    genres = pd.read_csv(os.path.join(DATA_DIR, 'fma_metadata/genres.csv'))
    return dict(zip(genres['genre_id'].astype(str), genres['title']))

def extraer_nombre_subgenero(lista_str, genre_map):
    try:
        lista_ids = ast.literal_eval(lista_str)
        nombres = [genre_map.get(str(g), f"ID_{g}") for g in lista_ids if g != 4]
        return nombres[0] if len(nombres) > 0 else "Jazz Generico"
    except:
        return "Jazz Generico"

def procesar_audio(file_path):
    try:
        # Cargar los 30 segundos estándar
        y, sr = librosa.load(file_path, duration=30)
        
        # 1. MFCC + Chroma para el enfoque Tabular (Random Forest)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        feat_tabular = np.hstack([mfccs, chroma])
        
        # 2. Espectrograma de Mel para el enfoque de Deep Learning (CNN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], [128, 128]).numpy()
        
        return feat_tabular, mel_resized
    except:
        return None, None

def main():
    print("[+] Cargando metadatos y configurando etiquetas limpias...")
    genre_map = cargar_mapeo_generos()
    tracks = pd.read_csv(os.path.join(DATA_DIR, 'fma_metadata/tracks.csv'), index_col=0, header=[0, 1])
    jazz_tracks = tracks[tracks[('track', 'genre_top')] == 'Jazz'].copy()
    jazz_tracks['etiqueta_limpia'] = jazz_tracks[('track', 'genres')].astype(str).apply(lambda x: extraer_nombre_subgenero(x, genre_map))

    X_tabular, X_cnn, y_labels = [], [], []

    print("[+] Procesando archivos de audio locales...")
    for root, _, files in os.walk(AUDIO_DIR):
        for file in tqdm(files):
            if file.endswith(".mp3"):
                track_id = int(file.replace('.mp3', ''))
                file_path = os.path.join(root, file)
                
                try:
                    etiqueta = jazz_tracks.loc[track_id, 'etiqueta_limpia']
                    tab, mel = procesar_audio(file_path)
                    
                    if tab is not None:
                        X_tabular.append(tab)
                        X_cnn.append(mel)
                        y_labels.append(etiqueta)
                except:
                    pass

    # Convertir a arreglos de NumPy
    X_tabular = np.array(X_tabular)
    X_cnn = np.array(X_cnn)
    
    # Codificar etiquetas string a enteros
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    print(f"[+] Guardando matrices pre-procesadas en {DATA_DIR}...")
    np.save(os.path.join(DATA_DIR, "X_tabular.npy"), X_tabular)
    np.save(os.path.join(DATA_DIR, "X_cnn.npy"), X_cnn)
    np.save(os.path.join(DATA_DIR, "y_encoded.npy"), y_encoded)
    np.save(os.path.join(DATA_DIR, "classes.npy"), le.classes_)
    
    print("[!] Matrices guardadas con éxito. Listas para el entrenamiento.")

if __name__ == "__main__":
    main()