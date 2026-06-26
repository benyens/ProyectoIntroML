import os
import ast
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "../proy/data/"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_jazz/")

def cargar_mapeo_generos():
    genres = pd.read_csv(os.path.join(DATA_DIR, 'fma_metadata/genres.csv'))
    return dict(zip(genres['genre_id'].astype(str), genres['title']))

def extraer_nombre_subgenero(lista_str, genre_map):
    try:
        lista_ids = ast.literal_eval(lista_str)
        nombres = [genre_map.get(str(g), f"ID_{g}") for g in lista_ids if g != 4]
        return nombres[0] if len(nombres) > 0 else "Jazz Generico"
    except:
        return "Jazz Generico"

def procesar_audio(file_path, augment=False):
    """
    Extrae características de una pista. Si augment=True, clona el audio 
    cambiando la afinación (Pitch-Shifting) para NO alterar el ritmo.
    """
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        def extraer_features_de_onda(onda):
            # 1. TABULAR: MFCC + Deltas + Delta-Deltas + Chroma
            mfccs = librosa.feature.mfcc(y=onda, sr=sr, n_mfcc=20)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            chroma = librosa.feature.chroma_stft(y=onda, sr=sr)
            
            rolloff = librosa.feature.spectral_rolloff(y=onda, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(onda)

            feat_tabular = np.hstack([
                np.mean(mfccs.T, axis=0),
                np.mean(delta_mfccs.T, axis=0),
                np.mean(delta2_mfccs.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(rolloff),   # 1 feature
                np.mean(zcr)        # 1 feature
            ])
            
            # 2. VISUAL: Espectrograma (CNN)
            mel = librosa.feature.melspectrogram(y=onda, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
           
            # mel_db tiene rango [-80, 0] aprox. Lo normalizamos a [0, 1]
            mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            mel_resized = tf.image.resize(mel_db_norm[..., np.newaxis], [128, 128]).numpy()
            
            return feat_tabular, mel_resized

        # Extraer el audio original
        tab_orig, mel_orig = extraer_features_de_onda(y)
        resultados = [(tab_orig, mel_orig)]
        
        # ==========================================
        # NUEVO AUMENTO: PITCH-SHIFTING (Tono)
        # ==========================================
        if augment:
            # Versión Aguda (Sube 2 semitonos, ritmo intacto)
            y_high = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            tab_h, mel_h = extraer_features_de_onda(y_high)
            resultados.append((tab_h, mel_h))
            
            # Versión Grave (Baja 2 semitonos, ritmo intacto)
            y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
            tab_l, mel_l = extraer_features_de_onda(y_low)
            resultados.append((tab_l, mel_l))
            
        return resultados
    except Exception as e:
        return []

def main():
    print("[+] Cargando metadatos y configurando etiquetas limpias...")
    genre_map = cargar_mapeo_generos()
    tracks = pd.read_csv(os.path.join(DATA_DIR, 'fma_metadata/tracks.csv'), index_col=0, header=[0, 1])
    jazz_tracks = tracks[tracks[('track', 'genre_top')] == 'Jazz'].copy()
    jazz_tracks['etiqueta_limpia'] = jazz_tracks[('track', 'genres')].astype(str).apply(lambda x: extraer_nombre_subgenero(x, genre_map))

    X_tabular, X_cnn, y_labels = [], [], []

    print("[+] Procesando archivos de audio y aplicando Aumento de Datos...")
    for root, _, files in os.walk(AUDIO_DIR):
        for file in tqdm(files):
            if file.endswith(".mp3"):
                track_id = int(file.replace('.mp3', ''))
                file_path = os.path.join(root, file)
                
                try:
                    etiqueta = jazz_tracks.loc[track_id, 'etiqueta_limpia']
                    
                    # Decisión Inteligente: Solo aumentamos si NO es el aburrido Jazz Genérico
                    aplicar_aumento = (etiqueta != "Jazz Generico")
                    
                    # procesar_audio ahora devuelve una lista de tuplas (hasta 3 versiones de la canción)
                    features_extraidos = procesar_audio(file_path, augment=aplicar_aumento)
                    
                    for tab, mel in features_extraidos:
                        X_tabular.append(tab)
                        X_cnn.append(mel)
                        y_labels.append(etiqueta) # Etiquetamos los clones físicos con el mismo género
                except:
                    pass

    X_tabular = np.array(X_tabular)
    X_cnn = np.array(X_cnn)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    print(f"[+] Guardando matrices potenciadas (Tamaño tabular: {X_tabular.shape[1]} features) en {DATA_DIR}...")
    np.save(os.path.join(DATA_DIR, "X_tabular.npy"), X_tabular)
    np.save(os.path.join(DATA_DIR, "X_cnn.npy"), X_cnn)
    np.save(os.path.join(DATA_DIR, "y_encoded.npy"), y_encoded)
    np.save(os.path.join(DATA_DIR, "classes.npy"), le.classes_)
    
    print("[!] Matrices guardadas con éxito. Modelos listos para entrenar.")

if __name__ == "__main__":
    main()